from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_continual_summary_metrics
from evaluation.performance_matrix import PerformanceMatrix
from task_generation.task_builder import ContinualTask
from .checkpoint import CheckpointManager
from .data_loader import batch_samples
from .logger import StreamingMetricsLogger, generate_conclusion_report

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from transformers import get_linear_schedule_with_warmup
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SequentialFTTrainer:
    """Trainer executing sequential fine-tuning across Continual Tasks.

    Strict Continual Learning Principles:
        - Exactly ONE model instance fine-tuned sequentially from Task 1 to Task N.
        - Backbone and classifier weights are NEVER reinitialized between tasks.
        - Training on Task t uses ONLY Task t's training samples (M=0, no replay).
        - After completing Task t, evaluates immediately on all observed tasks j in [0..t].
        - Each evaluation is immediately streamed to metrics.jsonl.
    """

    def __init__(
        self,
        config: dict[str, Any],
        tasks: list[ContinualTask],
        model: Any,
        evaluator: Evaluator | None = None,
        results_dir: Path | str | None = None,
    ) -> None:
        self.config = config
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.model = model
        self.evaluator = evaluator or Evaluator()

        res_dir = results_dir or config.get("output", {}).get("results_dir", "results/fewrel/5shot/B0_sequential_ft/seed_2021")
        self.results_dir = Path(res_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_manager = CheckpointManager(self.results_dir / "checkpoints")
        self.logger = StreamingMetricsLogger(self.results_dir)

        # Performance matrices for Accuracy and Macro-F1
        seed = int(config.get("seed", 2021))
        task_order_path = str(config.get("dataset", {}).get("task_order_path", ""))

        self.acc_matrix = PerformanceMatrix(
            num_tasks=self.num_tasks,
            metric="accuracy",
            dataset=config.get("dataset", {}).get("name", "fewrel"),
            seed=seed,
            task_order_path=task_order_path,
        )
        self.f1_matrix = PerformanceMatrix(
            num_tasks=self.num_tasks,
            metric="macro_f1",
            dataset=config.get("dataset", {}).get("name", "fewrel"),
            seed=seed,
            task_order_path=task_order_path,
        )

    def train_task(self, stage: int, task: ContinualTask) -> None:
        """Train model on a single task's training set without access to old data."""
        logger.info(f"--- Training Stage {stage} (Task {task.task_id + 1}) ---")
        relations_per_task = len(task.relations)
        seen_classes = (stage + 1) * relations_per_task

        if hasattr(self.model, "set_seen_classes"):
            self.model.set_seen_classes(seen_classes)

        train_samples = task.train_samples
        batch_size = int(self.config.get("training", {}).get("batch_size", 16))
        epochs = int(self.config.get("training", {}).get("epochs_per_task", 5))
        lr = float(self.config.get("training", {}).get("learning_rate", 2.0e-5))
        weight_decay = float(self.config.get("training", {}).get("weight_decay", 0.01))
        warmup_ratio = float(self.config.get("training", {}).get("warmup_ratio", 0.1))

        if HAS_TORCH and hasattr(self.model, "encoder") and hasattr(self.model, "classifier"):
            # Real PyTorch training loop
            device = self.model.device
            self.model.encoder.train()
            self.model.classifier.train()

            # Group parameters with weight decay
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)]
                    + [p for n, p in self.model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)]
                    + [p for n, p in self.model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
            total_steps = max(1, (len(train_samples) // batch_size + (1 if len(train_samples) % batch_size else 0)) * epochs)
            warmup_steps = int(total_steps * warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                batches = batch_samples(train_samples, batch_size=batch_size, shuffle=True)
                epoch_loss = 0.0

                for b in batches:
                    optimizer.zero_grad()
                    batch_data = self.model.encode_batch(b)
                    logits = self.model.forward(
                        input_ids=batch_data["input_ids"],
                        attention_mask=batch_data["attention_mask"],
                        e1_indices=batch_data["e1_indices"],
                        e2_indices=batch_data["e2_indices"],
                    )
                    # Compute cross entropy over seen classes
                    labels = batch_data["labels"]
                    loss = loss_fn(logits, labels)
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.encoder.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.model.classifier.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    epoch_loss += float(loss.item())

                avg_loss = epoch_loss / max(1, len(batches))
                logger.info(f"Stage {stage} | Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        elif hasattr(self.model, "set_training_stage"):
            # Mock / CPU model simulation
            relation_ids = [s.relation_id for s in train_samples]
            self.model.set_training_stage(stage, relation_ids)

    def evaluate_seen_tasks(self, stage: int) -> dict[int, dict[str, float]]:
        """Evaluate current model on all observed tasks j = 0..stage."""
        results: dict[int, dict[str, float]] = {}

        for j in range(stage + 1):
            test_samples = self.tasks[j].test_samples
            eval_result = self.evaluator.evaluate(self.model, test_samples)

            acc = eval_result.accuracy
            f1 = eval_result.macro_f1

            # Record in matrices
            self.acc_matrix.set_score(trained_until=stage, evaluated_task=j, score=acc)
            self.f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=f1)

            # Stream immediately to log
            self.logger.log_evaluation(
                stage=stage,
                test_task=j,
                accuracy=acc,
                macro_f1=f1,
                sample_count=len(test_samples),
            )

            results[j] = {"accuracy": acc, "macro_f1": f1}
            logger.info(f"Stage {stage} -> Eval Task {j + 1} (T{j + 1}): Acc={acc:.4f}, F1={f1:.4f}")

        # Update matrices on disk immediately after stage finishes
        self.acc_matrix.export_csv(self.results_dir / "performance_matrix.csv")
        self.acc_matrix.export_json(self.results_dir / "performance_matrix.json")
        self.f1_matrix.export_csv(self.results_dir / "f1_performance_matrix.csv")
        self.f1_matrix.export_json(self.results_dir / "f1_performance_matrix.json")

        return results

    def run(self) -> dict[str, Any]:
        """Execute the entire continual learning experiment T1..TN."""
        logger.info(f"Starting Sequential Fine-Tuning across {self.num_tasks} tasks.")

        for stage in range(self.num_tasks):
            task = self.tasks[stage]

            # 1. Train current task
            self.train_task(stage, task)

            # 2. Save stage checkpoint
            self.checkpoint_manager.save_checkpoint(
                stage=stage,
                model=self.model,
                metadata={"task_id": task.task_id, "relations": task.relations},
            )

            # 3. Evaluate on all seen tasks j in [0..stage]
            self.evaluate_seen_tasks(stage)

        # 4. Generate final continual summary metrics
        summary = compute_continual_summary_metrics(self.acc_matrix)
        f1_summary = compute_continual_summary_metrics(self.f1_matrix)
        summary["final_macro_f1"] = f1_summary["final_average_accuracy"]
        summary["experiment"] = self.config.get("experiment", {}).get("id", "B0")
        summary["method"] = "sequential_ft"
        summary["seed"] = self.config.get("seed", 2021)
        summary["shot"] = self.config.get("dataset", {}).get("train_shot", 5)

        self.logger.save_summary(summary)

        # 5. Generate conclusion report
        conclusion_path = self.results_dir / "conclusion.md"
        generate_conclusion_report(summary, self.acc_matrix.as_list(), conclusion_path)

        # 6. Generate publication plots
        try:
            from .plotting import generate_b0_plots
            generate_b0_plots(self.results_dir, self.acc_matrix, summary)
            logger.info("Generated accuracy and forgetting plots.")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

        logger.info("Sequential FT complete. Summary saved.")
        return summary
