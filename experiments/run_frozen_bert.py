#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_SRC = REPO_ROOT / "dataset-pipelines" / "continual-relation-extraction" / "src"
BASELINES_SRC = SCRIPT_DIR / "continual-relation-baselines" / "src"

for p in [PIPELINE_SRC, BASELINES_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from datasets.fewrel import FewRelDataset
from evaluation.evaluator import compute_accuracy, compute_macro_f1
from evaluation.metrics import compute_continual_summary_metrics
from evaluation.performance_matrix import PerformanceMatrix
from task_generation.task_builder import ContinualTask, ContinualTaskBuilder
from task_generation.task_order import TaskOrder
from schemas.relation_sample import RelationSample

from cl_re_baselines.checkpoint import CheckpointManager
from cl_re_baselines.logger import StreamingMetricsLogger, generate_conclusion_report
from cl_re_baselines.model import BERTRelationClassifier
from generate_plots_svg import generate_svg_and_html

logger = logging.getLogger(__name__)


def resolve_path(candidate: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    p = Path(candidate)
    if p.is_absolute() and p.exists():
        return p
    if (base_dir / p).exists():
        return base_dir / p
    if (Path.cwd() / p).exists():
        return Path.cwd() / p
    return base_dir / p


def load_config(config_path: Path | str) -> dict:
    resolved = resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if resolved.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            with resolved.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            json_candidate = resolved.with_suffix(".json")
            if json_candidate.exists():
                with json_candidate.open("r", encoding="utf-8") as f:
                    return json.load(f)
            raise
    else:
        with resolved.open("r", encoding="utf-8") as f:
            return json.load(f)


def extract_features(
    model: BERTRelationClassifier,
    samples: Sequence[RelationSample],
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract concatenated entity marker representations [h_e1; h_e2] using frozen BERT."""
    if not samples:
        return (
            torch.empty((0, model.encoder.config.hidden_size * 2), device=model.device),
            torch.empty((0,), dtype=torch.long, device=model.device),
        )

    all_feats: list[torch.Tensor] = []
    all_labels: list[int] = []

    model.encoder.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            batch = model.encode_batch(chunk)
            outputs = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            seq_out = outputs[0]  # [B, L, H]
            batch_indices = torch.arange(batch["input_ids"].size(0), device=model.device)
            h_e1 = seq_out[batch_indices, batch["e1_indices"]]
            h_e2 = seq_out[batch_indices, batch["e2_indices"]]
            h_entity = torch.cat([h_e1, h_e2], dim=-1)  # [B, 2*H] = 1536
            all_feats.append(h_entity)
            all_labels.extend(batch["labels"].cpu().tolist())

    return torch.cat(all_feats, dim=0), torch.tensor(all_labels, dtype=torch.long, device=model.device)


def run_frozen_bert_experiment(
    config: dict[str, Any],
    tasks: list[ContinualTask],
    model: BERTRelationClassifier,
    results_dir: Path,
) -> dict[str, Any]:
    """Execute Sequential Linear Probing on Frozen BERT with feature caching."""
    num_tasks = len(tasks)
    seed = int(config.get("seed", 2021))
    task_order_path = str(config.get("dataset", {}).get("task_order_path", ""))
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs_per_task", 15))
    lr = float(train_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(results_dir / "checkpoints")
    metrics_logger = StreamingMetricsLogger(results_dir)

    acc_matrix = PerformanceMatrix(
        num_tasks=num_tasks,
        metric="accuracy",
        dataset=config.get("dataset", {}).get("name", "fewrel"),
        seed=seed,
        task_order_path=task_order_path,
    )
    f1_matrix = PerformanceMatrix(
        num_tasks=num_tasks,
        metric="macro_f1",
        dataset=config.get("dataset", {}).get("name", "fewrel"),
        seed=seed,
        task_order_path=task_order_path,
    )

    # 1. Feature Caching Phase
    print("\n------------------------------------------------------------------")
    print("Feature Caching: Extracting [h_e1; h_e2] embeddings with Frozen BERT...")
    start_cache_time = time.time()

    cached_train: list[tuple[torch.Tensor, torch.Tensor]] = []
    cached_test: list[tuple[torch.Tensor, torch.Tensor]] = []

    for t_idx, task in enumerate(tasks):
        t_feat, t_lbl = extract_features(model, task.train_samples)
        cached_train.append((t_feat, t_lbl))

        te_feat, te_lbl = extract_features(model, task.test_samples)
        cached_test.append((te_feat, te_lbl))
        print(f"  Task {t_idx + 1} cached: train={t_feat.size(0)} vectors, test={te_feat.size(0)} vectors")

    cache_elapsed = time.time() - start_cache_time
    total_cached_vectors = sum(x.size(0) + y.size(0) for (x, _), (y, _) in zip(cached_train, cached_test))
    print(f"Feature caching finished in {cache_elapsed:.2f}s ({total_cached_vectors} total vectors cached in RAM/VRAM).")
    print("------------------------------------------------------------------")

    loss_fn = nn.CrossEntropyLoss()

    # 2. Sequential Continual Learning Loop
    for stage in range(num_tasks):
        task = tasks[stage]
        relations_per_task = len(task.relations)
        seen_classes = (stage + 1) * relations_per_task
        model.set_seen_classes(seen_classes)

        print(f"\n>>> Stage {stage + 1}/{num_tasks}: Learning Task {stage + 1} (seen_classes={seen_classes})")
        train_x, train_y = cached_train[stage]
        n_samples = train_x.size(0)

        # Optimizer & scheduler for linear head
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = max(1, (n_samples // batch_size + (1 if n_samples % batch_size else 0)) * epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # Train linear classifier head
        model.classifier.train()
        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=model.device)
            epoch_loss = 0.0
            num_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                b_indices = perm[start_idx : start_idx + batch_size]
                b_x = train_x[b_indices]
                b_y = train_y[b_indices]

                optimizer.zero_grad()
                b_x_drop = model.dropout(b_x)
                logits = model.classifier(b_x_drop)

                if seen_classes < model.num_classes:
                    logits[:, seen_classes:] = -1e9

                loss = loss_fn(logits, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch + 1:2d}/{epochs} | Loss: {epoch_loss / max(1, num_batches):.4f}")

        # Save stage checkpoint (saving model.pt containing frozen BERT + trained classifier head)
        checkpoint_dir = checkpoint_manager.save_checkpoint(
            stage=stage,
            model=model,
            metadata={
                "task_id": task.task_id,
                "relations": task.relations,
                "seen_classes": seen_classes,
                "frozen_encoder": True,
            },
        )
        print(f"  Saved Stage {stage + 1} checkpoint at {checkpoint_dir.name}/model.pt")

        # Evaluate on all observed tasks j in [0..stage]
        model.classifier.eval()
        with torch.no_grad():
            for j in range(stage + 1):
                test_x, test_y = cached_test[j]
                logits = model.classifier(test_x)
                if seen_classes < model.num_classes:
                    logits[:, seen_classes:] = -1e9

                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                labels = test_y.cpu().tolist()

                acc = compute_accuracy(labels, preds)
                f1 = compute_macro_f1(labels, preds)

                acc_matrix.set_score(trained_until=stage, evaluated_task=j, score=acc)
                f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=f1)

                metrics_logger.log_evaluation(
                    stage=stage,
                    test_task=j,
                    accuracy=acc,
                    macro_f1=f1,
                    sample_count=len(labels),
                )
                print(f"    Eval on T{j + 1}: Acc = {acc * 100:.2f}% | Macro-F1 = {f1 * 100:.2f}%")

        # Stream update matrices to disk immediately
        acc_matrix.export_csv(results_dir / "performance_matrix.csv")
        acc_matrix.export_json(results_dir / "performance_matrix.json")
        f1_matrix.export_csv(results_dir / "f1_performance_matrix.csv")
        f1_matrix.export_json(results_dir / "f1_performance_matrix.json")

    # 3. Continual Summary Metrics & Final Reports
    summary = compute_continual_summary_metrics(acc_matrix)
    f1_summary = compute_continual_summary_metrics(f1_matrix)
    summary["final_macro_f1"] = f1_summary["final_average_accuracy"]
    summary["experiment"] = config.get("experiment", {}).get("id", "E003")
    summary["method"] = config.get("experiment", {}).get("name", "frozen_bert_linear_head")
    summary["baseline_role"] = config.get("experiment", {}).get("baseline_role", "representation_probing")
    summary["seed"] = seed
    summary["shot"] = int(config.get("dataset", {}).get("train_shot", 5))

    metrics_logger.save_summary(summary)

    # Generate conclusion report
    conclusion_path = results_dir / "conclusion.md"
    generate_conclusion_report(summary, acc_matrix.as_list(), conclusion_path)

    # Generate publication vector & raster plots
    try:
        generate_svg_and_html(results_dir)
        print("  Generated SVG/PNG plots and dashboard.html successfully.")
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E003: Frozen BERT + Linear Head Sequential Probing Runner."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/frozen_bert_linear_head_fewrel_5shot_seed2021.json",
        help="Path to experiment config file (JSON or YAML)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Override number of tasks (e.g. 2 for smoke test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device (e.g. mps, cuda, cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    seed = int(config.get("seed", 2021))
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})

    print("==================================================================")
    print("E003: Frozen BERT + Linear Head (Sequential Probing Baseline)")
    print("==================================================================")
    print(f"Experiment ID: {config.get('experiment', {}).get('id')}")
    print(f"Seed:          {seed}")
    print(f"Train Shot:    {dataset_cfg.get('train_shot')}-shot")
    print(f"Val Shot:      {dataset_cfg.get('val_shot')}-shot")
    print(f"Config:        {args.config}")
    print("==================================================================")

    # 1. Load Dataset
    raw_data_dir = resolve_path(dataset_cfg.get("raw_data_dir", "dataset-pipelines/continual-relation-extraction/data/raw/fewrel"))
    print(f"\n[1/5] Loading FewRel dataset from {raw_data_dir}...")
    dataset = FewRelDataset(
        data_dir=raw_data_dir,
        train_val_test_counts=(
            int(dataset_cfg.get("train_shot", 5)),
            int(dataset_cfg.get("val_shot", 5)),
            140,
        ),
    )
    relations = dataset.get_relations()
    print(f"  Loaded {len(relations)} relations.")

    # 2. Load Task Order
    task_order_path = resolve_path(dataset_cfg.get("task_order_path", "dataset-pipelines/continual-relation-extraction/task-orders/fewrel/order_seed_2021.json"))
    print(f"\n[2/5] Loading Task Order from {task_order_path}...")
    task_order = TaskOrder.load(task_order_path)
    task_order.validate_against_expected_relations(relations)
    print(f"  Validated task order: {task_order.num_tasks} tasks, seed {task_order.seed}")

    # 3. Build Continual Tasks
    print("\n[3/5] Building continual tasks...")
    builder = ContinualTaskBuilder()
    tasks = builder.build_tasks(dataset, task_order)

    if args.num_tasks is not None:
        tasks = tasks[: args.num_tasks]
        print(f"  Restricted tasks to first {len(tasks)} tasks (Smoke Test mode).")

    for t in tasks:
        print(f"  Task {t.task_id + 1} (T{t.task_id + 1}): {len(t.relations)} rels | train={t.train_count}, val={t.val_count}, test={t.test_count}")

    # 4. Instantiate Model & Freeze Encoder
    print("\n[4/5] Initializing model & freezing encoder...")
    device_override = args.device or config.get("training", {}).get("device", "auto")

    relation_to_id = dataset.get_relation_to_id()
    relation_to_class_idx = {rel: idx for idx, rel in enumerate(task_order.all_relations)}
    class_idx_to_rel_id = {idx: relation_to_id[rel] for rel, idx in relation_to_class_idx.items()}

    model = BERTRelationClassifier(
        backbone_name=config.get("model", {}).get("backbone", "bert-base-uncased"),
        num_classes=int(config.get("model", {}).get("num_classes", 80)),
        dropout_rate=float(config.get("model", {}).get("classifier_dropout", 0.1)),
        max_seq_length=int(config.get("model", {}).get("max_seq_length", 128)),
        device=device_override,
        relation_to_class_idx=relation_to_class_idx,
        class_idx_to_rel_id=class_idx_to_rel_id,
    )

    # Strict freezing of BERT backbone
    frozen_param_count = 0
    for param in model.encoder.parameters():
        param.requires_grad = False
        frozen_param_count += param.numel()
    model.encoder.eval()

    trainable_param_count = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"  Frozen Backbone Parameters:  {frozen_param_count:,} (requires_grad=False)")
    print(f"  Trainable Linear Head Params: {trainable_param_count:,} (requires_grad=True)")

    # 5. Execute Experiment
    print("\n[5/5] Executing Sequential Linear Probing...")
    results_dir_str = args.output_dir or output_cfg.get("results_dir", "results/fewrel/5shot/Frozen_BERT_linear_head/seed_2021")
    results_dir = resolve_path(results_dir_str)

    summary = run_frozen_bert_experiment(
        config=config,
        tasks=tasks,
        model=model,
        results_dir=results_dir,
    )

    print("\n==================================================================")
    print("E003 Frozen BERT + Linear Head Experiment Completed Successfully!")
    print("==================================================================")
    print(f"Final Average Accuracy (Final AA): {summary['final_average_accuracy'] * 100:.2f}%")
    print(f"Final Macro-F1:                    {summary.get('final_macro_f1', 0.0) * 100:.2f}%")
    print(f"Average Incremental Accuracy (AIA):{summary['average_incremental_accuracy'] * 100:.2f}%")
    print(f"Average Catastrophic Forgetting:   {summary['average_forgetting'] * 100:.2f}%")
    print(f"Backward Transfer (BWT):           {summary['backward_transfer'] * 100:.2f}%")
    most_f = summary.get("most_forgotten_task", {})
    if most_f.get("task_name"):
        print(f"Most Forgotten Task:               {most_f['task_name']} (drop: {float(most_f.get('forgetting', 0.0)) * 100:.2f}%)")
    print("==================================================================")
    print(f"Artifacts exported to: {results_dir}")
    print(f"  - Performance Matrix (CSV): {results_dir / 'performance_matrix.csv'}")
    print(f"  - Performance Matrix (JSON):{results_dir / 'performance_matrix.json'}")
    print(f"  - Metrics Log:              {results_dir / 'metrics.jsonl'}")
    print(f"  - Summary:                  {results_dir / 'summary.json'}")
    print(f"  - Conclusion:               {results_dir / 'conclusion.md'}")
    print(f"  - Checkpoints:              {results_dir / 'checkpoints'}")
    print(f"  - Plots:                    {results_dir / 'plots'}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
