#!/usr/bin/env python3
"""Runner for Proposed Method Baselines P0 (Shared Prompt) and P1 (Oracle-Task Prompt).

Protocol: Protocol v1 Track A (FewRel 8 tasks x 10 relations, 5-shot)
Backbone: 100% Frozen BERT-base-uncased with soft prompt prepending.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

# Setup search paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
PIPELINE_SRC = REPO_ROOT / "dataset-pipelines" / "continual-relation-extraction" / "src"
BASELINES_SRC = SCRIPT_DIR / "continual-relation-baselines" / "src"

for p in [REPO_ROOT, SRC_DIR, PIPELINE_SRC, BASELINES_SRC, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation.evaluator import compute_accuracy, compute_macro_f1
from evaluation.metrics import compute_continual_summary_metrics
from evaluation.performance_matrix import PerformanceMatrix
from cl_re_baselines.checkpoint import CheckpointManager
from cl_re_baselines.logger import StreamingMetricsLogger, generate_conclusion_report
from generate_plots_svg import generate_svg_and_html

from src.data.protocol_v1_loader import (
    ProtocolV1Dataset,
    ProtocolV1Sample,
    ProtocolV1Task,
    collate_samples_for_bert,
)
from src.models.prompt_bert import PromptedBERTRelationClassifier

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    import yaml
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        diff_stat = subprocess.check_output(
            ["git", "diff", "--stat"], cwd=REPO_ROOT, text=True
        ).strip()
        return {
            "commit": commit,
            "branch": branch,
            "has_dirty_working_tree": bool(diff_stat),
            "diff_stat": diff_stat,
        }
    except Exception as e:
        return {"error": str(e)}


def train_single_task(
    model: PromptedBERTRelationClassifier,
    task: ProtocolV1Task,
    config: dict[str, Any],
    memory_samples: list[ProtocolV1Sample] | None = None,
) -> float:
    """Train prompt and classifier head on current task samples (and replay memory if M>0)."""
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs_per_task", 15))
    lr = float(train_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))

    # Combine task train samples with memory samples
    all_train_samples = list(task.train_samples)
    if memory_samples:
        all_train_samples.extend(memory_samples)

    n_samples = len(all_train_samples)
    trainable_params = model.get_trainable_params(task_id=task.task_id)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, (n_samples // batch_size + (1 if n_samples % batch_size else 0)) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.classifier.train()
    total_epoch_loss = 0.0

    for epoch in range(epochs):
        # Shuffle per epoch
        perm = list(range(n_samples))
        random.shuffle(perm)
        epoch_loss = 0.0
        num_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            batch_samples = [all_train_samples[i] for i in batch_indices]

            batch_data = collate_samples_for_bert(
                batch_samples,
                tokenizer=model.tokenizer,
                device=model.target_device,
                max_seq_length=int(config.get("model", {}).get("max_seq_length", 128)),
            )

            # In P1 (oracle_pool), provide task_id
            task_id_arg = task.task_id if model.prompt_mode == "oracle_pool" else None

            optimizer.zero_grad()
            out = model(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                e1_indices=batch_data["e1_indices"],
                e2_indices=batch_data["e2_indices"],
                task_id=task_id_arg,
            )

            loss = F.cross_entropy(out["logits"], batch_data["labels"])
            loss.backward()

            nn.utils.clip_grad_norm_(trainable_params, clip_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        total_epoch_loss = avg_loss
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.4f}")

    # Compute and store prototypes for the task's classes
    model.eval()
    with torch.no_grad():
        for class_idx in task.class_indices:
            class_samples = [s for s in task.train_samples if s.class_idx == class_idx]
            if class_samples:
                c_data = collate_samples_for_bert(
                    class_samples,
                    tokenizer=model.tokenizer,
                    device=model.target_device,
                    max_seq_length=int(config.get("model", {}).get("max_seq_length", 128)),
                )
                task_id_arg = task.task_id if model.prompt_mode == "oracle_pool" else None
                h_r = model.extract_features(
                    c_data["input_ids"],
                    c_data["attention_mask"],
                    c_data["e1_indices"],
                    c_data["e2_indices"],
                    task_id=task_id_arg,
                )
                model.update_prototypes(class_idx, h_r)

    return total_epoch_loss


def evaluate_single_task(
    model: PromptedBERTRelationClassifier,
    test_samples: list[ProtocolV1Sample],
    eval_task_id: int,
    config: dict[str, Any],
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate model on a specific task's test set."""
    model.eval()
    max_seq_len = int(config.get("model", {}).get("max_seq_length", 128))

    all_logits: list[torch.Tensor] = []
    all_reps: list[torch.Tensor] = []
    all_labels: list[int] = []

    # For P1 (oracle_pool), use Oracle Task ID eval_task_id
    task_id_arg = eval_task_id if model.prompt_mode == "oracle_pool" else None

    with torch.no_grad():
        for i in range(0, len(test_samples), batch_size):
            chunk = test_samples[i : i + batch_size]
            batch_data = collate_samples_for_bert(
                chunk,
                tokenizer=model.tokenizer,
                device=model.target_device,
                max_seq_length=max_seq_len,
            )
            out = model(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                e1_indices=batch_data["e1_indices"],
                e2_indices=batch_data["e2_indices"],
                task_id=task_id_arg,
            )
            all_logits.append(out["logits"])
            all_reps.append(out["representations"])
            all_labels.extend(batch_data["labels"].cpu().tolist())

    logits = torch.cat(all_logits, dim=0)
    reps = torch.cat(all_reps, dim=0)

    # Class-Incremental Linear Head predictions
    cil_preds = torch.argmax(logits, dim=-1).cpu().tolist()
    cil_acc = compute_accuracy(all_labels, cil_preds)
    cil_f1 = compute_macro_f1(all_labels, cil_preds)

    # Prototype predictions
    proto_logits = model.predict_prototype(reps, seen_classes_only=True)
    proto_preds = torch.argmax(proto_logits, dim=-1).cpu().tolist()
    proto_acc = compute_accuracy(all_labels, proto_preds)

    # Task-Incremental (TIL) predictions: mask to task's classes only
    task_class_start = eval_task_id * 10
    task_class_end = task_class_start + 10
    til_logits = logits.clone()
    til_mask = torch.ones_like(til_logits, dtype=torch.bool)
    til_mask[:, task_class_start:task_class_end] = False
    til_logits[til_mask] = -1e9
    til_preds = torch.argmax(til_logits, dim=-1).cpu().tolist()
    til_acc = compute_accuracy(all_labels, til_preds)

    return {
        "accuracy": cil_acc,
        "macro_f1": cil_f1,
        "prototype_accuracy": proto_acc,
        "til_accuracy": til_acc,
        "sample_count": len(all_labels),
    }


def run_experiment(config_path: str | Path, device_override: str | None = None) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    seed = int(config.get("seed", 2021))
    set_seed(seed)

    exp_cfg = config.get("experiment", {})
    exp_id = exp_cfg.get("id", "P0")
    model_cfg = config.get("model", {})
    continual_cfg = config.get("continual", {})
    output_cfg = config.get("output", {})

    results_dir = REPO_ROOT / output_cfg.get("results_dir", f"runs/{exp_id.lower()}/fewrel_5shot/seed_{seed}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup file and stdout logging
    log_file = results_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    print("==================================================================")
    print(f"Continual Relation Extraction: {exp_id} ({exp_cfg.get('name')})")
    print(f"Description: {exp_cfg.get('description')}")
    print(f"Role:        {exp_cfg.get('baseline_role')}")
    print(f"Seed:        {seed}")
    print(f"Results Dir: {results_dir}")
    print("==================================================================")

    # 1. Load Protocol v1 Dataset
    dataset_cfg = config.get("dataset", {})
    dataset = ProtocolV1Dataset(
        base_dir=REPO_ROOT / dataset_cfg.get("base_dir", "data/protocol_v1/fewrel"),
        split_subdir=dataset_cfg.get("split_subdir", "CFRLdata_10_100_10_5"),
        task_order_strategy=dataset_cfg.get("task_order_strategy", "conpl_seed_shuffle"),
        seed=seed,
        train_shot=int(dataset_cfg.get("train_shot", 5)),
        test_shot=int(dataset_cfg.get("test_shot", 100)),
        max_seq_length=int(model_cfg.get("max_seq_length", 128)),
    )
    tasks = dataset.build_tasks()
    num_tasks = len(tasks)
    print(f"\n[1/4] Loaded Protocol v1 Dataset: {num_tasks} continual tasks.")
    for t in tasks:
        print(f"  Task {t.task_id + 1} (Cluster {t.cluster_id}): 10 rels | Train: {len(t.train_samples)}, Test: {len(t.test_samples)}")

    # 2. Instantiate Model
    target_device = device_override or config.get("training", {}).get("device", "auto")
    prompt_mode = model_cfg.get("prompt_mode", "shared")
    prompt_length = int(model_cfg.get("prompt_length", 8))

    print(f"\n[2/4] Initializing PromptedBERTRelationClassifier (mode={prompt_mode}, L={prompt_length})...")
    model = PromptedBERTRelationClassifier(
        backbone_name=model_cfg.get("backbone", "bert-base-uncased"),
        prompt_mode=prompt_mode,
        prompt_length=prompt_length,
        num_tasks=num_tasks,
        num_classes=int(model_cfg.get("num_classes", 80)),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.1)),
        device=target_device,
    )
    print(f"  Target device: {model.target_device}")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_trainable:,} (BERT backbone is 100% frozen)")

    # 3. Setup Matrices and Loggers
    acc_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="accuracy", dataset="fewrel", seed=seed)
    f1_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="macro_f1", dataset="fewrel", seed=seed)
    proto_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="prototype_accuracy", dataset="fewrel", seed=seed)
    til_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="til_accuracy", dataset="fewrel", seed=seed)

    checkpoint_manager = CheckpointManager(results_dir / "checkpoints")
    metrics_logger = StreamingMetricsLogger(results_dir)

    memory_size = int(continual_cfg.get("memory_size", 0))
    memory_buffer: list[ProtocolV1Sample] = []

    # 4. Continual Learning Loop
    print("\n[3/4] Starting Continual Learning across 8 tasks...")
    start_time = time.time()

    for stage in range(num_tasks):
        task = tasks[stage]
        seen_classes = (stage + 1) * 10
        model.set_seen_classes(seen_classes)

        print(f"\n>>> Stage {stage + 1}/{num_tasks}: Learning Task {stage + 1} (Cluster {task.cluster_id}, Seen Classes: {seen_classes})")
        stage_t0 = time.time()

        # Train on current task
        train_loss = train_single_task(
            model=model,
            task=task,
            config=config,
            memory_samples=memory_buffer if memory_size > 0 else None,
        )

        # Update memory buffer if M > 0
        if memory_size > 0:
            for c_idx in task.class_indices:
                candidates = [s for s in task.train_samples if s.class_idx == c_idx]
                if candidates:
                    memory_buffer.extend(candidates[:memory_size])

        # Evaluate on all observed tasks j in [0..stage]
        print(f"  Evaluating on tasks 1..{stage + 1}:")
        for j in range(stage + 1):
            eval_res = evaluate_single_task(
                model=model,
                test_samples=tasks[j].test_samples,
                eval_task_id=j,
                config=config,
                batch_size=int(config.get("evaluation", {}).get("batch_size", 64)),
            )

            acc_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["accuracy"])
            f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["macro_f1"])
            proto_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["prototype_accuracy"])
            til_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["til_accuracy"])

            metrics_logger.log_evaluation(
                stage=stage,
                test_task=j,
                accuracy=eval_res["accuracy"],
                macro_f1=eval_res["macro_f1"],
                sample_count=eval_res["sample_count"],
            )

            print(
                f"    Eval on T{j + 1}: CIL Acc = {eval_res['accuracy'] * 100:6.2f}% | "
                f"Proto Acc = {eval_res['prototype_accuracy'] * 100:6.2f}% | "
                f"TIL Acc = {eval_res['til_accuracy'] * 100:6.2f}% | "
                f"F1 = {eval_res['macro_f1'] * 100:6.2f}%"
            )

        # Stream update matrices to disk
        acc_matrix.export_csv(results_dir / "performance_matrix.csv")
        acc_matrix.export_json(results_dir / "performance_matrix.json")
        f1_matrix.export_csv(results_dir / "f1_performance_matrix.csv")
        proto_matrix.export_csv(results_dir / "prototype_performance_matrix.csv")
        til_matrix.export_csv(results_dir / "til_performance_matrix.csv")

        # Save stage checkpoint
        checkpoint_manager.save_checkpoint(
            stage=stage,
            model=model,
            metadata={
                "task_id": task.task_id,
                "cluster_id": task.cluster_id,
                "seen_classes": seen_classes,
                "prompt_mode": prompt_mode,
            },
        )
        print(f"  Stage {stage + 1} completed in {time.time() - stage_t0:.2f}s.")

    total_duration = time.time() - start_time
    print(f"\n[4/4] Continual Learning Finished in {total_duration:.2f}s!")

    # 5. Compute Final Continual Summary Metrics
    summary = compute_continual_summary_metrics(acc_matrix)
    proto_summary = compute_continual_summary_metrics(proto_matrix)
    til_summary = compute_continual_summary_metrics(til_matrix)

    summary["final_prototype_accuracy"] = proto_summary["final_average_accuracy"]
    summary["final_til_accuracy"] = til_summary["final_average_accuracy"]
    summary["proto_average_incremental_accuracy"] = proto_summary["average_incremental_accuracy"]
    summary["til_average_incremental_accuracy"] = til_summary["average_incremental_accuracy"]
    summary["experiment"] = exp_id
    summary["method"] = exp_cfg.get("name")
    summary["baseline_role"] = exp_cfg.get("baseline_role")
    summary["seed"] = seed
    summary["shot"] = int(dataset_cfg.get("train_shot", 5))
    summary["prompt_length"] = prompt_length
    summary["total_duration_seconds"] = round(total_duration, 2)

    # Save summary
    metrics_logger.save_summary(summary)

    # Save standardized metrics.json conforming to Protocol v1
    metrics_json_path = results_dir / "metrics.json"
    full_metrics = {
        "experiment_id": exp_id,
        "method": exp_cfg.get("name"),
        "role": exp_cfg.get("baseline_role"),
        "seed": seed,
        "protocol": "Protocol v1 Track A (FewRel 5-shot)",
        "memory_size": memory_size,
        "accuracy_whole": summary["final_average_accuracy"],
        "average_incremental_accuracy": summary["average_incremental_accuracy"],
        "average_forgetting": summary["average_forgetting"],
        "backward_transfer": summary["backward_transfer"],
        "prototype_accuracy_whole": summary["final_prototype_accuracy"],
        "til_accuracy_whole": summary["final_til_accuracy"],
        "matrix_A": acc_matrix.as_list(),
        "proto_matrix_A": proto_matrix.as_list(),
        "til_matrix_A": til_matrix.as_list(),
        "execution_duration_sec": total_duration,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  Saved {metrics_json_path}")

    # Save provenance info
    git_info = get_git_info()
    provenance = {
        "git": git_info,
        "config": config,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(model.target_device),
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with (results_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # Generate Markdown conclusion report
    conclusion_path = results_dir / "conclusion.md"
    generate_conclusion_report(summary, acc_matrix.as_list(), conclusion_path)

    # Generate SVG/PNG plots
    try:
        generate_svg_and_html(results_dir)
        print("  Generated SVG/PNG plots and dashboard.html successfully.")
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    print("\n==================================================================")
    print(f"FINAL METRICS SUMMARY: {exp_id}")
    print(f"  Final Accuracy Whole (A_T):     {summary['final_average_accuracy'] * 100:.2f}%")
    print(f"  Average Incremental Acc (AIA):  {summary['average_incremental_accuracy'] * 100:.2f}%")
    print(f"  Average Forgetting (F_T):       {summary['average_forgetting'] * 100:.2f}%")
    print(f"  Backward Transfer (BWT):        {summary['backward_transfer'] * 100:.2f}%")
    print(f"  Prototype Accuracy Whole:       {summary['final_prototype_accuracy'] * 100:.2f}%")
    print(f"  TIL (Within-Task) Accuracy:     {summary['final_til_accuracy'] * 100:.2f}%")
    print("==================================================================")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="P0/P1 Soft-Prompted Frozen BERT Continual Relation Extraction Runner.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config file.")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (e.g. mps, cuda, cpu).")
    args = parser.parse_args()

    run_experiment(config_path=args.config, device_override=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
