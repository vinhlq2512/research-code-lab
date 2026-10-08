#!/usr/bin/env python3
"""Upper Bound Runner: Joint Training (Multitask) for Continual Relation Extraction.

All 80 relations are trained jointly across all classes without temporal task splits,
establishing the empirical Upper Bound performance on FewRel Track A (5-shot, seed 2021).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

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
from evaluation.evaluator import Evaluator
from task_generation.task_builder import ContinualTaskBuilder
from task_generation.task_order import TaskOrder
from cl_re_baselines.data_loader import batch_samples
from cl_re_baselines.model import BERTRelationClassifier

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


def load_config(config_path: Path | str) -> dict[str, Any]:
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
            raise ImportError(f"PyYAML is not installed to read {resolved.name}.")
    else:
        with resolved.open("r", encoding="utf-8") as f:
            return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def generate_upper_bound_conclusion(
    summary: dict[str, Any],
    per_task: list[dict[str, Any]],
    output_path: Path,
    b0_summary: dict[str, Any] | None = None,
) -> None:
    overall_acc = summary["overall_test_accuracy"] * 100
    overall_f1 = summary["overall_test_macro_f1"] * 100
    best_val_acc = summary["best_val_accuracy"] * 100
    best_epoch = summary["best_val_epoch"]
    total_epochs = summary["total_epochs"]

    lines = [
        "# Upper Bound: Joint Training (Multitask) Empirical Report",
        "",
        "## 1. Executive Summary",
        f"- **Benchmark:** FewRel Track A (80 relations, 5-shot, seed {summary.get('seed', 2021)})",
        f"- **Method:** Joint Training (Multitask Upper Bound, all 80 classes unmasked)",
        f"- **Best Validation Epoch:** Epoch {best_epoch}/{total_epochs} (Val Accuracy: {best_val_acc:.2f}%)",
        f"- **Overall Test Accuracy (80-way):** **{overall_acc:.2f}%**",
        f"- **Overall Test Macro-F1:** **{overall_f1:.2f}%**",
        "",
        "## 2. Head-to-Head Comparison: Upper Bound vs Lower Bound (B0)",
        "",
    ]

    if b0_summary:
        b0_final_aa = b0_summary.get("final_average_accuracy", 0.0) * 100
        b0_final_f1 = b0_summary.get("final_macro_f1", 0.0) * 100
        b0_aia = b0_summary.get("average_incremental_accuracy", 0.0) * 100
        b0_af = b0_summary.get("average_forgetting", 0.0) * 100
        acc_gap = overall_acc - b0_final_aa
        f1_gap = overall_f1 - b0_final_f1

        lines.extend([
            "| Metric | Upper Bound (Joint) | Lower Bound (B0 Sequential FT) | Gap (Upper - Lower) |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Final Accuracy (AA)** | **{overall_acc:.2f}%** | {b0_final_aa:.2f}% | **+{acc_gap:.2f}%** |",
            f"| **Final Macro-F1** | **{overall_f1:.2f}%** | {b0_final_f1:.2f}% | **+{f1_gap:.2f}%** |",
            f"| **Catastrophic Forgetting (AF)** | **0.00%** (N/A) | {b0_af:.2f}% | — |",
            f"| **Average Incremental Acc (AIA)** | **{overall_acc:.2f}%** | {b0_aia:.2f}% | **+{overall_acc - b0_aia:.2f}%** |",
            "",
            "> [!NOTE]",
            f"> Joint Training đạt **{overall_acc:.2f}%** trên không gian 80 lớp, thiết lập trần hiệu năng (Upper Bound) vững chắc.",
            f"> Khoảng cách giữa Upper Bound và Lower Bound B0 là **{acc_gap:.2f}%**, đây chính là không gian tối ưu hóa (Optimization Room) cho các phương pháp Continual Learning như ConPL, CPL và TAPTA.",
            "",
        ])

    lines.extend([
        "## 3. Per-Task Breakdown on Test Set (1,400 samples / task)",
        "",
        "| Task ID | Task Name | Relations | Sample Count | Test Accuracy | Test Macro-F1 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for row in per_task:
        lines.append(
            f"| {row['task_id']} | {row['task_name']} | {row['num_relations']} | {row['sample_count']} | "
            f"{row['accuracy'] * 100:.2f}% | {row['macro_f1'] * 100:.2f}% |"
        )

    lines.extend([
        "",
        "## 4. Scientific Conclusion",
        "1. **Phân loại 80 lớp Few-Shot (5-shot):** Dù chỉ có 5 mẫu huấn luyện cho mỗi quan hệ, việc huấn luyện đa nhiệm đồng thời cho phép biểu diễn thực thể của BERT học không gian đặc trưng toàn cục mà không bị trôi trọng số.",
        "2. **Thước đo chuẩn mực (Benchmark Ceiling):** Mọi mô hình Continual Learning trong các nghiên cứu tiếp theo sẽ được đánh giá dựa trên mức độ tiếp cận Cận trên này so với Cận dưới B0.",
        "",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upper Bound Joint Training Runner for Continual Relation Extraction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/upper_bound_joint_fewrel_5shot_seed2021.json",
        help="Path to experiment config file (JSON or YAML)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device (e.g. cpu, cuda, mps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    seed = int(config.get("seed", 2021))
    set_seed(seed)

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})
    device_name = args.device or training_cfg.get("device", "auto")

    results_dir_str = args.output_dir or output_cfg.get("results_dir", "results/fewrel/5shot/Upper_Bound_joint/seed_2021")
    results_dir = resolve_path(results_dir_str)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("UPPER BOUND: Joint Training (FewRel Track A 5-shot)")
    print("==================================================================")
    print(f"Experiment ID: {config.get('experiment', {}).get('id')}")
    print(f"Seed:          {seed}")
    print(f"Device:        {device_name}")
    print(f"Output Dir:    {results_dir}")
    print("==================================================================")

    # 1. Load Dataset & Task Order
    raw_data_dir = resolve_path(dataset_cfg.get("raw_data_dir", "dataset-pipelines/continual-relation-extraction/data/raw/fewrel"))
    task_order_path = resolve_path(dataset_cfg.get("task_order_path", "dataset-pipelines/continual-relation-extraction/task-orders/fewrel/order_seed_2021.json"))

    print(f"\n[1/5] Loading FewRel dataset and Task Order...")
    dataset = FewRelDataset(
        data_dir=raw_data_dir,
        train_val_test_counts=(
            int(dataset_cfg.get("train_shot", 5)),
            int(dataset_cfg.get("val_shot", 5)),
            int(dataset_cfg.get("max_test_per_relation", 140)),
        ),
    )
    relations = dataset.get_relations()
    task_order = TaskOrder.load(task_order_path)
    task_order.validate_against_expected_relations(relations)

    # 2. Build tasks and pool samples across all 80 relations
    print(f"\n[2/5] Pooling training and validation samples across all 80 relations...")
    builder = ContinualTaskBuilder()
    tasks = builder.build_tasks(dataset, task_order)

    all_train_samples = [s for t in tasks for s in t.train_samples]
    all_val_samples = [s for t in tasks for s in t.validation_samples]
    all_test_samples = [s for t in tasks for s in t.test_samples]

    print(f"  Total Pooled Train Samples: {len(all_train_samples)} ({len(relations)} relations × 5)")
    print(f"  Total Pooled Val Samples:   {len(all_val_samples)} ({len(relations)} relations × 5)")
    print(f"  Total Pooled Test Samples:  {len(all_test_samples)} ({len(relations)} relations × 140)")

    # 3. Model Initialization
    print(f"\n[3/5] Initializing BERTRelationClassifier on {device_name}...")
    relation_to_id = dataset.get_relation_to_id()
    relation_to_class_idx = {rel: idx for idx, rel in enumerate(task_order.all_relations)}
    class_idx_to_rel_id = {idx: relation_to_id[rel] for rel, idx in relation_to_class_idx.items()}

    model = BERTRelationClassifier(
        backbone_name=config.get("model", {}).get("backbone", "bert-base-uncased"),
        num_classes=int(config.get("model", {}).get("num_classes", 80)),
        dropout_rate=float(config.get("model", {}).get("classifier_dropout", 0.1)),
        max_seq_length=int(config.get("model", {}).get("max_seq_length", 128)),
        device=device_name,
        relation_to_class_idx=relation_to_class_idx,
        class_idx_to_rel_id=class_idx_to_rel_id,
    )
    # Ensure all 80 classes are unmasked
    model.set_seen_classes(80)

    # 4. Joint Training Loop with Best Val Checkpointing
    epochs = args.epochs or int(training_cfg.get("epochs", 15))
    batch_size = int(training_cfg.get("batch_size", 16))
    lr = float(training_cfg.get("learning_rate", 2.0e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))

    print(f"\n[4/5] Starting Joint Training ({epochs} epochs, batch_size={batch_size}, LR={lr})...")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)]
            + [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)]
            + [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    steps_per_epoch = max(1, len(all_train_samples) // batch_size + (1 if len(all_train_samples) % batch_size else 0))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    evaluator = Evaluator()
    training_log_path = results_dir / output_cfg.get("training_log", "training_log.jsonl")
    best_checkpoint_path = results_dir / output_cfg.get("best_checkpoint", "best_model.pt")

    best_val_acc = -1.0
    best_val_epoch = -1
    best_state_dict: dict[str, Any] | None = None

    with training_log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, epochs + 1):
            model.encoder.train()
            model.classifier.train()

            # Shuffle training pool each epoch
            shuffled_train = list(all_train_samples)
            random.shuffle(shuffled_train)
            batches = batch_samples(shuffled_train, batch_size=batch_size, shuffle=False)

            epoch_loss = 0.0
            for b in batches:
                optimizer.zero_grad()
                batch_data = model.encode_batch(b)
                logits = model.forward(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    e1_indices=batch_data["e1_indices"],
                    e2_indices=batch_data["e2_indices"],
                )
                loss = loss_fn(logits, batch_data["labels"])
                loss.backward()

                nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())

            avg_train_loss = epoch_loss / max(1, len(batches))

            # Evaluate on Validation set (400 samples)
            val_result = evaluator.evaluate(model, all_val_samples)
            val_acc = val_result.accuracy
            val_f1 = val_result.macro_f1

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_state_dict = model.get_state_dict()
                torch.save(best_state_dict, best_checkpoint_path)

            log_entry = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "is_best": is_best,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            star = " ⭐ [BEST]" if is_best else ""
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Val F1: {val_f1 * 100:.2f}%{star}")

    print(f"\n  Training completed. Best model found at Epoch {best_val_epoch} (Val Acc: {best_val_acc * 100:.2f}%).")

    # 5. Full Evaluation using Best Model
    print(f"\n[5/5] Evaluating Best Model on Test Set (11,200 samples)...")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    overall_eval = evaluator.evaluate(model, all_test_samples)
    print(f"  Overall 80-Way Test Accuracy: {overall_eval.accuracy * 100:.2f}%")
    print(f"  Overall 80-Way Test Macro-F1: {overall_eval.macro_f1 * 100:.2f}%")

    # Per-task breakdown
    per_task_results: list[dict[str, Any]] = []
    print("\n  Per-Task Test Performance Breakdown:")
    for t in tasks:
        task_eval = evaluator.evaluate(model, t.test_samples)
        task_row = {
            "task_id": t.task_id,
            "task_name": f"T{t.task_id + 1}",
            "num_relations": len(t.relations),
            "sample_count": len(t.test_samples),
            "accuracy": task_eval.accuracy,
            "macro_f1": task_eval.macro_f1,
        }
        per_task_results.append(task_row)
        print(f"    Task {t.task_id + 1} (T{t.task_id + 1}): Acc = {task_eval.accuracy * 100:.2f}%, F1 = {task_eval.macro_f1 * 100:.2f}%")

    # Export Per-Task Breakdown CSV & JSON
    breakdown_csv = results_dir / output_cfg.get("breakdown_csv", "per_task_breakdown.csv")
    with breakdown_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "task_name", "num_relations", "sample_count", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(per_task_results)

    breakdown_json = results_dir / output_cfg.get("breakdown_json", "per_task_breakdown.json")
    with breakdown_json.open("w", encoding="utf-8") as f:
        json.dump(per_task_results, f, indent=2)

    # Load B0 summary for comparison
    b0_summary_path = resolve_path("results/fewrel/5shot/B0_sequential_ft/seed_2021/summary.json")
    b0_summary = None
    if b0_summary_path.exists():
        with b0_summary_path.open("r", encoding="utf-8") as f:
            b0_summary = json.load(f)

    # Save summary.json
    summary_data = {
        "experiment": "UPPER_BOUND",
        "method": "joint_training",
        "seed": seed,
        "train_shot": 5,
        "val_shot": 5,
        "total_relations": 80,
        "total_train_samples": len(all_train_samples),
        "total_test_samples": len(all_test_samples),
        "total_epochs": epochs,
        "best_val_epoch": best_val_epoch,
        "best_val_accuracy": best_val_acc,
        "overall_test_accuracy": overall_eval.accuracy,
        "overall_test_macro_f1": overall_eval.macro_f1,
        "per_task_accuracy": {row["task_name"]: row["accuracy"] for row in per_task_results},
        "per_task_macro_f1": {row["task_name"]: row["macro_f1"] for row in per_task_results},
    }
    if b0_summary:
        summary_data["b0_comparison"] = {
            "b0_final_accuracy": b0_summary.get("final_average_accuracy"),
            "b0_final_macro_f1": b0_summary.get("final_macro_f1"),
            "accuracy_gain_over_b0": overall_eval.accuracy - b0_summary.get("final_average_accuracy", 0.0),
            "macro_f1_gain_over_b0": overall_eval.macro_f1 - b0_summary.get("final_macro_f1", 0.0),
        }

    summary_file = results_dir / output_cfg.get("summary_file", "summary.json")
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Generate conclusion.md
    conclusion_file = results_dir / output_cfg.get("conclusion_file", "conclusion.md")
    generate_upper_bound_conclusion(summary_data, per_task_results, conclusion_file, b0_summary)

    print("\n==================================================================")
    print("UPPER BOUND Joint Training Completed Successfully!")
    print("==================================================================")
    print(f"Overall 80-Way Test Accuracy: {overall_eval.accuracy * 100:.2f}%")
    print(f"Overall 80-Way Test Macro-F1: {overall_eval.macro_f1 * 100:.2f}%")
    if b0_summary:
        gain = (overall_eval.accuracy - b0_summary.get("final_average_accuracy", 0.0)) * 100
        print(f"Gain over Lower Bound (B0):   +{gain:.2f}% (B0: {b0_summary.get('final_average_accuracy', 0.0) * 100:.2f}%)")
    print(f"Artifacts saved in:           {results_dir}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
