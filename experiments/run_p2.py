#!/usr/bin/env python3
"""Runner for Proposed Method P2: Task-Aware Router (P2a Centroid Cosine & P2b Learned MLP Router).

Protocol: Protocol v1 Track A (FewRel 8 tasks x 10 relations, strict 5-shot, M=0)
Backbone: 100% Frozen BERT-base-uncased with task-routed soft prompt pool.
Inference: Strict Class-Incremental Learning (Zero Oracle Task ID at test time).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
from src.models.task_router import TaskRouter

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


def train_single_task_and_router(
    model: PromptedBERTRelationClassifier,
    router: TaskRouter,
    task: ProtocolV1Task,
    config: dict[str, Any],
) -> dict[str, float]:
    """Train task-specific soft prompt P_t, compute class prototypes, and update TaskRouter."""
    train_cfg = config.get("training", {})
    router_cfg = config.get("router", {})
    max_seq_len = int(config.get("model", {}).get("max_seq_length", 128))
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs_per_task", 15))
    lr = float(train_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))

    train_samples = list(task.train_samples)
    n_samples = len(train_samples)

    # 1. Train task-specific prompt P_t and classifier head
    trainable_params = model.get_trainable_params(task_id=task.task_id)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, (n_samples // batch_size + (1 if n_samples % batch_size else 0)) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.classifier.train()
    prompt_loss = 0.0

    for epoch in range(epochs):
        perm = list(range(n_samples))
        random.shuffle(perm)
        epoch_loss = 0.0
        num_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            batch_samples = [train_samples[i] for i in batch_indices]

            batch_data = collate_samples_for_bert(
                batch_samples,
                tokenizer=model.tokenizer,
                device=model.target_device,
                max_seq_length=max_seq_len,
            )

            optimizer.zero_grad()
            out = model(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                e1_indices=batch_data["e1_indices"],
                e2_indices=batch_data["e2_indices"],
                task_id=task.task_id,
            )

            loss = F.cross_entropy(out["logits"], batch_data["labels"])
            loss.backward()

            nn.utils.clip_grad_norm_(trainable_params, clip_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        prompt_loss = epoch_loss / max(1, num_batches)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    [Prompt P_{task.task_id + 1}] Epoch {epoch + 1:2d}/{epochs} | Loss: {prompt_loss:.4f}")

    # 2. Extract prompted class prototypes for NCM / Prototype Classifier
    model.eval()
    with torch.no_grad():
        for class_idx in task.class_indices:
            class_samples = [s for s in train_samples if s.class_idx == class_idx]
            if class_samples:
                c_data = collate_samples_for_bert(
                    class_samples,
                    tokenizer=model.tokenizer,
                    device=model.target_device,
                    max_seq_length=max_seq_len,
                )
                h_r = model.extract_features(
                    c_data["input_ids"],
                    c_data["attention_mask"],
                    c_data["e1_indices"],
                    c_data["e2_indices"],
                    task_id=task.task_id,
                )
                model.update_prototypes(class_idx, h_r)

    # 3. Extract unprompted stationary query vectors q(x) and update TaskRouter
    with torch.no_grad():
        full_task_data = collate_samples_for_bert(
            train_samples,
            tokenizer=model.tokenizer,
            device=model.target_device,
            max_seq_length=max_seq_len,
        )
        task_queries = model.extract_unprompted_query(
            full_task_data["input_ids"],
            full_task_data["attention_mask"],
            full_task_data["e1_indices"],
            full_task_data["e2_indices"],
        )
        router.register_task_queries(
            task_id=task.task_id,
            query_features=task_queries,
            class_indices=task.class_indices,
            sample_labels=full_task_data["labels"],
        )

    router_loss = 0.0
    if router.router_type == "learned_mlp":
        router_loss = router.fit_learned_router(
            current_task_id=task.task_id,
            current_queries=task_queries,
            epochs=int(router_cfg.get("router_epochs", 30)),
            lr=float(router_cfg.get("router_lr", 2.0e-3)),
            synthetic_samples_per_rel=int(router_cfg.get("synthetic_samples_per_rel", 12)),
        )
        print(f"    [Router P2b] Trained MLP Cosine Router across {task.task_id + 1} tasks | Final Loss: {router_loss:.4f}")

    return {"prompt_loss": prompt_loss, "router_loss": router_loss}


@torch.no_grad()
def evaluate_routed_task(
    model: PromptedBERTRelationClassifier,
    router: TaskRouter,
    test_samples: list[ProtocolV1Sample],
    true_task_id: int,
    seen_tasks: int,
    config: dict[str, Any],
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate model on a task's test set using predicted Task IDs from TaskRouter (Zero Oracle)."""
    model.eval()
    router.eval()
    max_seq_len = int(config.get("model", {}).get("max_seq_length", 128))
    relations_per_task = int(config.get("dataset", {}).get("relations_per_task", 10))

    all_logits: list[torch.Tensor] = []
    all_reps: list[torch.Tensor] = []
    all_pred_tasks: list[torch.Tensor] = []
    all_labels: list[int] = []

    for i in range(0, len(test_samples), batch_size):
        chunk = test_samples[i : i + batch_size]
        batch_data = collate_samples_for_bert(
            chunk,
            tokenizer=model.tokenizer,
            device=model.target_device,
            max_seq_length=max_seq_len,
        )

        # Stage 1: Unprompted query extraction & Task Routing (NO Oracle Task ID used!)
        q_r = model.extract_unprompted_query(
            batch_data["input_ids"],
            batch_data["attention_mask"],
            batch_data["e1_indices"],
            batch_data["e2_indices"],
        )
        route_out = router.predict_task(q_r, seen_tasks=seen_tasks)
        pred_task_ids = route_out["pred_task_ids"]  # [B]

        # Stage 2: Extract relation features using each sample's predicted prompt P_{\hat{t}}
        out = model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            e1_indices=batch_data["e1_indices"],
            e2_indices=batch_data["e2_indices"],
            task_id=pred_task_ids,
        )

        all_logits.append(out["logits"])
        all_reps.append(out["representations"])
        all_pred_tasks.append(pred_task_ids)
        all_labels.extend(batch_data["labels"].cpu().tolist())

    logits = torch.cat(all_logits, dim=0)
    reps = torch.cat(all_reps, dim=0)
    pred_tasks_tensor = torch.cat(all_pred_tasks, dim=0)
    pred_tasks_list = pred_tasks_tensor.cpu().tolist()

    # 1. Router Accuracy & Confusion Counts
    true_tasks_list = [true_task_id] * len(all_labels)
    router_acc = compute_accuracy(true_tasks_list, pred_tasks_list)
    confusion_counts = [0] * seen_tasks
    for pt in pred_tasks_list:
        if 0 <= int(pt) < seen_tasks:
            confusion_counts[int(pt)] += 1

    # 2. Global Prototype Classifier Accuracy (over all seen classes)
    proto_logits = model.predict_prototype(reps, seen_classes_only=True)
    proto_preds = torch.argmax(proto_logits, dim=-1).cpu().tolist()
    proto_acc = compute_accuracy(all_labels, proto_preds)
    proto_f1 = compute_macro_f1(all_labels, proto_preds)

    # 3. Routed-Gated Prototype Classifier Accuracy (gated to predicted task's classes)
    routed_proto_logits = model.predict_routed_prototype(
        reps, pred_tasks_tensor, relations_per_task=relations_per_task
    )
    routed_proto_preds = torch.argmax(routed_proto_logits, dim=-1).cpu().tolist()
    routed_proto_acc = compute_accuracy(all_labels, routed_proto_preds)

    # 4. Linear Head CIL Accuracy
    cil_preds = torch.argmax(logits, dim=-1).cpu().tolist()
    cil_acc = compute_accuracy(all_labels, cil_preds)

    return {
        "router_accuracy": router_acc,
        "prototype_accuracy": proto_acc,
        "routed_prototype_accuracy": routed_proto_acc,
        "macro_f1": proto_f1,
        "linear_cil_accuracy": cil_acc,
        "sample_count": len(all_labels),
        "confusion_counts": confusion_counts,
    }


def export_confusion_matrix_csv(matrix: list[list[float]], path: Path) -> None:
    num_tasks = len(matrix)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["true_task \\ pred_task"] + [f"T{j + 1}" for j in range(num_tasks)]
        writer.writerow(header)
        for i, row in enumerate(matrix):
            writer.writerow([f"T{i + 1}"] + [f"{val:.4f}" for val in row])


def run_experiment(config_path: str | Path, device_override: str | None = None) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    seed = int(config.get("seed", 2021))
    set_seed(seed)

    exp_cfg = config.get("experiment", {})
    exp_id = exp_cfg.get("id", "P2a")
    model_cfg = config.get("model", {})
    router_cfg = config.get("router", {})
    continual_cfg = config.get("continual", {})
    output_cfg = config.get("output", {})

    results_dir = REPO_ROOT / output_cfg.get("results_dir", f"runs/{exp_id.lower()}/fewrel_5shot/seed_{seed}")
    results_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Router Type: {router_cfg.get('type', 'centroid')}")
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
    relations_per_task = int(dataset_cfg.get("relations_per_task", 10))
    print(f"\n[1/4] Loaded Protocol v1 Dataset: {num_tasks} continual tasks.")

    # 2. Instantiate Prompted BERT & TaskRouter
    target_device = device_override or config.get("training", {}).get("device", "auto")
    prompt_mode = model_cfg.get("prompt_mode", "routed_pool")
    prompt_length = int(model_cfg.get("prompt_length", 8))

    model = PromptedBERTRelationClassifier(
        backbone_name=model_cfg.get("backbone", "bert-base-uncased"),
        prompt_mode=prompt_mode,
        prompt_length=prompt_length,
        num_tasks=num_tasks,
        num_classes=int(model_cfg.get("num_classes", 80)),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.1)),
        device=target_device,
    )

    router = TaskRouter(
        input_dim=model.rep_dim,
        num_tasks=num_tasks,
        relations_per_task=relations_per_task,
        router_type=router_cfg.get("type", "centroid"),
        subcentroid_weight=float(router_cfg.get("subcentroid_weight", 0.7)),
        hidden_dim=int(router_cfg.get("hidden_dim", 512)),
        temperature=float(router_cfg.get("temperature", 16.0)),
        residual_anchor_weight=float(router_cfg.get("residual_anchor_weight", 0.5)),
    ).to(model.target_device)

    print(f"\n[2/4] Initialized Model & TaskRouter (type={router.router_type}) on device={model.target_device}")

    # 3. Setup Performance Matrices
    proto_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="prototype_accuracy", dataset="fewrel", seed=seed)
    routed_proto_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="routed_prototype_accuracy", dataset="fewrel", seed=seed
    )
    router_acc_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="router_accuracy", dataset="fewrel", seed=seed)
    f1_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="macro_f1", dataset="fewrel", seed=seed)
    linear_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="linear_cil_accuracy", dataset="fewrel", seed=seed)

    checkpoint_manager = CheckpointManager(results_dir / "checkpoints")
    metrics_logger = StreamingMetricsLogger(results_dir)

    final_confusion_matrix: list[list[float]] = [[0.0] * num_tasks for _ in range(num_tasks)]

    # 4. Continual Learning Loop
    print("\n[3/4] Starting Continual Learning & Routed Evaluation across 8 tasks...")
    start_time = time.time()

    for stage in range(num_tasks):
        task = tasks[stage]
        seen_tasks = stage + 1
        seen_classes = seen_tasks * relations_per_task
        model.set_seen_classes(seen_classes)

        print(
            f"\n>>> Stage {seen_tasks}/{num_tasks}: Learning Task {seen_tasks} "
            f"(Cluster {task.cluster_id}, Seen Classes: {seen_classes})"
        )
        stage_t0 = time.time()

        train_single_task_and_router(
            model=model,
            router=router,
            task=task,
            config=config,
        )

        print(f"  Evaluating on tasks 1..{seen_tasks} (Task-Agnostic Routed Inference):")
        for j in range(seen_tasks):
            eval_res = evaluate_routed_task(
                model=model,
                router=router,
                test_samples=tasks[j].test_samples,
                true_task_id=j,
                seen_tasks=seen_tasks,
                config=config,
                batch_size=int(config.get("evaluation", {}).get("batch_size", 64)),
            )

            proto_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["prototype_accuracy"])
            routed_proto_matrix.set_score(
                trained_until=stage, evaluated_task=j, score=eval_res["routed_prototype_accuracy"]
            )
            router_acc_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["router_accuracy"])
            f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["macro_f1"])
            linear_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["linear_cil_accuracy"])

            if stage == num_tasks - 1:
                total_j = max(1, eval_res["sample_count"])
                for k_idx, cnt in enumerate(eval_res["confusion_counts"]):
                    final_confusion_matrix[j][k_idx] = cnt / total_j

            metrics_logger.log_evaluation(
                stage=stage,
                test_task=j,
                accuracy=eval_res["prototype_accuracy"],
                macro_f1=eval_res["macro_f1"],
                sample_count=eval_res["sample_count"],
            )

            print(
                f"    Eval on T{j + 1}: Router Acc = {eval_res['router_accuracy'] * 100:6.2f}% | "
                f"Proto Acc = {eval_res['prototype_accuracy'] * 100:6.2f}% | "
                f"Routed-Gated Acc = {eval_res['routed_prototype_accuracy'] * 100:6.2f}%"
            )

        # Export intermediate matrices
        proto_matrix.export_csv(results_dir / "performance_matrix.csv")
        proto_matrix.export_json(results_dir / "performance_matrix.json")
        proto_matrix.export_csv(results_dir / "prototype_performance_matrix.csv")
        routed_proto_matrix.export_csv(results_dir / "routed_prototype_performance_matrix.csv")
        router_acc_matrix.export_csv(results_dir / "router_accuracy_matrix.csv")
        f1_matrix.export_csv(results_dir / "f1_performance_matrix.csv")

        checkpoint_manager.save_checkpoint(
            stage=stage,
            model=model,
            metadata={
                "task_id": task.task_id,
                "cluster_id": task.cluster_id,
                "seen_classes": seen_classes,
                "router_type": router.router_type,
            },
        )
        print(f"  Stage {seen_tasks} completed in {time.time() - stage_t0:.2f}s.")

    total_duration = time.time() - start_time
    print(f"\n[4/4] Continual Learning & Routing Finished in {total_duration:.2f}s!")

    # 5. Compute Summary & Gap Closed Metrics
    proto_summary = compute_continual_summary_metrics(proto_matrix)
    routed_proto_summary = compute_continual_summary_metrics(routed_proto_matrix)
    router_summary = compute_continual_summary_metrics(router_acc_matrix)
    linear_summary = compute_continual_summary_metrics(linear_matrix)

    ref_p0 = float(continual_cfg.get("reference_p0_proto_acc", 0.52825))
    ref_p1 = float(continual_cfg.get("reference_p1_proto_acc", 0.701375))
    oracle_span = max(1e-6, ref_p1 - ref_p0)

    final_proto_acc = proto_summary["final_average_accuracy"]
    final_routed_proto_acc = routed_proto_summary["final_average_accuracy"]
    best_proto_acc = max(final_proto_acc, final_routed_proto_acc)
    gap_closed_pct = ((best_proto_acc - ref_p0) / oracle_span) * 100.0

    # Export Final Router Confusion Matrix
    export_confusion_matrix_csv(final_confusion_matrix, results_dir / "router_confusion_matrix.csv")
    with (results_dir / "router_confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": exp_id,
                "router_type": router.router_type,
                "final_router_accuracy": router_summary["final_average_accuracy"],
                "confusion_matrix": final_confusion_matrix,
            },
            f,
            indent=2,
        )

    summary = dict(proto_summary)
    summary["final_prototype_accuracy"] = final_proto_acc
    summary["final_routed_prototype_accuracy"] = final_routed_proto_acc
    summary["final_router_accuracy"] = router_summary["final_average_accuracy"]
    summary["router_average_incremental_accuracy"] = router_summary["average_incremental_accuracy"]
    summary["oracle_gap_closed_pct"] = round(gap_closed_pct, 2)
    summary["experiment"] = exp_id
    summary["method"] = exp_cfg.get("name")
    summary["router_type"] = router.router_type
    summary["seed"] = seed
    summary["total_duration_seconds"] = round(total_duration, 2)

    metrics_logger.save_summary(summary)

    metrics_json_path = results_dir / "metrics.json"
    full_metrics = {
        "experiment_id": exp_id,
        "method": exp_cfg.get("name"),
        "role": exp_cfg.get("baseline_role"),
        "router_type": router.router_type,
        "seed": seed,
        "protocol": "Protocol v1 Track A (FewRel 5-shot, Rehearsal-Free M=0)",
        "memory_size": 0,
        "task_oracle_inference": False,
        "prototype_accuracy_whole": final_proto_acc,
        "routed_prototype_accuracy_whole": final_routed_proto_acc,
        "router_accuracy_whole": router_summary["final_average_accuracy"],
        "router_average_incremental_accuracy": router_summary["average_incremental_accuracy"],
        "average_incremental_accuracy": proto_summary["average_incremental_accuracy"],
        "average_forgetting": proto_summary["average_forgetting"],
        "backward_transfer": proto_summary["backward_transfer"],
        "linear_accuracy_whole": linear_summary["final_average_accuracy"],
        "reference_p0_proto_acc": ref_p0,
        "reference_p1_proto_acc": ref_p1,
        "oracle_gap_closed_pct": round(gap_closed_pct, 2),
        "proto_matrix_A": proto_matrix.as_list(),
        "routed_proto_matrix_A": routed_proto_matrix.as_list(),
        "router_acc_matrix_A": router_acc_matrix.as_list(),
        "router_confusion_matrix": final_confusion_matrix,
        "execution_duration_sec": total_duration,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  Saved {metrics_json_path}")

    provenance = {
        "git": get_git_info(),
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

    generate_conclusion_report(summary, proto_matrix.as_list(), results_dir / "conclusion.md")

    try:
        generate_svg_and_html(results_dir)
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    print("\n==================================================================")
    print(f"FINAL ROUTED METRICS SUMMARY: {exp_id} (Router: {router.router_type})")
    print(f"  Final Router Accuracy (Task ID):       {router_summary['final_average_accuracy'] * 100:.2f}%")
    print(f"  Final Prototype Accuracy Whole:        {final_proto_acc * 100:.2f}%")
    print(f"  Final Routed-Gated Prototype Accuracy: {final_routed_proto_acc * 100:.2f}%")
    print(f"  Average Incremental Accuracy (AIA):    {proto_summary['average_incremental_accuracy'] * 100:.2f}%")
    print(f"  Average Forgetting (F_T):              {proto_summary['average_forgetting'] * 100:.2f}%")
    print(f"  Oracle Gap Closed (P0 -> P1):          {gap_closed_pct:.2f}%")
    print("==================================================================")

    return full_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="P2 Task-Aware Router Continual Relation Extraction Runner.")
    parser.add_argument("--config", type=str, required=True, help="Path to P2 YAML config file.")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (e.g. cuda, mps, cpu).")
    args = parser.parse_args()

    run_experiment(config_path=args.config, device_override=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
