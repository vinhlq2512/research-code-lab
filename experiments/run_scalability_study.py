#!/usr/bin/env python3
"""Runner for TAPTA Data Scalability Study: From Few-Shot to High-Resource Continual RE.

Supports:
- Scalable train shots (5, 10, 50, 100, and 'full').
- Zero leakage enforcement with ScalableFewRelDataset.
- Deterministic nested sampling.
- Resource usage tracking (GPU memory, training time, persistent memory).
- Oracle routing diagnostic (--oracle_diagnostic) to measure Routing Gap.
- Smoke test mode (--smoke_test) for instant pipeline verification.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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

from evaluation.performance_matrix import PerformanceMatrix
from evaluation.metrics import compute_continual_summary_metrics
from cl_re_baselines.checkpoint import CheckpointManager
from cl_re_baselines.logger import StreamingMetricsLogger, generate_conclusion_report
from generate_plots_svg import generate_svg_and_html

from src.data.protocol_v1_loader import (
    ProtocolV1Dataset,
    ProtocolV1Sample,
    ProtocolV1Task,
    collate_samples_for_bert,
)
from src.data.scalable_fewrel_loader import ScalableFewRelDataset, normalize_sentence_for_hash
from src.models.prompt_bert import PromptedBERTRelationClassifier
from src.models.task_router import TaskRouter

from experiments.run_p4 import (
    set_seed,
    load_yaml_config,
    get_git_info,
    train_single_hierarchical_task_and_router,
    evaluate_hierarchical_task,
)

logger = logging.getLogger(__name__)


def compute_persistent_memory_bytes(model: PromptedBERTRelationClassifier, router: TaskRouter) -> dict[str, int]:
    """Measure exact persistent non-parametric and parametric memory in bytes."""
    # 1. Prototype memory
    proto_bytes = 0
    if model.prototypes is not None:
        proto_bytes = model.prototypes.element_size() * model.prototypes.nelement()

    # 2. Router non-parametric buffers (centroids and subcentroids)
    router_bytes = 0
    for name, buf in router.named_buffers():
        router_bytes += buf.element_size() * buf.nelement()

    # 3. Prompt parameters (root + leaves)
    prompt_bytes = 0
    if hasattr(model, "root_prompt") and model.root_prompt is not None:
        prompt_bytes += model.root_prompt.element_size() * model.root_prompt.nelement()
    if hasattr(model, "leaf_prompts") and model.leaf_prompts is not None:
        for p in model.leaf_prompts:
            prompt_bytes += p.element_size() * p.nelement()

    total_persistent = proto_bytes + router_bytes + prompt_bytes
    return {
        "prototype_memory_bytes": proto_bytes,
        "router_memory_bytes": router_bytes,
        "prompt_parameter_bytes": prompt_bytes,
        "total_persistent_memory_bytes": total_persistent,
        "total_persistent_memory_kb": round(total_persistent / 1024, 2),
    }


def generate_dataset_manifest(tasks: list[ProtocolV1Task], seed: int, train_shot: Any) -> dict[str, Any]:
    """Generate reproducible dataset manifest with sample counts and SHA-256 fingerprint."""
    task_manifests = []
    hasher = hashlib.sha256()

    for t in tasks:
        train_texts = [s.marked_text for s in t.train_samples]
        test_texts = [s.marked_text for s in t.test_samples]

        for txt in train_texts + test_texts:
            hasher.update(txt.encode("utf-8"))

        task_manifests.append({
            "task_id": t.task_id,
            "cluster_id": t.cluster_id,
            "num_relations": len(t.relations),
            "relation_ids": t.relations,
            "relation_names": t.relation_names,
            "num_train_samples": len(t.train_samples),
            "num_test_samples": len(t.test_samples),
            "first_train_sentence": train_texts[0][:80] if train_texts else "",
        })

    return {
        "dataset_fingerprint_sha256": hasher.hexdigest(),
        "seed": seed,
        "train_shot": str(train_shot),
        "total_tasks": len(tasks),
        "total_train_samples": sum(t["num_train_samples"] for t in task_manifests),
        "total_test_samples": sum(t["num_test_samples"] for t in task_manifests),
        "tasks": task_manifests,
    }


def run_scalability_experiment(
    config_path: str | Path,
    device_override: str | None = None,
    smoke_test: bool = False,
    oracle_diagnostic: bool = False,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    exp_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    router_cfg = config.get("router", {})
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    seed = int(config.get("seed", 2021))
    set_seed(seed)

    if smoke_test:
        train_cfg["epochs_per_task"] = 1
        train_cfg["batch_size"] = 4
        print("[SMOKE TEST MODE ENABLED] 1 epoch/task, batch size 4")

    results_dir = REPO_ROOT / output_cfg.get(
        "results_dir", f"runs/tapta_scalability/fewrel_{dataset_cfg.get('train_shot', 5)}shot/seed_{seed}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = results_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    train_shot = dataset_cfg.get("train_shot", 5)
    print("==================================================================")
    print(f"TAPTA Data Scalability Study: {exp_cfg.get('id', 'DS')} ({train_shot}-shot)")
    print(f"Description: {exp_cfg.get('description')}")
    print(f"Seed:        {seed}")
    print(f"Results Dir: {results_dir}")
    print("==================================================================")

    # 1. Dataset Loading
    loader_type = dataset_cfg.get("loader_type", "scalable")
    if loader_type == "scalable":
        dataset = ScalableFewRelDataset(
            data_json_path=REPO_ROOT / dataset_cfg.get("data_json_path", "paper-reproductions/WAVE-CRE/datasets/data_with_marker.json"),
            base_dir=REPO_ROOT / dataset_cfg.get("base_dir", "data/protocol_v1/fewrel"),
            split_subdir=dataset_cfg.get("split_subdir", "CFRLdata_10_100_10_5"),
            train_shot=train_shot,
            test_shot=int(dataset_cfg.get("test_shot", 100)),
            full_train_limit=int(dataset_cfg.get("full_train_limit", 420)),
            seed=seed,
            task_order_strategy=dataset_cfg.get("task_order_strategy", "conpl_seed_shuffle"),
        )
    else:
        dataset = ProtocolV1Dataset(
            base_dir=REPO_ROOT / dataset_cfg.get("base_dir", "data/protocol_v1/fewrel"),
            split_subdir=dataset_cfg.get("split_subdir", "CFRLdata_10_100_10_5"),
            task_order_strategy=dataset_cfg.get("task_order_strategy", "conpl_seed_shuffle"),
            seed=seed,
            train_shot=int(train_shot),
            test_shot=int(dataset_cfg.get("test_shot", 100)),
            max_seq_length=int(model_cfg.get("max_seq_length", 128)),
        )

    tasks = dataset.build_tasks()
    if smoke_test:
        tasks = tasks[:2]  # Limit to 2 tasks for smoke test
    num_tasks = len(tasks)
    relations_per_task = int(dataset_cfg.get("relations_per_task", 10))

    # Save dataset manifest
    manifest = generate_dataset_manifest(tasks, seed=seed, train_shot=train_shot)
    with (results_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Dataset Manifest] Fingerprint: {manifest['dataset_fingerprint_sha256'][:16]}... Saved.")

    # 2. Instantiate Model and Router
    target_device = device_override or train_cfg.get("device", "auto")
    prompt_length = int(model_cfg.get("prompt_length", 8))
    root_prompt_length = int(model_cfg.get("root_prompt_length", 4))

    model = PromptedBERTRelationClassifier(
        backbone_name=model_cfg.get("backbone", "bert-base-uncased"),
        prompt_mode="hierarchical_tree",
        prompt_length=prompt_length,
        root_prompt_length=root_prompt_length,
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
    )

    print(f"Model initialized on device: {model.target_device}")

    # Matrices
    fusion_matrix = PerformanceMatrix(num_tasks=num_tasks)
    gated_proto_matrix = PerformanceMatrix(num_tasks=num_tasks)
    router_top1_matrix = PerformanceMatrix(num_tasks=num_tasks)
    router_topk_matrix = PerformanceMatrix(num_tasks=num_tasks)
    tree_top1_matrix = PerformanceMatrix(num_tasks=num_tasks)
    f1_matrix = PerformanceMatrix(num_tasks=num_tasks)

    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    task_training_times = []
    start_total_time = time.time()
    eval_batch_size = int(config.get("evaluation", {}).get("batch_size", 64))

    # 3. Continual Learning Loop (Tasks 0..num_tasks-1)
    seen_classes = []
    for stage, task in enumerate(tasks):
        t_id = task.task_id
        seen_classes.extend(task.class_indices)
        print(f"\n==================== [Stage {stage + 1}/{num_tasks} — Task {t_id + 1}] ====================")
        print(f"Relations: {task.relation_names}")
        print(f"Train samples: {len(task.train_samples)} | Test samples: {len(task.test_samples)}")

        # Train task prompt and router
        t_start = time.time()
        train_stats = train_single_hierarchical_task_and_router(
            model=model,
            router=router,
            task=task,
            config=config,
        )
        t_dur = time.time() - t_start
        task_training_times.append(round(t_dur, 2))
        print(f"  Stage {stage + 1} trained in {t_dur:.2f}s | Prompt loss: {train_stats['prompt_loss']:.4f}")

        # Evaluate on all seen tasks
        for j in range(stage + 1):
            past_task = tasks[j]
            eval_res = evaluate_hierarchical_task(
                model=model,
                router=router,
                test_samples=past_task.test_samples,
                true_task_id=j,
                seen_tasks=stage + 1,
                config=config,
                batch_size=eval_batch_size,
            )

            fusion_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["decision_fusion_accuracy"])
            gated_proto_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["tree_top1_gated_prototype_accuracy"])
            tree_top1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["tree_top1_prototype_accuracy"])
            router_top1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["router_top1_accuracy"])
            router_topk_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["router_topk_coverage"])
            f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["macro_f1"])

            print(
                f"    Eval on T{j + 1}: Router Top-1={eval_res['router_top1_accuracy'] * 100:5.2f}% | "
                f"Top-K Cov={eval_res['router_topk_coverage'] * 100:5.2f}% | "
                f"Tree Top-1={eval_res['tree_top1_prototype_accuracy'] * 100:5.2f}% | "
                f"Fusion={eval_res['decision_fusion_accuracy'] * 100:5.2f}%"
            )

        # Checkpoint
        checkpoint_manager.save_checkpoint(
            stage=stage,
            model=model,
            metadata={
                "task_id": task.task_id,
                "cluster_id": task.cluster_id,
                "seen_classes": seen_classes,
                "top_k": router_cfg.get("top_k", 2),
            },
        )

    total_training_time = round(time.time() - start_total_time, 2)

    # 4. Resource & Persistent Memory Profiling
    mem_profile = compute_persistent_memory_bytes(model, router)
    peak_gpu_mem = 0.0
    if torch.cuda.is_available():
        peak_gpu_mem = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    resource_usage = {
        "device": str(model.target_device),
        "total_training_time_seconds": total_training_time,
        "task_training_times_seconds": task_training_times,
        "average_task_time_seconds": round(sum(task_training_times) / max(1, len(task_training_times)), 2),
        "peak_gpu_memory_mb": peak_gpu_mem,
        **mem_profile,
    }

    with (results_dir / "resource_usage.json").open("w", encoding="utf-8") as f:
        json.dump(resource_usage, f, indent=2)

    # 5. Summarize Metrics
    fusion_summary = compute_continual_summary_metrics(fusion_matrix)
    gated_summary = compute_continual_summary_metrics(gated_proto_matrix)
    router_summary = compute_continual_summary_metrics(router_top1_matrix)

    summary = {
        "experiment_id": exp_cfg.get("id"),
        "train_shot": train_shot,
        "final_decision_fusion_accuracy": fusion_summary["final_average_accuracy"],
        "final_gated_proto_accuracy": gated_summary["final_average_accuracy"],
        "average_incremental_accuracy": fusion_summary["average_incremental_accuracy"],
        "average_forgetting": fusion_summary["average_forgetting"],
        "router_top1_accuracy": router_summary["final_average_accuracy"],
        "total_training_time_seconds": total_training_time,
        "persistent_memory_kb": mem_profile["total_persistent_memory_kb"],
    }

    if oracle_diagnostic and oracle_matrix is not None:
        oracle_summary = compute_continual_summary_metrics(oracle_matrix)
        summary["oracle_accuracy"] = oracle_summary["final_average_accuracy"]
        summary["routing_gap"] = round(summary["oracle_accuracy"] - summary["final_decision_fusion_accuracy"], 4)

    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fusion_matrix.export_csv(results_dir / "performance_matrix.csv")

    print("\n==================================================================")
    print(f"Scalability Run Complete: {exp_cfg.get('id')} ({train_shot}-shot)")
    print(f"Final Decision Fusion Accuracy:  {summary['final_decision_fusion_accuracy'] * 100:.2f}%")
    print(f"Final Gated Proto Accuracy:      {summary['final_gated_proto_accuracy'] * 100:.2f}%")
    print(f"Average Incremental Acc (AIA):   {summary['average_incremental_accuracy'] * 100:.2f}%")
    print(f"Average Forgetting:              {summary['average_forgetting'] * 100:.2f}%")
    print(f"Router Top-1 Accuracy:           {summary['router_top1_accuracy'] * 100:.2f}%")
    if "routing_gap" in summary:
        print(f"Oracle Accuracy:                 {summary['oracle_accuracy'] * 100:.2f}%")
        print(f"Routing Gap:                     {summary['routing_gap'] * 100:.2f}%")
    print(f"Training Time:                   {total_training_time}s")
    print(f"Persistent Memory:               {mem_profile['total_persistent_memory_kb']} KB")
    print(f"Artifacts saved to:              {results_dir}")
    print("==================================================================")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TAPTA Data Scalability Study")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cuda', 'mps', 'cpu')")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 2-task smoke test")
    parser.add_argument("--oracle_diagnostic", action="store_true", help="Compute Oracle Routing Diagnostic")
    args = parser.parse_args()

    run_scalability_experiment(
        config_path=args.config,
        device_override=args.device,
        smoke_test=args.smoke_test,
        oracle_diagnostic=args.oracle_diagnostic,
    )


if __name__ == "__main__":
    main()
