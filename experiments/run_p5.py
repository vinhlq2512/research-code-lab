#!/usr/bin/env python3
"""Runner for Proposed Method P5: TAPTA Contrastive-Margin Tree & Anisotropy-Resistant Router.

Ablation modes:
- p4_baseline: Standard P4 (Hierarchical Tree + Soft Fusion)
- p5_a:        + Module A (Whitened Anisotropy-Resistant Router)
- p5_b:        + Module A + Module C (Confidence-Gated Candidate Masking)
- p5_c:        + Module A + Module C + Module B (Supervised Prototype Contrastive Loss)
- p5_full:     Full P5 (+ Module D: Momentum Root Prompt Anchor Regularization)

Strict Invariants:
- 100% Frozen BERT-base Backbone (109.5M, requires_grad=False)
- Strict Equal 5-shot (including Task 1)
- Rehearsal-Free M=0 (No raw text, no generative replay)
- Task-Agnostic Class-Incremental Inference (Zero Oracle Task ID)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

# Search paths
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
from cl_re_baselines.logger import StreamingMetricsLogger

from src.data.protocol_v1_loader import (
    ProtocolV1Dataset,
    ProtocolV1Sample,
    ProtocolV1Task,
    collate_samples_for_bert,
)
from src.models.p5_prompt_bert import P5PromptedBERT
from src.models.p5_router import P5TaskRouter

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


def train_single_p5_task(
    model: P5PromptedBERT,
    router: P5TaskRouter,
    task: ProtocolV1Task,
    config: dict[str, Any],
    mode: str = "p5_full",
) -> dict[str, float]:
    """Train task prompt with optional contrastive prototype loss and momentum root anchor."""
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    max_seq_len = int(model_cfg.get("max_seq_length", 128))
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs_per_task", 15))
    lr = float(train_cfg.get("learning_rate", 1.0e-3))
    root_lr_mult = float(train_cfg.get("root_lr_multiplier", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    orth_lambda = float(train_cfg.get("orthogonality_lambda", 0.05))
    con_lambda = float(model_cfg.get("proto_contrastive_lambda", 0.2))
    anchor_lambda = float(model_cfg.get("root_anchor_lambda", 0.1))

    # Feature flags from mode
    enable_contrastive = mode in ("p5_c", "p5_full")
    enable_momentum_root = mode == "p5_full"

    train_samples = list(task.train_samples)
    n_samples = len(train_samples)

    # 1. Parameter groups
    param_groups: list[dict[str, Any]] = [
        {"params": list(model.classifier.parameters()), "lr": lr},
        {"params": [model.leaf_prompts[task.task_id]], "lr": lr},
    ]
    if task.task_id == 0:
        param_groups.append({"params": [model.root_prompt], "lr": lr})
    elif enable_momentum_root:
        param_groups.append({"params": [model.root_prompt], "lr": lr * root_lr_mult})

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    total_steps = max(1, (n_samples // batch_size + (1 if n_samples % batch_size else 0)) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    model.classifier.train()
    for epoch in range(epochs):
        perm = list(range(n_samples))
        random.shuffle(perm)
        for start_idx in range(0, n_samples, batch_size):
            batch_samples = [train_samples[i] for i in perm[start_idx : start_idx + batch_size]]
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

            loss_ce = F.cross_entropy(out["logits"], batch_data["labels"])
            loss_orth = model.compute_orthogonality_loss(task.task_id)
            total_loss = loss_ce + orth_lambda * loss_orth

            # Module B: Supervised Prototype Contrastive Loss
            if enable_contrastive and model.seen_classes > 0:
                loss_con = model.compute_prototype_contrastive_loss(
                    h_r=out["representations"],
                    labels=batch_data["labels"],
                )
                total_loss = total_loss + con_lambda * loss_con

            # Module D: Momentum Root Prompt Regularization
            if enable_momentum_root and task.task_id > 0:
                loss_anchor = model.compute_root_anchor_loss()
                total_loss = total_loss + anchor_lambda * loss_anchor

            total_loss.backward()
            all_params = [p for g in param_groups for p in g["params"]]
            nn.utils.clip_grad_norm_(all_params, clip_norm)
            optimizer.step()
            scheduler.step()

    # Anchor root prompt after Task 0
    if task.task_id == 0:
        model.anchor_root_prompt()
        if not enable_momentum_root:
            model.root_prompt.requires_grad = False

    # 2. Extract prompted prototypes for current task classes
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

    # 3. Register stationary unprompted query statistics into Router
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

        # Update whitening transform if Module A is active
        seen_tasks = task.task_id + 1
        router.update_whitening_transform(seen_tasks=seen_tasks)

    return {}


@torch.no_grad()
def evaluate_p5_task(
    model: P5PromptedBERT,
    router: P5TaskRouter,
    test_samples: list[ProtocolV1Sample],
    true_task_id: int,
    seen_tasks: int,
    config: dict[str, Any],
    mode: str = "p5_full",
    batch_size: int = 64,
    task_class_indices: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate task using Confidence-Gated Candidate Masking and monitor diagnostics."""
    model.eval()
    router.eval()
    max_seq_len = int(config.get("model", {}).get("max_seq_length", 128))
    relations_per_task = int(config.get("dataset", {}).get("relations_per_task", 10))
    conf_thresh = float(config.get("router", {}).get("confidence_threshold", 0.05))
    temperature = float(config.get("router", {}).get("temperature", 0.2))

    enable_gating = mode in ("p5_b", "p5_c", "p5_full")

    all_p5_preds: list[int] = []
    all_hard_top1_preds: list[int] = []
    all_soft_fusion_preds: list[int] = []
    all_pred_tasks: list[int] = []
    all_topk_tasks: list[list[int]] = []
    all_labels: list[int] = []

    for i in range(0, len(test_samples), batch_size):
        chunk = test_samples[i : i + batch_size]
        batch_data = collate_samples_for_bert(
            chunk,
            tokenizer=model.tokenizer,
            device=model.target_device,
            max_seq_length=max_seq_len,
        )

        # 1. Unprompted stationary queries & Router confidence
        q_r = model.extract_unprompted_query(
            batch_data["input_ids"],
            batch_data["attention_mask"],
            batch_data["e1_indices"],
            batch_data["e2_indices"],
        )
        routing_out = router.get_confidence_routing(
            query_features=q_r,
            seen_tasks=seen_tasks,
            top_k=2,
        )
        topk_indices = routing_out["topk_indices"]  # [B, 2]
        router_scores = routing_out["task_scores"]  # [B, seen_tasks]
        pred_task_1 = topk_indices[:, 0]  # [B]

        # 2. Extract prompted features for Top-K candidate prompts
        cand_sims: list[torch.Tensor] = []
        for k_idx in range(topk_indices.size(1)):
            c_ids = topk_indices[:, k_idx]
            h_c = model.extract_features(
                batch_data["input_ids"],
                batch_data["attention_mask"],
                batch_data["e1_indices"],
                batch_data["e2_indices"],
                task_id=c_ids,
            )
            sim_c = model.predict_prototype(h_c, seen_classes_only=True)
            cand_sims.append(sim_c)

        # 3. Soft Decision Fusion baseline (D05)
        top2_scores = routing_out["topk_scores"]
        top2_w = F.softmax(top2_scores / max(temperature, 1e-4), dim=-1).unsqueeze(-1)  # [B, 2, 1]
        stacked_sims = torch.stack(cand_sims, dim=1)  # [B, 2, C]
        soft_fused_sim = torch.sum(top2_w * stacked_sims, dim=1)  # [B, C]

        # 4. Hard Top-1 baseline (D04)
        hard_top1_sim = cand_sims[0]

        # 5. P5 Confidence-Gated Candidate Masking (Module C)
        if enable_gating:
            final_p5_sim = model.predict_confidence_gated_prototype(
                candidate_sims=cand_sims,
                topk_task_ids=topk_indices,
                router_scores=router_scores,
                relations_per_task=relations_per_task,
                confidence_margin_threshold=conf_thresh,
                temperature=temperature,
                task_class_indices=task_class_indices,
            )
        else:
            final_p5_sim = soft_fused_sim

        all_p5_preds.extend(torch.argmax(final_p5_sim, dim=-1).cpu().tolist())
        all_hard_top1_preds.extend(torch.argmax(hard_top1_sim, dim=-1).cpu().tolist())
        all_soft_fusion_preds.extend(torch.argmax(soft_fused_sim, dim=-1).cpu().tolist())
        all_pred_tasks.extend(pred_task_1.cpu().tolist())
        all_topk_tasks.extend(topk_indices.cpu().tolist())
        all_labels.extend(batch_data["labels"].cpu().tolist())

    p5_acc = compute_accuracy(all_labels, all_p5_preds)
    p5_f1 = compute_macro_f1(all_labels, all_p5_preds)
    hard_top1_acc = compute_accuracy(all_labels, all_hard_top1_preds)
    soft_fusion_acc = compute_accuracy(all_labels, all_soft_fusion_preds)

    true_tasks = [true_task_id] * len(all_labels)
    router_top1_acc = compute_accuracy(true_tasks, all_pred_tasks)
    router_top2_cov = sum(1 for topk in all_topk_tasks if true_task_id in topk) / max(1, len(all_labels))

    return {
        "p5_accuracy": p5_acc,
        "p5_macro_f1": p5_f1,
        "hard_top1_accuracy": hard_top1_acc,
        "soft_fusion_accuracy": soft_fusion_acc,
        "router_top1_accuracy": router_top1_acc,
        "router_top2_coverage": router_top2_cov,
        "sample_count": len(all_labels),
    }


def run_experiment(
    config_path: str | Path,
    mode: str = "p5_full",
    device_override: str | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Execute complete continual learning workflow for P5."""
    config = load_yaml_config(config_path)
    exp_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    continual_cfg = config.get("continual", {})
    router_cfg = config.get("router", {})
    dataset_cfg = config.get("dataset", {})

    seed = seed_override if seed_override is not None else int(config.get("seed", 2021))
    set_seed(seed)

    dataset_name = dataset_cfg.get("name", "tacred")
    num_tasks = int(dataset_cfg.get("num_tasks", 8))
    relations_per_task = int(dataset_cfg.get("relations_per_task", 5 if dataset_name == "tacred" else 10))
    total_relations = int(dataset_cfg.get("total_relations", 40 if dataset_name == "tacred" else 80))

    device = device_override if device_override is not None else (
        "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )

    results_dir = REPO_ROOT / "runs" / f"p5_{mode}" / f"{dataset_name}_5shot" / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TAPTA P5 EXPERIMENT RUNNER — Mode: {mode.upper()} ({dataset_name.upper()} 5-shot, Seed: {seed})")
    print(f"Target Device: {device} | Output Directory: {results_dir}")
    print("=" * 80 + "\n")

    # 1. Dataset
    print("[1/4] Loading Protocol v1 Dataset...")
    dataset = ProtocolV1Dataset(
        base_dir=dataset_cfg.get("base_dir", f"data/protocol_v1/{dataset_name}"),
        split_subdir=dataset_cfg.get("split_subdir", "CFRLdata_10_100_10_5"),
        task_order_strategy=dataset_cfg.get("task_order_strategy", "conpl_seed_shuffle"),
        seed=seed,
        train_shot=int(dataset_cfg.get("train_shot", 5)),
        test_shot=int(dataset_cfg.get("test_shot", 100)),
        max_seq_length=int(model_cfg.get("max_seq_length", 128)),
    )
    tasks = dataset.build_tasks()

    # 2. Model & Router
    print(f"[2/4] Initializing P5PromptedBERT & P5TaskRouter...")
    model = P5PromptedBERT(
        backbone_name=model_cfg.get("backbone", "bert-base-uncased"),
        prompt_length=int(model_cfg.get("prompt_length", 8)),
        root_prompt_length=int(model_cfg.get("root_prompt_length", 4)),
        num_tasks=num_tasks,
        num_classes=total_relations,
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.1)),
        device=device,
        proto_contrastive_temp=float(model_cfg.get("proto_contrastive_temp", 0.1)),
    )
    router = P5TaskRouter(
        input_dim=model.rep_dim,
        num_tasks=num_tasks,
        relations_per_task=relations_per_task,
        subcentroid_weight=float(router_cfg.get("subcentroid_weight", 0.4)),
        enable_whitening=mode in ("p5_a", "p5_b", "p5_c", "p5_full") and bool(router_cfg.get("enable_whitening", True)),
        whitening_dim=int(router_cfg.get("whitening_dim", 32)),
        confidence_threshold=float(router_cfg.get("confidence_threshold", 0.05)),
    ).to(device)

    # Performance Matrices
    p5_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="p5_accuracy", dataset=dataset_name, seed=seed)
    hard_top1_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="hard_top1", dataset=dataset_name, seed=seed)
    soft_fusion_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="soft_fusion", dataset=dataset_name, seed=seed)
    router_top1_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="router_top1", dataset=dataset_name, seed=seed)
    router_top2_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="router_top2", dataset=dataset_name, seed=seed)
    f1_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="macro_f1", dataset=dataset_name, seed=seed)

    ckpt_mgr = CheckpointManager(results_dir / "checkpoints")
    metrics_logger = StreamingMetricsLogger(results_dir)

    # 3. Continual Loop
    print(f"\n[3/4] Continual Learning across {num_tasks} tasks (Mode: {mode})...")
    t_start = time.time()

    for stage in range(num_tasks):
        t_stage = tasks[stage]
        seen_tasks = stage + 1
        seen_classes = sum(len(tasks[k].class_indices) for k in range(stage + 1))
        model.set_seen_classes(seen_classes)

        print(f"\n>>> Stage {seen_tasks}/{num_tasks}: Learning Task {seen_tasks} (Cluster {t_stage.cluster_id}, Seen Classes: {seen_classes})")
        t_train0 = time.time()

        train_single_p5_task(
            model=model,
            router=router,
            task=t_stage,
            config=config,
            mode=mode,
        )
        print(f"  Training Stage {seen_tasks} done in {time.time() - t_train0:.2f}s.")

        print(f"  Evaluating on tasks 1..{seen_tasks}:")
        for j in range(seen_tasks):
            eval_res = evaluate_p5_task(
                model=model,
                router=router,
                test_samples=tasks[j].test_samples,
                true_task_id=j,
                seen_tasks=seen_tasks,
                config=config,
                mode=mode,
                batch_size=int(config.get("evaluation", {}).get("batch_size", 64)),
                task_class_indices=[t.class_indices for t in tasks],
            )

            p5_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["p5_accuracy"])
            hard_top1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["hard_top1_accuracy"])
            soft_fusion_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["soft_fusion_accuracy"])
            router_top1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["router_top1_accuracy"])
            router_top2_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["router_top2_coverage"])
            f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["p5_macro_f1"])

            metrics_logger.log_evaluation(
                stage=stage,
                test_task=j,
                accuracy=eval_res["p5_accuracy"],
                macro_f1=eval_res["p5_macro_f1"],
                sample_count=eval_res["sample_count"],
            )

            print(
                f"    Eval on T{j+1}: P5 Acc={eval_res['p5_accuracy']*100:5.2f}% | "
                f"Hard Top-1={eval_res['hard_top1_accuracy']*100:5.2f}% | "
                f"Soft Fusion={eval_res['soft_fusion_accuracy']*100:5.2f}% | "
                f"Router T1={eval_res['router_top1_accuracy']*100:5.2f}% | "
                f"T2 Cov={eval_res['router_top2_coverage']*100:5.2f}%"
            )

        # Export intermediate checkpoints & matrices
        p5_matrix.export_csv(results_dir / "performance_matrix.csv")
        p5_matrix.export_json(results_dir / "performance_matrix.json")
        hard_top1_matrix.export_csv(results_dir / "hard_top1_matrix.csv")
        soft_fusion_matrix.export_csv(results_dir / "soft_fusion_matrix.csv")
        router_top1_matrix.export_csv(results_dir / "router_top1_matrix.csv")
        router_top2_matrix.export_csv(results_dir / "router_top2_matrix.csv")
        f1_matrix.export_csv(results_dir / "f1_matrix.csv")

        # Save checkpoint only at final stage (or if explicitly requested) to save disk & zip size
        if stage == num_tasks - 1 or config.get("save_all_checkpoints", False):
            ckpt_mgr.save_checkpoint(
                stage=stage,
                model=model,
                metadata={"stage": stage, "seen_classes": seen_classes, "mode": mode},
                router=router,
            )

    # 4. Final Summary
    total_duration = time.time() - t_start
    print(f"\n[4/4] Continual Learning Finished in {total_duration:.2f}s!")

    summary = compute_continual_summary_metrics(p5_matrix)
    hard_top1_summary = compute_continual_summary_metrics(hard_top1_matrix)
    soft_fusion_summary = compute_continual_summary_metrics(soft_fusion_matrix)
    router_top1_summary = compute_continual_summary_metrics(router_top1_matrix)
    router_top2_summary = compute_continual_summary_metrics(router_top2_matrix)

    summary["mode"] = mode
    summary["experiment_id"] = f"{exp_cfg.get('id', 'P5')}_{mode.upper()}"
    summary["final_p5_accuracy"] = summary["final_average_accuracy"]
    summary["final_hard_top1_accuracy"] = hard_top1_summary["final_average_accuracy"]
    summary["final_soft_fusion_accuracy"] = soft_fusion_summary["final_average_accuracy"]
    summary["final_router_top1_accuracy"] = router_top1_summary["final_average_accuracy"]
    summary["final_router_top2_coverage"] = router_top2_summary["final_average_accuracy"]
    summary["router_top1_aia"] = router_top1_summary["average_incremental_accuracy"]
    summary["router_top2_aia"] = router_top2_summary["average_incremental_accuracy"]
    summary["total_duration_seconds"] = round(total_duration, 2)
    summary["gain_over_p4_pct"] = round((summary["final_average_accuracy"] - float(continual_cfg.get("reference_p4_acc", 0.4245))) * 100.0, 2)

    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"P5 ({mode.upper()}) FINAL RESULTS SUMMARY ({dataset_name.upper()})")
    print("=" * 80)
    print(f"Final Average Accuracy (A_last):        {summary['final_p5_accuracy'] * 100:6.2f}%")
    print(f"Average Incremental Accuracy (AIA):      {summary['average_incremental_accuracy'] * 100:6.2f}%")
    print(f"Average Forgetting (F):                  {summary['average_forgetting'] * 100:6.2f}%")
    print(f"Router Top-1 Accuracy:                   {summary['final_router_top1_accuracy'] * 100:6.2f}%")
    print(f"Router Top-2 Coverage:                   {summary['final_router_top2_coverage'] * 100:6.2f}%")
    print(f"--> Gain over P4 Baseline:               {summary['gain_over_p4_pct']:+6.2f}%")
    print("=" * 80 + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TAPTA P5 Refined Method and Ablation Studies")
    parser.add_argument("--config", type=str, required=True, help="Path to P5 config YAML")
    parser.add_argument(
        "--mode",
        type=str,
        default="p5_full",
        choices=["p4_baseline", "p5_a", "p5_b", "p5_c", "p5_full"],
        help="Ablation mode to execute",
    )
    parser.add_argument("--device", type=str, default=None, help="Compute device override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        mode=args.mode,
        device_override=args.device,
        seed_override=args.seed,
    )


if __name__ == "__main__":
    main()
