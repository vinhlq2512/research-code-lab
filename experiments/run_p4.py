#!/usr/bin/env python3
"""Runner for Proposed Method P4: TAPTA Hierarchical Prompt Tree & Decision Fusion.

Protocol: Protocol v1 Track A (FewRel 8 tasks x 10 relations, strict 5-shot, M=0)
Backbone: 100% Frozen BERT-base-uncased with 2-Tier Hierarchical Prompt Tree (L_root=4, L_leaf=4).
Inference: Two-Tier Hierarchical Inference with Decision-Level Soft Prototype Fusion (Zero Oracle Task ID).
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


def train_single_hierarchical_task_and_router(
    model: PromptedBERTRelationClassifier,
    router: TaskRouter,
    task: ProtocolV1Task,
    config: dict[str, Any],
) -> dict[str, float]:
    """Train hierarchical prompt tree (P^root + P^leaf_t), update prototypes, and register stationary queries."""
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    max_seq_len = int(model_cfg.get("max_seq_length", 128))
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs_per_task", 15))
    lr = float(train_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    orth_lambda = float(train_cfg.get("orthogonality_lambda", 0.05))
    freeze_root_after_task0 = bool(model_cfg.get("freeze_root_after_task0", True))

    train_samples = list(task.train_samples)
    n_samples = len(train_samples)

    # 1. Get trainable parameters (Task 0: leaf_0 + root_prompt; Task > 0: leaf_t only)
    trainable_params = model.get_trainable_params(task_id=task.task_id)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, (n_samples // batch_size + (1 if n_samples % batch_size else 0)) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.classifier.train()
    prompt_loss = 0.0
    orth_loss_val = 0.0

    for epoch in range(epochs):
        perm = list(range(n_samples))
        random.shuffle(perm)
        epoch_loss = 0.0
        epoch_orth_loss = 0.0
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

            loss_ce = F.cross_entropy(out["logits"], batch_data["labels"])
            loss_orth = model.compute_orthogonality_loss(task.task_id)
            total_loss = loss_ce + orth_lambda * loss_orth

            total_loss.backward()

            nn.utils.clip_grad_norm_(trainable_params, clip_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss_ce.item())
            epoch_orth_loss += float(loss_orth.item())
            num_batches += 1

        prompt_loss = epoch_loss / max(1, num_batches)
        orth_loss_val = epoch_orth_loss / max(1, num_batches)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            stage_info = (
                "Root + Leaf_0" if task.task_id == 0 else f"Leaf P_{task.task_id + 1}"
            )
            print(
                f"    [{stage_info}] Epoch {epoch + 1:2d}/{epochs} | "
                f"CE Loss: {prompt_loss:.4f} | Orth Loss: {orth_loss_val:.4f}"
            )

    # If Task 0 completed and freeze_root_after_task0 is enabled, freeze root_prompt parameter
    if task.task_id == 0 and freeze_root_after_task0:
        model.root_prompt.requires_grad = False
        print("    [P4 Tree] Root prompt P^root frozen after Task 0 for invariant representation anchoring.")

    # 2. Extract prompted class prototypes using hierarchical prompt [P^root; P^leaf_t]
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

    return {"prompt_loss": prompt_loss, "orth_loss": orth_loss_val}


@torch.no_grad()
def evaluate_hierarchical_task(
    model: PromptedBERTRelationClassifier,
    router: TaskRouter,
    test_samples: list[ProtocolV1Sample],
    true_task_id: int,
    seen_tasks: int,
    config: dict[str, Any],
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate model using Two-Tier Hierarchical Inference & Decision-Level Soft Fusion (Zero Oracle Task ID)."""
    model.eval()
    router.eval()
    max_seq_len = int(config.get("model", {}).get("max_seq_length", 128))
    relations_per_task = int(config.get("dataset", {}).get("relations_per_task", 10))
    top_k = int(config.get("router", {}).get("top_k", 2))
    temperature = float(config.get("router", {}).get("temperature", 0.2))

    all_decision_fusion_sims: list[torch.Tensor] = []
    all_decision_fusion_topk_gated_sims: list[torch.Tensor] = []
    all_tree_top1_sims: list[torch.Tensor] = []
    all_tree_top1_gated_sims: list[torch.Tensor] = []
    all_tree_topk_gated_sims: list[torch.Tensor] = []
    all_linear_logits: list[torch.Tensor] = []

    all_pred_tasks: list[torch.Tensor] = []
    all_topk_tasks: list[torch.Tensor] = []
    all_labels: list[int] = []

    for i in range(0, len(test_samples), batch_size):
        chunk = test_samples[i : i + batch_size]
        batch_data = collate_samples_for_bert(
            chunk,
            tokenizer=model.tokenizer,
            device=model.target_device,
            max_seq_length=max_seq_len,
        )

        # Stage 1: Unprompted stationary query extraction & Router Top-K Candidate selection
        q_r = model.extract_unprompted_query(
            batch_data["input_ids"],
            batch_data["attention_mask"],
            batch_data["e1_indices"],
            batch_data["e2_indices"],
        )
        routing_out = router.get_topk_routing(
            query_features=q_r,
            seen_tasks=seen_tasks,
            top_k=top_k,
            temperature=temperature,
        )
        topk_indices = routing_out["topk_indices"]  # [B, K]
        topk_weights = routing_out["topk_weights"]  # [B, K]
        pred_task_ids = routing_out["pred_task_ids"]  # [B] Top-1 prediction

        k_active = topk_indices.size(1)

        # Stage 2: Feature extraction for candidate 0 (Top-1 predicted task)
        cand0_ids = topk_indices[:, 0]  # [B]
        out0 = model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            e1_indices=batch_data["e1_indices"],
            e2_indices=batch_data["e2_indices"],
            task_id=cand0_ids,
        )
        h_r_0 = out0["representations"]  # [B, 1536]
        logits_0 = out0["logits"]  # [B, 80]
        sim_0 = model.predict_prototype(h_r_0, seen_classes_only=True)  # [B, 80]

        # Stage 3: Decision-Level Soft Fusion across Top-K candidates
        if k_active <= 1:
            fused_sim = sim_0
        else:
            cand_sims = [sim_0]
            for k_idx in range(1, k_active):
                cand_k_ids = topk_indices[:, k_idx]  # [B]
                h_r_k = model.extract_features(
                    batch_data["input_ids"],
                    batch_data["attention_mask"],
                    batch_data["e1_indices"],
                    batch_data["e2_indices"],
                    task_id=cand_k_ids,
                )
                sim_k = model.predict_prototype(h_r_k, seen_classes_only=True)
                cand_sims.append(sim_k)
            fused_sim = model.predict_decision_fusion_prototype(cand_sims, topk_weights)

        # Top-K Gated Decision Fusion
        topk_gated_fused_sim = model.gate_similarity_to_topk_tasks(
            fused_sim, topk_indices, relations_per_task=relations_per_task
        )

        # Tree Top-1 Prototype variants (based on pure Top-1 candidate representation h_r_0)
        tree_top1_sim = sim_0
        tree_top1_gated_sim = model.predict_routed_prototype(
            h_r_0, pred_task_ids, relations_per_task=relations_per_task
        )
        tree_topk_gated_sim = model.predict_topk_gated_prototype(
            h_r_0, topk_indices, relations_per_task=relations_per_task
        )

        all_decision_fusion_sims.append(fused_sim)
        all_decision_fusion_topk_gated_sims.append(topk_gated_fused_sim)
        all_tree_top1_sims.append(tree_top1_sim)
        all_tree_top1_gated_sims.append(tree_top1_gated_sim)
        all_tree_topk_gated_sims.append(tree_topk_gated_sim)
        all_linear_logits.append(logits_0)

        all_pred_tasks.append(pred_task_ids)
        all_topk_tasks.append(topk_indices)
        all_labels.extend(batch_data["labels"].cpu().tolist())

    fused_sim_tensor = torch.cat(all_decision_fusion_sims, dim=0)
    topk_gated_fused_sim_tensor = torch.cat(all_decision_fusion_topk_gated_sims, dim=0)
    tree_top1_sim_tensor = torch.cat(all_tree_top1_sims, dim=0)
    tree_top1_gated_sim_tensor = torch.cat(all_tree_top1_gated_sims, dim=0)
    tree_topk_gated_sim_tensor = torch.cat(all_tree_topk_gated_sims, dim=0)
    linear_logits_tensor = torch.cat(all_linear_logits, dim=0)

    pred_tasks_tensor = torch.cat(all_pred_tasks, dim=0)
    topk_tasks_tensor = torch.cat(all_topk_tasks, dim=0)
    pred_tasks_list = pred_tasks_tensor.cpu().tolist()

    # 1. Top-1 Router Accuracy & Confusion Matrix Counts
    true_tasks_list = [true_task_id] * len(all_labels)
    router_top1_acc = compute_accuracy(true_tasks_list, pred_tasks_list)
    confusion_counts = [0] * seen_tasks
    for pt in pred_tasks_list:
        if 0 <= int(pt) < seen_tasks:
            confusion_counts[int(pt)] += 1

    # 2. Top-K Router Coverage (True task in Top-K candidate list)
    true_task_tensor = torch.full((len(all_labels), 1), true_task_id, device=topk_tasks_tensor.device)
    in_topk = (topk_tasks_tensor == true_task_tensor).any(dim=-1).float()
    router_topk_coverage = float(in_topk.mean().item())

    # 3. Decision-Level Soft Fusion Accuracy (Primary P4 Metric)
    decision_fusion_preds = torch.argmax(fused_sim_tensor, dim=-1).cpu().tolist()
    decision_fusion_acc = compute_accuracy(all_labels, decision_fusion_preds)
    decision_fusion_f1 = compute_macro_f1(all_labels, decision_fusion_preds)

    # 4. Top-K Gated Decision Fusion Accuracy
    decision_fusion_topk_gated_preds = torch.argmax(topk_gated_fused_sim_tensor, dim=-1).cpu().tolist()
    decision_fusion_topk_gated_acc = compute_accuracy(all_labels, decision_fusion_topk_gated_preds)

    # 5. Tree Top-1 Prototype Accuracy (Global prototype matching with Tree Top-1 prompt)
    tree_top1_preds = torch.argmax(tree_top1_sim_tensor, dim=-1).cpu().tolist()
    tree_top1_acc = compute_accuracy(all_labels, tree_top1_preds)

    # 6. Tree Top-1 Gated Prototype Accuracy
    tree_top1_gated_preds = torch.argmax(tree_top1_gated_sim_tensor, dim=-1).cpu().tolist()
    tree_top1_gated_acc = compute_accuracy(all_labels, tree_top1_gated_preds)

    # 7. Tree Top-K Gated Prototype Accuracy
    tree_topk_gated_preds = torch.argmax(tree_topk_gated_sim_tensor, dim=-1).cpu().tolist()
    tree_topk_gated_acc = compute_accuracy(all_labels, tree_topk_gated_preds)

    # 8. Linear Head CIL Accuracy
    linear_preds = torch.argmax(linear_logits_tensor, dim=-1).cpu().tolist()
    linear_acc = compute_accuracy(all_labels, linear_preds)

    return {
        "router_top1_accuracy": router_top1_acc,
        "router_topk_coverage": router_topk_coverage,
        "decision_fusion_accuracy": decision_fusion_acc,
        "decision_fusion_topk_gated_accuracy": decision_fusion_topk_gated_acc,
        "tree_top1_prototype_accuracy": tree_top1_acc,
        "tree_top1_gated_prototype_accuracy": tree_top1_gated_acc,
        "tree_topk_gated_prototype_accuracy": tree_topk_gated_acc,
        "macro_f1": decision_fusion_f1,
        "linear_cil_accuracy": linear_acc,
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


def run_experiment(
    config_path: str | Path,
    device_override: str | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    exp_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    continual_cfg = config.get("continual", {})
    router_cfg = config.get("router", {})
    exp_id = exp_cfg.get("id", "P4")

    seed = seed_override if seed_override is not None else int(config.get("seed", 2021))
    set_seed(seed)

    output_cfg = config.get("output", {})
    raw_results_dir = output_cfg.get("results_dir", "runs/p4_hierarchical_tree/fewrel_5shot/seed_2021")
    if seed_override is not None:
        import re
        raw_results_dir = re.sub(r"seed_\d+", f"seed_{seed}", raw_results_dir)
    results_dir = REPO_ROOT / raw_results_dir
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

    print("==================================================================")
    print(f"Continual Relation Extraction: {exp_id} ({exp_cfg.get('name')})")
    print(f"Description: {exp_cfg.get('description')}")
    print(
        f"Prompt Tree: L_root={model_cfg.get('root_prompt_length', 4)} | "
        f"L_leaf={model_cfg.get('leaf_prompt_length', 4)} | "
        f"L_total={model_cfg.get('prompt_length', 8)}"
    )
    print(f"Decision Fusion: K={router_cfg.get('top_k', 2)} | Temp={router_cfg.get('temperature', 0.2)}")
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

    # 2. Instantiate Prompted BERT (hierarchical_tree mode) & TaskRouter
    target_device = device_override or config.get("training", {}).get("device", "auto")
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
        hidden_dim=int(router_cfg.get("hidden_dim", 512)),
        temperature=float(router_cfg.get("temperature", 16.0)),
        residual_anchor_weight=float(router_cfg.get("residual_anchor_weight", 0.5)),
    ).to(model.target_device)

    print(f"\n[2/4] Initialized Model & TaskRouter on device={model.target_device}")

    # 3. Setup Performance Matrices
    # Primary: Decision Fusion Accuracy
    fusion_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="decision_fusion_accuracy", dataset="fewrel", seed=seed
    )
    fusion_topk_gated_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="decision_fusion_topk_gated_accuracy", dataset="fewrel", seed=seed
    )
    tree_top1_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="tree_top1_prototype_accuracy", dataset="fewrel", seed=seed
    )
    tree_top1_gated_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="tree_top1_gated_prototype_accuracy", dataset="fewrel", seed=seed
    )
    tree_topk_gated_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="tree_topk_gated_prototype_accuracy", dataset="fewrel", seed=seed
    )
    router_top1_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="router_top1_accuracy", dataset="fewrel", seed=seed
    )
    router_topk_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="router_topk_coverage", dataset="fewrel", seed=seed
    )
    f1_matrix = PerformanceMatrix(num_tasks=num_tasks, metric="macro_f1", dataset="fewrel", seed=seed)
    linear_matrix = PerformanceMatrix(
        num_tasks=num_tasks, metric="linear_cil_accuracy", dataset="fewrel", seed=seed
    )

    checkpoint_manager = CheckpointManager(results_dir / "checkpoints")
    metrics_logger = StreamingMetricsLogger(results_dir)

    final_confusion_matrix: list[list[float]] = [[0.0] * num_tasks for _ in range(num_tasks)]

    # 4. Continual Learning Loop
    print("\n[3/4] Starting Continual Learning with Hierarchical Prompt Tree across 8 tasks...")
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

        train_single_hierarchical_task_and_router(
            model=model,
            router=router,
            task=task,
            config=config,
        )

        print(f"  Evaluating on tasks 1..{seen_tasks} (Hierarchical Tree & Decision Fusion):")
        for j in range(seen_tasks):
            eval_res = evaluate_hierarchical_task(
                model=model,
                router=router,
                test_samples=tasks[j].test_samples,
                true_task_id=j,
                seen_tasks=seen_tasks,
                config=config,
                batch_size=int(config.get("evaluation", {}).get("batch_size", 64)),
            )

            fusion_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["decision_fusion_accuracy"])
            fusion_topk_gated_matrix.set_score(
                trained_until=stage, evaluated_task=j, score=eval_res["decision_fusion_topk_gated_accuracy"]
            )
            tree_top1_matrix.set_score(
                trained_until=stage, evaluated_task=j, score=eval_res["tree_top1_prototype_accuracy"]
            )
            tree_top1_gated_matrix.set_score(
                trained_until=stage, evaluated_task=j, score=eval_res["tree_top1_gated_prototype_accuracy"]
            )
            tree_topk_gated_matrix.set_score(
                trained_until=stage, evaluated_task=j, score=eval_res["tree_topk_gated_prototype_accuracy"]
            )
            router_top1_matrix.set_score(
                trained_until=stage, evaluated_task=j, score=eval_res["router_top1_accuracy"]
            )
            router_topk_matrix.set_score(
                trained_until=stage, evaluated_task=j, score=eval_res["router_topk_coverage"]
            )
            f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["macro_f1"])
            linear_matrix.set_score(trained_until=stage, evaluated_task=j, score=eval_res["linear_cil_accuracy"])

            if stage == num_tasks - 1:
                total_j = max(1, eval_res["sample_count"])
                for k_idx, cnt in enumerate(eval_res["confusion_counts"]):
                    final_confusion_matrix[j][k_idx] = cnt / total_j

            metrics_logger.log_evaluation(
                stage=stage,
                test_task=j,
                accuracy=eval_res["decision_fusion_accuracy"],
                macro_f1=eval_res["macro_f1"],
                sample_count=eval_res["sample_count"],
            )

            print(
                f"    Eval on T{j + 1}: Router Top-1={eval_res['router_top1_accuracy'] * 100:5.2f}% | "
                f"Top-K Cov={eval_res['router_topk_coverage'] * 100:5.2f}% | "
                f"Tree Top-1={eval_res['tree_top1_prototype_accuracy'] * 100:5.2f}% | "
                f"Fusion={eval_res['decision_fusion_accuracy'] * 100:5.2f}% | "
                f"Fusion Gated={eval_res['decision_fusion_topk_gated_accuracy'] * 100:5.2f}%"
            )

        # Export intermediate matrices
        fusion_matrix.export_csv(results_dir / "performance_matrix.csv")
        fusion_matrix.export_json(results_dir / "performance_matrix.json")
        fusion_matrix.export_csv(results_dir / "decision_fusion_performance_matrix.csv")
        fusion_topk_gated_matrix.export_csv(results_dir / "decision_fusion_topk_gated_matrix.csv")
        tree_top1_matrix.export_csv(results_dir / "tree_top1_prototype_matrix.csv")
        tree_top1_gated_matrix.export_csv(results_dir / "tree_top1_gated_prototype_matrix.csv")
        tree_topk_gated_matrix.export_csv(results_dir / "tree_topk_gated_prototype_matrix.csv")
        router_top1_matrix.export_csv(results_dir / "router_top1_accuracy_matrix.csv")
        router_topk_matrix.export_csv(results_dir / "router_topk_coverage_matrix.csv")
        f1_matrix.export_csv(results_dir / "f1_performance_matrix.csv")

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
        print(f"  Stage {seen_tasks} completed in {time.time() - stage_t0:.2f}s.")

    total_duration = time.time() - start_time
    print(f"\n[4/4] Continual Learning & Hierarchical Tree Finished in {total_duration:.2f}s!")

    # 5. Compute Summary & Gap Closed Metrics
    fusion_summary = compute_continual_summary_metrics(fusion_matrix)
    fusion_topk_gated_summary = compute_continual_summary_metrics(fusion_topk_gated_matrix)
    tree_top1_summary = compute_continual_summary_metrics(tree_top1_matrix)
    tree_top1_gated_summary = compute_continual_summary_metrics(tree_top1_gated_matrix)
    tree_topk_gated_summary = compute_continual_summary_metrics(tree_topk_gated_matrix)
    router_top1_summary = compute_continual_summary_metrics(router_top1_matrix)
    router_topk_summary = compute_continual_summary_metrics(router_topk_matrix)
    linear_summary = compute_continual_summary_metrics(linear_matrix)

    ref_p0 = float(continual_cfg.get("reference_p0_proto_acc", 0.52825))
    ref_p1 = float(continual_cfg.get("reference_p1_proto_acc", 0.701375))
    ref_p2a = float(continual_cfg.get("reference_p2a_proto_acc", 0.59275))
    oracle_span = max(1e-6, ref_p1 - ref_p0)

    final_fusion_acc = fusion_summary["final_average_accuracy"]
    final_fusion_topk_gated_acc = fusion_topk_gated_summary["final_average_accuracy"]
    final_tree_top1_acc = tree_top1_summary["final_average_accuracy"]
    final_tree_top1_gated_acc = tree_top1_gated_summary["final_average_accuracy"]

    best_p4_acc = max(final_fusion_acc, final_fusion_topk_gated_acc, final_tree_top1_acc, final_tree_top1_gated_acc)
    gap_closed_pct = ((best_p4_acc - ref_p0) / oracle_span) * 100.0

    # Export Final Router Confusion Matrix
    export_confusion_matrix_csv(final_confusion_matrix, results_dir / "router_confusion_matrix.csv")
    with (results_dir / "router_confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": exp_id,
                "top_k": router_cfg.get("top_k", 2),
                "temperature": router_cfg.get("temperature", 0.2),
                "final_router_top1_accuracy": router_top1_summary["final_average_accuracy"],
                "final_router_topk_coverage": router_topk_summary["final_average_accuracy"],
                "confusion_matrix": final_confusion_matrix,
            },
            f,
            indent=2,
        )

    summary = dict(fusion_summary)
    summary["final_decision_fusion_accuracy"] = final_fusion_acc
    summary["final_decision_fusion_topk_gated_accuracy"] = final_fusion_topk_gated_acc
    summary["final_tree_top1_prototype_accuracy"] = final_tree_top1_acc
    summary["final_tree_top1_gated_prototype_accuracy"] = final_tree_top1_gated_acc
    summary["final_tree_topk_gated_prototype_accuracy"] = tree_topk_gated_summary["final_average_accuracy"]
    summary["final_router_top1_accuracy"] = router_top1_summary["final_average_accuracy"]
    summary["final_router_topk_coverage"] = router_topk_summary["final_average_accuracy"]
    summary["router_top1_aia"] = router_top1_summary["average_incremental_accuracy"]
    summary["router_topk_aia"] = router_topk_summary["average_incremental_accuracy"]
    summary["oracle_gap_closed_pct"] = round(gap_closed_pct, 2)
    summary["gain_over_p2a_pct"] = round((best_p4_acc - ref_p2a) * 100.0, 2)
    summary["experiment"] = exp_id
    summary["method"] = exp_cfg.get("name")
    summary["root_prompt_length"] = root_prompt_length
    summary["leaf_prompt_length"] = model.leaf_prompt_length
    summary["top_k"] = router_cfg.get("top_k", 2)
    summary["temperature"] = router_cfg.get("temperature", 0.2)
    summary["seed"] = seed
    summary["total_duration_seconds"] = round(total_duration, 2)

    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Generate Conclusion Report & Plots
    git_info = get_git_info()
    metrics_payload = {
        "summary": summary,
        "performance_matrix": fusion_matrix.to_dict(),
        "fusion_topk_gated_matrix": fusion_topk_gated_matrix.to_dict(),
        "tree_top1_matrix": tree_top1_matrix.to_dict(),
        "tree_top1_gated_matrix": tree_top1_gated_matrix.to_dict(),
        "tree_topk_gated_matrix": tree_topk_gated_matrix.to_dict(),
        "router_top1_matrix": router_top1_matrix.to_dict(),
        "router_topk_matrix": router_topk_matrix.to_dict(),
        "linear_matrix": linear_matrix.to_dict(),
        "confusion_matrix": final_confusion_matrix,
        "git_commit": git_info.get("commit"),
        "git_branch": git_info.get("branch"),
        "has_dirty_working_tree": git_info.get("has_dirty_working_tree"),
        "config": config,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    generate_conclusion_report(
        output_dir=results_dir,
        config=config,
        summary=summary,
        performance_matrix=fusion_matrix,
    )

    try:
        generate_svg_and_html(
            results_dir=results_dir,
            matrix=fusion_matrix,
            summary=summary,
            config=config,
            baseline_name=f"{exp_id}: Hierarchical Tree & Decision Fusion",
        )
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    print("\n==================================================================")
    print(f"P4 Experiment Completed Successfully!")
    print(f"Primary Metric (Decision Fusion Accuracy):          {final_fusion_acc * 100:.2f}%")
    print(f"Decision Fusion Top-K Gated Accuracy:              {final_fusion_topk_gated_acc * 100:.2f}%")
    print(f"Tree Top-1 Prototype Accuracy:                     {final_tree_top1_acc * 100:.2f}%")
    print(f"Tree Top-1 Gated Prototype Accuracy:               {final_tree_top1_gated_acc * 100:.2f}%")
    print(f"Top-1 Router Accuracy:                             {router_top1_summary['final_average_accuracy'] * 100:.2f}%")
    print(f"Top-K Router Coverage:                             {router_topk_summary['final_average_accuracy'] * 100:.2f}%")
    print(f"Average Incremental Accuracy (AIA):                {fusion_summary['average_incremental_accuracy'] * 100:.2f}%")
    print(f"Whole Forgetting:                                  {fusion_summary['whole_forgetting'] * 100:.2f}%")
    print(f"Oracle Gap Closed Percentage:                      {gap_closed_pct:.2f}%")
    print(f"Results saved to: {results_dir}")
    print("==================================================================")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P4 Hierarchical Prompt Tree & Decision Fusion Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/p4_hierarchical_prompt_tree_fewrel_5shot_seed2021.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed and output directory (e.g. --seed 2022)",
    )
    args = parser.parse_args()

    run_experiment(config_path=args.config, device_override=args.device, seed_override=args.seed)


if __name__ == "__main__":
    main()
