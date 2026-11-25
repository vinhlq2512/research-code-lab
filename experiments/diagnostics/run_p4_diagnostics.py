#!/usr/bin/env python3
"""Diagnostic Experiment Runner for TAPTA P4 (D01 - D09).

Deconstructs P4 performance into:
- D01: Oracle Task + Task-Local Classifier (within-task discrimination ceiling)
- D02: Oracle Task + Global Classifier (cross-task interference penalty)
- D03: Oracle Top-K Routing (router miss penalty)
- D04: Hard Top-1 Routing (K=1)
- D05: Reference Top-2 Soft Fusion (K=2, P4 standard)
- D06: Top-4 Soft Fusion (K=4)
- D07: All-Task Fusion (K=seen_tasks)
- D08: Score Calibration (Z-score normalized decision fusion)
- D09: Frozen BERT + Flat Nearest Prototype (Zero-prompt baseline)

Exports sample-level predictions to `predictions.jsonl` for error attribution.
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
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Search paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
BASELINES_SRC = REPO_ROOT / "experiments" / "continual-relation-baselines" / "src"
PIPELINE_SRC = REPO_ROOT / "dataset-pipelines" / "continual-relation-extraction" / "src"

for p in [REPO_ROOT, SRC_DIR, BASELINES_SRC, PIPELINE_SRC, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation.evaluator import compute_accuracy, compute_macro_f1
from cl_re_baselines.checkpoint import CheckpointManager
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


def train_single_task(
    model: PromptedBERTRelationClassifier,
    router: TaskRouter,
    task: ProtocolV1Task,
    config: dict[str, Any],
) -> dict[str, float]:
    """Train prompt tree, extract prototypes, and register router statistics."""
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
    freeze_root = bool(model_cfg.get("freeze_root_after_task0", True))

    train_samples = list(task.train_samples)
    n_samples = len(train_samples)

    trainable_params = model.get_trainable_params(task_id=task.task_id)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, (n_samples // batch_size + (1 if n_samples % batch_size else 0)) * epochs)
    from transformers import get_linear_schedule_with_warmup

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
            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, clip_norm)
            optimizer.step()
            scheduler.step()

    if task.task_id == 0 and freeze_root:
        model.root_prompt.requires_grad = False

    # Extract prompted prototypes
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

    # Register router queries in unprompted stationary space
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
    return {}


@torch.no_grad()
def evaluate_diagnostics(
    model: PromptedBERTRelationClassifier,
    router: TaskRouter,
    all_test_samples: list[ProtocolV1Sample],
    num_tasks: int,
    relations_per_task: int,
    seen_classes: int,
    config: dict[str, Any],
    unprompted_prototypes: torch.Tensor | None = None,
    batch_size: int = 64,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Execute D01 - D09 comprehensive diagnostics across all evaluated samples."""
    model.eval()
    router.eval()
    max_seq_len = int(config.get("model", {}).get("max_seq_length", 128))
    temperature = float(config.get("router", {}).get("temperature", 0.2))

    sample_logs: list[dict[str, Any]] = []

    # Accumulators for accuracy
    mode_correct: dict[str, int] = {
        "D01_oracle_local": 0,
        "D02_oracle_global": 0,
        "D03_oracle_topk": 0,
        "D04_hard_top1": 0,
        "D05_ref_top2": 0,
        "D06_top4": 0,
        "D07_all_fusion": 0,
        "D08_calibrated": 0,
        "D09_unprompted_proto": 0,
    }
    router_top1_correct = 0
    router_top2_coverage = 0
    router_top4_coverage = 0
    total_samples = len(all_test_samples)

    for i in range(0, total_samples, batch_size):
        chunk = all_test_samples[i : i + batch_size]
        batch_data = collate_samples_for_bert(
            chunk,
            tokenizer=model.tokenizer,
            device=model.target_device,
            max_seq_length=max_seq_len,
        )
        b_size = len(chunk)

        # 1. Stationary unprompted queries
        q_r = model.extract_unprompted_query(
            batch_data["input_ids"],
            batch_data["attention_mask"],
            batch_data["e1_indices"],
            batch_data["e2_indices"],
        )

        # Router scores across all seen tasks: [B, num_tasks]
        router_scores = router.compute_centroid_scores(q_r, seen_tasks=num_tasks)
        router_top1 = torch.argmax(router_scores, dim=-1)  # [B]
        _, top2_indices = torch.topk(router_scores, k=min(2, num_tasks), dim=-1)  # [B, 2]
        _, top4_indices = torch.topk(router_scores, k=min(4, num_tasks), dim=-1)  # [B, 4]
        all_task_indices = torch.arange(num_tasks, device=q_r.device).expand(b_size, -1)

        # Softmax weights for fusion
        weights_top2 = F.softmax(torch.gather(router_scores, 1, top2_indices) / temperature, dim=-1)
        weights_top4 = F.softmax(torch.gather(router_scores, 1, top4_indices) / temperature, dim=-1)
        weights_all = F.softmax(router_scores / temperature, dim=-1)

        # 2. Extract prompted features per task for each sample in batch
        # To avoid re-extracting, compute representations for needed tasks
        # Precompute candidate representations for all seen tasks [num_tasks, B, D]
        task_sims: list[torch.Tensor] = []
        for t_idx in range(num_tasks):
            t_tensor = torch.full((b_size,), t_idx, dtype=torch.long, device=q_r.device)
            h_t = model.extract_features(
                batch_data["input_ids"],
                batch_data["attention_mask"],
                batch_data["e1_indices"],
                batch_data["e2_indices"],
                task_id=t_tensor,
            )
            sim_t = model.predict_prototype(h_t, seen_classes_only=True)  # [B, seen_classes]
            task_sims.append(sim_t)
        # Stacked sims: [B, num_tasks, seen_classes]
        stacked_sims = torch.stack(task_sims, dim=1)

        # 3. Unprompted Prototype Baseline (D09)
        if unprompted_prototypes is not None:
            norm_q = F.normalize(q_r, p=2, dim=-1)
            norm_unp_proto = F.normalize(unprompted_prototypes[:seen_classes], p=2, dim=-1)
            unp_sim = torch.mm(norm_q, norm_unp_proto.t())  # [B, seen_classes]
            d09_preds = torch.argmax(unp_sim, dim=-1)
        else:
            d09_preds = torch.zeros(b_size, dtype=torch.long, device=q_r.device)

        # Evaluate each sample in chunk
        for b in range(b_size):
            s = chunk[b]
            y_true = s.class_idx
            t_true = s.task_id
            t_pred1 = int(router_top1[b].item())
            top2_t = top2_indices[b].tolist()
            top4_t = top4_indices[b].tolist()

            r_top1_ok = (t_pred1 == t_true)
            r_top2_ok = (t_true in top2_t)
            r_top4_ok = (t_true in top4_t)

            if r_top1_ok:
                router_top1_correct += 1
            if r_top2_ok:
                router_top2_coverage += 1
            if r_top4_ok:
                router_top4_coverage += 1

            # --- D01: Oracle Task + Task-Local Classifier ---
            # True task prompt sim, sliced to local classes
            sim_true_task = stacked_sims[b, t_true]  # [seen_classes]
            c_start = t_true * relations_per_task
            c_end = min(c_start + relations_per_task, seen_classes)
            local_sim = sim_true_task[c_start:c_end]
            d01_pred = c_start + int(torch.argmax(local_sim).item())
            d01_ok = (d01_pred == y_true)
            if d01_ok:
                mode_correct["D01_oracle_local"] += 1

            # --- D02: Oracle Task + Global Classifier ---
            d02_pred = int(torch.argmax(sim_true_task).item())
            d02_ok = (d02_pred == y_true)
            if d02_ok:
                mode_correct["D02_oracle_global"] += 1

            # --- D03: Oracle Top-K Routing (K=2) ---
            # Construct oracle top-2: if t_true not in top-2, replace 2nd candidate with t_true
            if t_true in top2_t:
                oracle_top2_t = top2_t
            else:
                oracle_top2_t = [top2_t[0], t_true]
            # Weights for oracle top-2
            sc_o = torch.tensor([router_scores[b, t].item() for t in oracle_top2_t], device=q_r.device)
            w_o = F.softmax(sc_o / temperature, dim=-1)
            sim_d03 = w_o[0] * stacked_sims[b, oracle_top2_t[0]] + w_o[1] * stacked_sims[b, oracle_top2_t[1]]
            d03_pred = int(torch.argmax(sim_d03).item())
            d03_ok = (d03_pred == y_true)
            if d03_ok:
                mode_correct["D03_oracle_topk"] += 1

            # --- D04: Hard Top-1 Routing ---
            sim_d04 = stacked_sims[b, t_pred1]
            d04_pred = int(torch.argmax(sim_d04).item())
            d04_ok = (d04_pred == y_true)
            if d04_ok:
                mode_correct["D04_hard_top1"] += 1

            # --- D05: Reference Top-2 Soft Fusion (P4) ---
            w2 = weights_top2[b]  # [2]
            sim_d05 = w2[0] * stacked_sims[b, top2_t[0]] + w2[1] * stacked_sims[b, top2_t[1]]
            d05_pred = int(torch.argmax(sim_d05).item())
            d05_ok = (d05_pred == y_true)
            if d05_ok:
                mode_correct["D05_ref_top2"] += 1

            # --- D06: Top-4 Soft Fusion ---
            w4 = weights_top4[b]  # [4]
            sim_d06 = sum(w4[k] * stacked_sims[b, top4_t[k]] for k in range(len(top4_t)))
            d06_pred = int(torch.argmax(sim_d06).item())
            d06_ok = (d06_pred == y_true)
            if d06_ok:
                mode_correct["D06_top4"] += 1

            # --- D07: All-Task Fusion ---
            wall = weights_all[b]  # [num_tasks]
            sim_d07 = torch.sum(wall.unsqueeze(-1) * stacked_sims[b], dim=0)
            d07_pred = int(torch.argmax(sim_d07).item())
            d07_ok = (d07_pred == y_true)
            if d07_ok:
                mode_correct["D07_all_fusion"] += 1

            # --- D08: Score Calibration (Z-Score Normalized Top-2 Fusion) ---
            # Z-score normalize cosine similarities per candidate prompt before weighting
            cand_sims_norm: list[torch.Tensor] = []
            for t_c in top2_t:
                s_raw = stacked_sims[b, t_c]
                s_mean = s_raw.mean()
                s_std = s_raw.std().clamp(min=1e-5)
                cand_sims_norm.append((s_raw - s_mean) / s_std)
            sim_d08 = w2[0] * cand_sims_norm[0] + w2[1] * cand_sims_norm[1]
            d08_pred = int(torch.argmax(sim_d08).item())
            d08_ok = (d08_pred == y_true)
            if d08_ok:
                mode_correct["D08_calibrated"] += 1

            # --- D09: Frozen BERT Unprompted Prototype ---
            d09_p = int(d09_preds[b].item())
            d09_ok = (d09_p == y_true)
            if d09_ok:
                mode_correct["D09_unprompted_proto"] += 1

            # Error Taxonomy Classification
            if d05_ok:
                error_cat = "correct"
            elif not r_top1_ok and not r_top2_ok:
                error_cat = "router_miss"
            elif r_top1_ok and not d01_ok:
                error_cat = "within_task_confusion"
            elif r_top1_ok and d01_ok and not d02_ok:
                error_cat = "cross_task_interference"
            elif r_top2_ok and d03_ok and not d05_ok:
                error_cat = "uncalibrated_fusion"
            else:
                error_cat = "inter_task_confusion"

            sample_logs.append({
                "sample_idx": i + b,
                "true_task": t_true,
                "true_rel": y_true,
                "true_rel_name": s.relation_name,
                "pred_task_top1": t_pred1,
                "top2_tasks": top2_t,
                "top4_tasks": top4_t,
                "router_correct_top1": r_top1_ok,
                "router_in_top2": r_top2_ok,
                "router_in_top4": r_top4_ok,
                "d01_local_pred": d01_pred,
                "d01_correct": d01_ok,
                "d02_global_pred": d02_pred,
                "d02_correct": d02_ok,
                "d03_oracle_topk_pred": d03_pred,
                "d03_correct": d03_ok,
                "d04_top1_pred": d04_pred,
                "d04_correct": d04_ok,
                "d05_p4_fusion_pred": d05_pred,
                "d05_correct": d05_ok,
                "d06_top4_pred": d06_pred,
                "d06_correct": d06_ok,
                "d07_all_fusion_pred": d07_pred,
                "d07_correct": d07_ok,
                "d08_calibrated_pred": d08_pred,
                "d08_correct": d08_ok,
                "d09_unprompted_pred": d09_p,
                "d09_correct": d09_ok,
                "error_category": error_cat,
            })

    results: dict[str, float] = {
        "D01_oracle_local_acc": mode_correct["D01_oracle_local"] / total_samples,
        "D02_oracle_global_acc": mode_correct["D02_oracle_global"] / total_samples,
        "D03_oracle_topk_acc": mode_correct["D03_oracle_topk"] / total_samples,
        "D04_hard_top1_acc": mode_correct["D04_hard_top1"] / total_samples,
        "D05_ref_top2_acc": mode_correct["D05_ref_top2"] / total_samples,
        "D06_top4_acc": mode_correct["D06_top4"] / total_samples,
        "D07_all_fusion_acc": mode_correct["D07_all_fusion"] / total_samples,
        "D08_calibrated_acc": mode_correct["D08_calibrated"] / total_samples,
        "D09_unprompted_proto_acc": mode_correct["D09_unprompted_proto"] / total_samples,
        "router_top1_accuracy": router_top1_correct / total_samples,
        "router_top2_coverage": router_top2_coverage / total_samples,
        "router_top4_coverage": router_top4_coverage / total_samples,
        "cross_task_interference_penalty": (
            mode_correct["D01_oracle_local"] - mode_correct["D02_oracle_global"]
        )
        / total_samples,
        "router_miss_penalty": (
            mode_correct["D03_oracle_topk"] - mode_correct["D05_ref_top2"]
        )
        / total_samples,
        "calibration_gain": (
            mode_correct["D08_calibrated"] - mode_correct["D05_ref_top2"]
        )
        / total_samples,
        "total_test_samples": total_samples,
    }
    return results, sample_logs


def extract_unprompted_prototypes(
    model: PromptedBERTRelationClassifier,
    tasks: list[ProtocolV1Task],
    num_classes: int,
) -> torch.Tensor:
    """Compute flat nearest prototypes directly in unprompted BERT space for D09."""
    device = model.target_device
    unp_prototypes = torch.zeros(num_classes, model.rep_dim, device=device)
    model.eval()
    with torch.no_grad():
        for task in tasks:
            train_samples = list(task.train_samples)
            for c_idx in task.class_indices:
                c_samples = [s for s in train_samples if s.class_idx == c_idx]
                if c_samples:
                    batch_data = collate_samples_for_bert(
                        c_samples,
                        tokenizer=model.tokenizer,
                        device=device,
                        max_seq_length=128,
                    )
                    q_c = model.extract_unprompted_query(
                        batch_data["input_ids"],
                        batch_data["attention_mask"],
                        batch_data["e1_indices"],
                        batch_data["e2_indices"],
                    )
                    unp_prototypes[c_idx] = q_c.mean(dim=0)
    return unp_prototypes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TAPTA P4 Performance Diagnostics (D01-D09)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, mps, cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to load checkpoint from")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save diagnostic artifacts")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 2021))
    set_seed(seed)

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    router_cfg = config.get("router", {})
    dataset_name = dataset_cfg.get("name", "fewrel")
    num_tasks = int(dataset_cfg.get("num_tasks", 8))
    relations_per_task = int(dataset_cfg.get("relations_per_task", 10 if dataset_name == "fewrel" else 5))
    total_relations = int(dataset_cfg.get("total_relations", 80 if dataset_name == "fewrel" else 40))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "runs" / "diagnostics" / f"{dataset_name}_5shot" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================================================================================")
    print(f"TAPTA P4 PERFORMANCE DIAGNOSTICS: {dataset_name.upper()} (Seed: {seed})")
    print(f"Output Directory: {output_dir}")
    print(f"================================================================================\n")

    # 1. Load Dataset
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
    all_test_samples: list[ProtocolV1Sample] = []
    for t in tasks:
        all_test_samples.extend(t.test_samples)

    # 2. Initialize Model & Router
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"[2/4] Initializing PromptedBERT and TaskRouter on device: {device}...")
    model = PromptedBERTRelationClassifier(
        backbone_name=model_cfg.get("backbone", "bert-base-uncased"),
        prompt_mode="hierarchical_tree",
        prompt_length=int(model_cfg.get("prompt_length", 8)),
        root_prompt_length=int(model_cfg.get("root_prompt_length", 4)),
        num_tasks=num_tasks,
        num_classes=total_relations,
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.1)),
        device=device,
    )
    router = TaskRouter(
        input_dim=model.rep_dim,
        num_tasks=num_tasks,
        relations_per_task=relations_per_task,
        router_type=router_cfg.get("type", "centroid"),
        subcentroid_weight=float(router_cfg.get("subcentroid_weight", 0.7)),
        temperature=float(router_cfg.get("temperature", 16.0)),
    ).to(device)

    # 3. Checkpoint or Train
    checkpoint_loaded = False
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_mgr = CheckpointManager(ckpt_dir)
        after_t8 = ckpt_mgr.get_stage_dir(num_tasks - 1)
        if (after_t8 / "model.pt").exists():
            print(f"Loading checkpoint from {after_t8}...")
            ckpt_mgr.load_checkpoint(num_tasks - 1, model=model, router=router)
            checkpoint_loaded = True

    if not checkpoint_loaded:
        print("[3/4] Checkpoint model.pt not found on disk. Running Continual Training 1..8 tasks...")
        t0 = time.time()
        for stage in range(num_tasks):
            t_stage = tasks[stage]
            seen_classes = (stage + 1) * relations_per_task
            model.set_seen_classes(seen_classes)
            print(f"  Training Stage {stage + 1}/{num_tasks}: Task {t_stage.task_id + 1} ({len(t_stage.train_samples)} samples)...")
            train_single_task(model, router, t_stage, config)
        print(f"Training completed in {time.time() - t0:.2f}s.")
        # Save checkpoints properly
        ckpt_mgr = CheckpointManager(output_dir / "checkpoints")
        ckpt_mgr.save_checkpoint(num_tasks - 1, model=model, metadata={"stage": num_tasks - 1}, router=router)

    # 4. Extract unprompted prototypes for D09
    print("[4/4] Computing Flat Unprompted Prototypes (D09 Baseline)...")
    unp_prototypes = extract_unprompted_prototypes(model, tasks, num_classes=total_relations)

    # 5. Run Full Diagnostics
    print(f"\nExecuting Diagnostics D01 - D09 across {len(all_test_samples)} test samples...")
    t_diag0 = time.time()
    results, sample_logs = evaluate_diagnostics(
        model=model,
        router=router,
        all_test_samples=all_test_samples,
        num_tasks=num_tasks,
        relations_per_task=relations_per_task,
        seen_classes=total_relations,
        config=config,
        unprompted_prototypes=unp_prototypes,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 64)),
    )
    diag_duration = time.time() - t_diag0
    results["diagnostic_duration_seconds"] = round(diag_duration, 2)
    results["dataset"] = dataset_name
    results["seed"] = seed

    # 6. Export Results
    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC RESULTS SUMMARY ({dataset_name.upper()} 5-shot, Seed {seed})")
    print("=" * 80)
    print(f"D01 (Oracle Task + Task-Local):          {results['D01_oracle_local_acc'] * 100:6.2f}% (Within-task ceiling)")
    print(f"D02 (Oracle Task + Global Classifier):   {results['D02_oracle_global_acc'] * 100:6.2f}%")
    print(f"D03 (Oracle Top-K Routing):              {results['D03_oracle_topk_acc'] * 100:6.2f}% (If router never missed)")
    print(f"D04 (Hard Top-1 Routing, K=1):           {results['D04_hard_top1_acc'] * 100:6.2f}%")
    print(f"D05 (Reference Top-2 Soft Fusion, K=2):  {results['D05_ref_top2_acc'] * 100:6.2f}% [P4 Standard]")
    print(f"D06 (Top-4 Soft Fusion, K=4):            {results['D06_top4_acc'] * 100:6.2f}%")
    print(f"D07 (All-Task Fusion, K=8):              {results['D07_all_fusion_acc'] * 100:6.2f}%")
    print(f"D08 (Score Calibration, Z-Score):        {results['D08_calibrated_acc'] * 100:6.2f}%")
    print(f"D09 (Frozen BERT Flat Prototype):        {results['D09_unprompted_proto_acc'] * 100:6.2f}% (Zero-prompt)")
    print("-" * 80)
    print(f"Router Top-1 Accuracy:                   {results['router_top1_accuracy'] * 100:6.2f}%")
    print(f"Router Top-2 Coverage:                   {results['router_top2_coverage'] * 100:6.2f}%")
    print(f"Router Top-4 Coverage:                   {results['router_top4_coverage'] * 100:6.2f}%")
    print(f"--> Cross-Task Interference Penalty:     {results['cross_task_interference_penalty'] * 100:6.2f}% (D01 - D02)")
    print(f"--> Router Miss Penalty:                 {results['router_miss_penalty'] * 100:6.2f}% (D03 - D05)")
    print(f"--> Score Calibration Gain:              {results['calibration_gain'] * 100:6.2f}% (D08 - D05)")
    print("=" * 80 + "\n")

    # Export JSON Summary
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Export CSV Matrix
    csv_rows = [
        ["Diagnostic_ID", "Description", "Accuracy", "Interpretation"],
        ["D01", "Oracle Task + Task-Local Classifier", f"{results['D01_oracle_local_acc']:.4f}", "Within-task discrimination ceiling"],
        ["D02", "Oracle Task + Global Classifier", f"{results['D02_oracle_global_acc']:.4f}", "Performance with oracle task prompt on global relations"],
        ["D03", "Oracle Top-K Routing", f"{results['D03_oracle_topk_acc']:.4f}", "Upper bound if router never missed true task"],
        ["D04", "Hard Top-1 Routing", f"{results['D04_hard_top1_acc']:.4f}", "Single prompt routing (K=1)"],
        ["D05", "Reference Top-2 Soft Fusion", f"{results['D05_ref_top2_acc']:.4f}", "P4 Standard Soft Fusion (K=2)"],
        ["D06", "Top-4 Soft Fusion", f"{results['D06_top4_acc']:.4f}", "Expanded candidate fusion (K=4)"],
        ["D07", "All-Task Fusion", f"{results['D07_all_fusion_acc']:.4f}", "Fusion over all seen tasks"],
        ["D08", "Score Calibration", f"{results['D08_calibrated_acc']:.4f}", "Z-score normalized similarity fusion"],
        ["D09", "Frozen BERT Flat Prototype", f"{results['D09_unprompted_proto_acc']:.4f}", "Unprompted stationary baseline"],
    ]
    with (output_dir / "diagnostic_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    # Export sample-level logs
    predictions_file = output_dir / "predictions.jsonl"
    with predictions_file.open("w", encoding="utf-8") as f:
        for item in sample_logs:
            f.write(json.dumps(item) + "\n")

    print(f"Artifacts successfully written to: {output_dir}")
    print(f"- Summary:     {output_dir / 'diagnostic_summary.json'}")
    print(f"- Matrix:      {output_dir / 'diagnostic_matrix.csv'}")
    print(f"- Predictions: {predictions_file} ({len(sample_logs)} records)")


if __name__ == "__main__":
    main()
