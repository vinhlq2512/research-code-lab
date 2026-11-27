"""Unified Continual Learning, Routing, Confusion, Tree, and Efficiency Metrics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Reuse canonical PerformanceMatrix & continual metric formulas from dataset-pipelines
_PIPELINE_SRC = Path(__file__).resolve().parents[2] / "dataset-pipelines" / "continual-relation-extraction" / "src"
if str(_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_SRC))

from evaluation.evaluator import compute_accuracy, compute_macro_f1  # noqa: E402
from evaluation.metrics import compute_continual_summary_metrics  # noqa: E402
from evaluation.performance_matrix import PerformanceMatrix  # noqa: E402

from src.dynatree_cre.tree import DynamicSemanticTree  # noqa: E402


def compute_routing_diagnostics(
    true_relation_ids: list[int],
    selected_topk_relation_ids: list[list[int]],
    routing_entropies: list[float],
    tree: DynamicSemanticTree,
    relation_to_task: dict[int, int],
    oracle_acc: float | None = None,
    routed_acc: float | None = None,
) -> dict[str, Any]:
    """Compute relation/cluster routing accuracy, Recall@K, entropy, and cross-task routing confusion."""
    n = len(true_relation_ids)
    if n == 0:
        return {}

    rel_top1_hits = 0
    rel_recall_at_k_hits = 0
    cluster_top1_hits = 0
    cross_task_mismatch = 0

    for true_r, topk_r in zip(true_relation_ids, selected_topk_relation_ids):
        pred_r = int(topk_r[0])
        if pred_r == int(true_r):
            rel_top1_hits += 1
        if int(true_r) in topk_r:
            rel_recall_at_k_hits += 1

        true_cid = tree.nodes[tree.relation_to_node[int(true_r)]].parent_id
        pred_cid = tree.nodes[tree.relation_to_node[pred_r]].parent_id
        if true_cid == pred_cid:
            cluster_top1_hits += 1

        if relation_to_task.get(pred_r) != relation_to_task.get(int(true_r)):
            cross_task_mismatch += 1

    out: dict[str, Any] = {
        "relation_routing_top1_acc": float(rel_top1_hits / n),
        "relation_routing_recall_at_k": float(rel_recall_at_k_hits / n),
        "cluster_routing_top1_acc": float(cluster_top1_hits / n),
        "mean_routing_entropy": float(np.mean(routing_entropies)) if routing_entropies else 0.0,
        "cross_task_routing_confusion_rate": float(cross_task_mismatch / n),
        "num_samples": int(n),
    }
    if oracle_acc is not None and routed_acc is not None:
        out["oracle_routing_gap"] = float(oracle_acc - routed_acc)
    return out


def compute_confusion_diagnostics(
    true_relation_ids: list[int],
    pred_relation_ids: list[int],
    current_task_relation_ids: set[int],
    tree: DynamicSemanticTree,
    relation_to_task: dict[int, int],
) -> dict[str, Any]:
    """Decompose classification errors into old/new, within-cluster, cross-cluster, and cross-task."""
    n = len(true_relation_ids)
    if n == 0:
        return {}

    old_total = 0
    old_correct = 0
    old_to_new = 0
    new_total = 0
    new_correct = 0
    new_to_old = 0
    within_cluster_err = 0
    cross_cluster_err = 0
    cross_task_err = 0
    total_err = 0

    for true_r, pred_r in zip(true_relation_ids, pred_relation_ids):
        is_new = true_r in current_task_relation_ids
        pred_is_new = pred_r in current_task_relation_ids
        if is_new:
            new_total += 1
            if pred_r == true_r:
                new_correct += 1
            elif not pred_is_new:
                new_to_old += 1
        else:
            old_total += 1
            if pred_r == true_r:
                old_correct += 1
            elif pred_is_new:
                old_to_new += 1

        if pred_r != true_r:
            total_err += 1
            true_cid = tree.nodes[tree.relation_to_node[true_r]].parent_id
            pred_cid = tree.nodes[tree.relation_to_node[pred_r]].parent_id
            if true_cid == pred_cid:
                within_cluster_err += 1
            else:
                cross_cluster_err += 1
            if relation_to_task.get(pred_r) != relation_to_task.get(true_r):
                cross_task_err += 1

    return {
        "old_relations_accuracy": float(old_correct / old_total) if old_total > 0 else None,
        "new_relations_accuracy": float(new_correct / new_total) if new_total > 0 else None,
        "old_to_new_error_rate": float(old_to_new / old_total) if old_total > 0 else 0.0,
        "new_to_old_error_rate": float(new_to_old / new_total) if new_total > 0 else 0.0,
        "within_cluster_error_share": float(within_cluster_err / total_err) if total_err > 0 else 0.0,
        "cross_cluster_error_share": float(cross_cluster_err / total_err) if total_err > 0 else 0.0,
        "cross_task_error_share": float(cross_task_err / total_err) if total_err > 0 else 0.0,
        "total_errors": int(total_err),
        "total_samples": int(n),
    }


__all__ = [
    "PerformanceMatrix",
    "compute_accuracy",
    "compute_macro_f1",
    "compute_continual_summary_metrics",
    "compute_routing_diagnostics",
    "compute_confusion_diagnostics",
]
