"""M7.0 & M7.1 Checkpoint Diagnostic Utilities (Tensor/Prediction Equality & Routing Decomposition)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data.protocol_v1_loader import collate_samples_for_bert
from src.dynatree_cre.benchmark import TaskStreamBundle
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.tree import DynamicSemanticTree


def _flatten_tensors(obj: Any, prefix: str = "") -> dict[str, torch.Tensor | np.ndarray]:
    out: dict[str, torch.Tensor | np.ndarray] = {}
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj.detach().cpu()
    elif isinstance(obj, np.ndarray):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_tensors(obj[k], sub_prefix))
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            out.update(_flatten_tensors(item, f"{prefix}[{idx}]"))
    return out


def compare_checkpoints_exact(ckpt_a: str | Path, ckpt_b: str | Path) -> dict[str, Any]:
    """Verify exact tensor equality across two saved state.pt checkpoints."""
    pa = torch.load(Path(ckpt_a), map_location="cpu", weights_only=False)
    pb = torch.load(Path(ckpt_b), map_location="cpu", weights_only=False)

    ta = _flatten_tensors(pa.get("model_state", {}))
    tb = _flatten_tensors(pb.get("model_state", {}))

    keys_a = set(ta.keys())
    keys_b = set(tb.keys())
    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    mismatched: list[dict[str, Any]] = []

    for k in sorted(keys_a & keys_b):
        va, vb = ta[k], tb[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape or not torch.equal(va, vb):
                max_diff = float((va.float() - vb.float()).abs().max().item()) if va.shape == vb.shape and va.numel() > 0 else None
                mismatched.append({"key": k, "shape_a": list(va.shape), "shape_b": list(vb.shape), "max_abs_diff": max_diff})
        elif isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
            if va.shape != vb.shape or not np.array_equal(va, vb):
                max_diff = float(np.abs(va.astype(np.float64) - vb.astype(np.float64)).max()) if va.shape == vb.shape and va.size > 0 else None
                mismatched.append({"key": k, "shape_a": list(va.shape), "shape_b": list(vb.shape), "max_abs_diff": max_diff})

    return {
        "ckpt_a": str(ckpt_a),
        "ckpt_b": str(ckpt_b),
        "completed_task_a": int(pa.get("completed_task_id", -1)),
        "completed_task_b": int(pb.get("completed_task_id", -1)),
        "dataset_fingerprint_equal": pa.get("dataset_fingerprint") == pb.get("dataset_fingerprint"),
        "task_order_fingerprint_equal": pa.get("task_order_fingerprint") == pb.get("task_order_fingerprint"),
        "total_tensors_checked": len(keys_a & keys_b),
        "missing_in_b": missing_in_b,
        "missing_in_a": missing_in_a,
        "mismatched_tensors": mismatched,
        "all_tensors_exact_equal": (len(missing_in_a) == 0 and len(missing_in_b) == 0 and len(mismatched) == 0),
    }


def _compute_2x2(records: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    n = len(records)
    rc_pc = sum(1 for r in records if r["routing_top1_correct"] and r[pred_key] == r["true_rel"])
    rc_pw = sum(1 for r in records if r["routing_top1_correct"] and r[pred_key] != r["true_rel"])
    rw_pc = sum(1 for r in records if (not r["routing_top1_correct"]) and r[pred_key] == r["true_rel"])
    rw_pw = sum(1 for r in records if (not r["routing_top1_correct"]) and r[pred_key] != r["true_rel"])
    return {
        "routing_correct_pred_correct": rc_pc,
        "routing_correct_pred_wrong": rc_pw,
        "routing_wrong_pred_correct": rw_pc,
        "routing_wrong_pred_wrong": rw_pw,
        "total": n,
        "routing_correct_pred_correct_rate": float(rc_pc / n) if n else 0.0,
        "routing_correct_pred_wrong_rate": float(rc_pw / n) if n else 0.0,
        "routing_wrong_pred_correct_rate": float(rw_pc / n) if n else 0.0,
        "routing_wrong_pred_wrong_rate": float(rw_pw / n) if n else 0.0,
    }


def compute_m7_routing_decomposition(
    records: list[dict[str, Any]],
    tree: DynamicSemanticTree,
    relation_to_task: dict[int, int],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Compute M7.1 sample-level 2x2 decomposition, Recall@K, per-cluster, and calibration metrics."""
    n = len(records)
    if n == 0:
        return {"num_samples": 0}

    recall_at_k: dict[str, float] = {}
    cond_ncm_acc_at_k: dict[str, float | None] = {}
    cond_cos_acc_at_k: dict[str, float | None] = {}
    for k in ks:
        in_k = [r for r in records if r["true_rel"] in r["ranked_rel_ids"][:k]]
        recall_at_k[f"R@{k}"] = float(len(in_k) / n)
        cond_ncm_acc_at_k[f"R@{k}"] = float(sum(1 for r in in_k if r["pred_ncm"] == r["true_rel"]) / len(in_k)) if in_k else None
        cond_cos_acc_at_k[f"R@{k}"] = float(sum(1 for r in in_k if r["pred_cos"] == r["true_rel"]) / len(in_k)) if in_k else None

    # Per-task breakdown
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_task[int(relation_to_task[r["true_rel"]])].append(r)
        cid = str(tree.nodes[tree.relation_to_node[r["true_rel"]]].parent_id)
        by_cluster[cid].append(r)

    per_task = {}
    for tid in sorted(by_task.keys()):
        recs = by_task[tid]
        m = len(recs)
        per_task[f"T{tid + 1}"] = {
            "num_samples": m,
            "ncm_acc": float(sum(1 for r in recs if r["pred_ncm"] == r["true_rel"]) / m),
            "cos_acc": float(sum(1 for r in recs if r["pred_cos"] == r["true_rel"]) / m),
            "route_top1": float(sum(1 for r in recs if r["routing_top1_correct"]) / m),
            "recall_at_3": float(sum(1 for r in recs if r["true_rel"] in r["ranked_rel_ids"][:3]) / m),
        }

    per_cluster = {}
    singleton_recs: list[dict[str, Any]] = []
    nonsingleton_recs: list[dict[str, Any]] = []
    for cid in sorted(by_cluster.keys()):
        recs = by_cluster[cid]
        c_size = len(tree.nodes[cid].children_ids)
        m = len(recs)
        if c_size == 1:
            singleton_recs.extend(recs)
        else:
            nonsingleton_recs.extend(recs)
        per_cluster[cid] = {
            "cluster_size": int(c_size),
            "num_samples": m,
            "ncm_acc": float(sum(1 for r in recs if r["pred_ncm"] == r["true_rel"]) / m),
            "cos_acc": float(sum(1 for r in recs if r["pred_cos"] == r["true_rel"]) / m),
            "route_top1": float(sum(1 for r in recs if r["routing_top1_correct"]) / m),
            "cluster_top1": float(sum(1 for r in recs if r["cluster_top1_correct"]) / m),
            "recall_at_3": float(sum(1 for r in recs if r["true_rel"] in r["ranked_rel_ids"][:3]) / m),
        }

    def _group_stats(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not recs:
            return None
        m = len(recs)
        return {
            "num_samples": m,
            "ncm_acc": float(sum(1 for r in recs if r["pred_ncm"] == r["true_rel"]) / m),
            "route_top1": float(sum(1 for r in recs if r["routing_top1_correct"]) / m),
        }

    # Confidence calibration bins (router top-1 probability)
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    confidence_bins = []
    for lo, hi in bins:
        sub = [r for r in records if lo <= r["top1_prob"] < hi]
        confidence_bins.append({
            "bin": f"[{lo:.1f}, {min(hi, 1.0):.1f}]",
            "count": len(sub),
            "ncm_acc": float(sum(1 for r in sub if r["pred_ncm"] == r["true_rel"]) / len(sub)) if sub else None,
            "route_top1": float(sum(1 for r in sub if r["routing_top1_correct"]) / len(sub)) if sub else None,
        })

    entropies = [r["entropy"] for r in records]
    return {
        "num_samples": n,
        "overall_ncm_acc": float(sum(1 for r in records if r["pred_ncm"] == r["true_rel"]) / n),
        "overall_cos_acc": float(sum(1 for r in records if r["pred_cos"] == r["true_rel"]) / n),
        "matrix_2x2_ncm": _compute_2x2(records, "pred_ncm"),
        "matrix_2x2_cos": _compute_2x2(records, "pred_cos"),
        "recall_at_k": recall_at_k,
        "conditional_ncm_acc_given_in_topk": cond_ncm_acc_at_k,
        "conditional_cos_acc_given_in_topk": cond_cos_acc_at_k,
        "per_task": per_task,
        "per_cluster": per_cluster,
        "singleton_clusters": _group_stats(singleton_recs),
        "nonsingleton_clusters": _group_stats(nonsingleton_recs),
        "confidence_bins": confidence_bins,
        "entropy_stats": {
            "mean": float(np.mean(entropies)),
            "std": float(np.std(entropies)),
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies)),
        },
    }


@torch.no_grad()
def evaluate_checkpoint_diagnostic(
    model: DynaTreeCREModel,
    stream: TaskStreamBundle,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    use_raw_retrieval: bool = False,
    routing_override: str | None = None,
    eta: float = 0.0,
    gamma: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run non-destructive sample-level diagnostic on a loaded model across seen tasks."""
    model.eval()
    device = model.device
    max_k = max(ks)
    records: list[dict[str, Any]] = []

    ordered_rel_ids = model.tree.seen_relation_ids()
    ordered_leaf_ids = [model.tree.relation_to_node[r] for r in ordered_rel_ids]
    ordered_cluster_ids = model.tree.cluster_ids()
    cluster_idx_map = {cid: idx for idx, cid in enumerate(ordered_cluster_ids)}

    if use_raw_retrieval:
        leaf_mat = np.stack(
            [model.prototype_bank.prototypes[r].raw_prototype_vector for r in ordered_rel_ids], axis=0
        )
        cluster_mat = None
    else:
        leaf_mat = np.stack([model.tree.nodes[lid].prototype for lid in ordered_leaf_ids], axis=0)
        cluster_mat = (
            np.stack([model.tree.nodes[cid].prototype for cid in ordered_cluster_ids], axis=0)
            if ordered_cluster_ids
            else None
        )

    leaf_protos_t = torch.from_numpy(leaf_mat).to(device=device, dtype=torch.float32)
    cluster_protos_t = (
        torch.from_numpy(cluster_mat).to(device=device, dtype=torch.float32)
        if cluster_mat is not None
        else None
    )

    tau_c = max(float(model.config.routing.cluster_temperature), 1e-3)
    tau_r = max(float(model.config.routing.relation_temperature), 1e-3)

    for task in stream.tasks:
        samples = list(task.test_samples)
        bs = model.config.training.eval_batch_size
        for i in range(0, len(samples), bs):
            chunk = samples[i : i + bs]
            batch = collate_samples_for_bert(
                chunk,
                tokenizer=model.query_encoder.tokenizer,
                device=device,
                max_seq_length=model.config.prototype.max_seq_length,
            )
            true_rels = [int(s.relation_id) for s in chunk]
            raw_q = model.encode_stationary_queries(
                batch["input_ids"], batch["attention_mask"], batch["e1_indices"], batch["e2_indices"]
            )
            q_for_retrieval = (
                raw_q
                if use_raw_retrieval
                else model.prototype_bank.transform_queries(raw_q)
            )
            norm_q = F.normalize(q_for_retrieval, p=2, dim=-1, eps=1e-12)
            leaf_sims = torch.mm(norm_q, leaf_protos_t.t())

            mode = routing_override or model.config.routing.strategy
            if mode == "none":
                # Prototype-only DT-C0: rank directly by similarity
                rel_probs = F.softmax(leaf_sims / tau_r, dim=-1)
                routing_entropy = -(rel_probs * torch.log(rel_probs.clamp(min=1e-12))).sum(dim=-1)
                cos_preds = [ordered_rel_ids[idx] for idx in torch.argmax(leaf_sims, dim=-1).cpu().tolist()]
                ncm_preds = list(cos_preds)
            else:
                if mode in ("hierarchical_soft", "size_corrected_hierarchical") and cluster_protos_t is not None:
                    cluster_sims = torch.mm(norm_q, cluster_protos_t.t())
                    log_c = F.log_softmax(cluster_sims / tau_c, dim=-1)
                    log_global = torch.empty_like(leaf_sims)
                    for c_idx, cid in enumerate(ordered_cluster_ids):
                        member_lids = model.tree.nodes[cid].children_ids
                        pos = [ordered_leaf_ids.index(lid) for lid in member_lids]
                        log_cond = F.log_softmax(leaf_sims[:, pos] / tau_r, dim=-1)
                        bias = float(eta) * float(np.log(max(1, len(pos)))) if mode == "size_corrected_hierarchical" else 0.0
                        log_global[:, pos] = log_c[:, c_idx : c_idx + 1] + log_cond + bias
                    rel_probs = F.softmax(log_global, dim=-1)
                elif mode == "hybrid_tree_routing" and cluster_protos_t is not None:
                    cluster_sims = torch.mm(norm_q, cluster_protos_t.t())
                    leaf_to_cidx = torch.tensor(
                        [cluster_idx_map[str(model.tree.nodes[lid].parent_id)] for lid in ordered_leaf_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    hybrid_scores = leaf_sims + float(gamma) * cluster_sims[:, leaf_to_cidx]
                    rel_probs = F.softmax(hybrid_scores / tau_r, dim=-1)
                else:
                    rel_probs = F.softmax(leaf_sims / tau_r, dim=-1)

                routing_entropy = -(rel_probs * torch.log(rel_probs.clamp(min=1e-12))).sum(dim=-1)

                # Compose prompts using top_k from model config
                k_eff = 1 if mode == "flat_hard" else min(int(model.config.routing.top_k), len(ordered_rel_ids))
                topk_p, topk_pos = torch.topk(rel_probs, k=k_eff, dim=-1)
                topk_w = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                full_leaf_w = torch.zeros_like(rel_probs).scatter_(1, topk_pos, topk_w)
                full_cluster_w = None
                if model.prompt_bank.config.use_cluster_prompts and ordered_cluster_ids:
                    full_cluster_w = torch.zeros((raw_q.size(0), len(ordered_cluster_ids)), device=device)
                    leaf_to_cidx = torch.tensor(
                        [cluster_idx_map[str(model.tree.nodes[lid].parent_id)] for lid in ordered_leaf_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    full_cluster_w.scatter_add_(1, leaf_to_cidx.unsqueeze(0).expand(raw_q.size(0), -1), full_leaf_w)

                composed = model.prompt_bank.compose_from_weights(
                    leaf_ids_ordered=ordered_leaf_ids,
                    leaf_weights=full_leaf_w.to(dtype=torch.float32),
                    cluster_ids_ordered=ordered_cluster_ids,
                    cluster_weights=full_cluster_w.to(dtype=torch.float32) if full_cluster_w is not None else None,
                )
                z = model.query_encoder.forward_prompted(
                    batch["input_ids"], batch["attention_mask"], batch["e1_indices"], batch["e2_indices"], composed
                )
                cls_ordered = model.classifier.seen_relation_ids
                cos_logits = model.classifier(z)
                cos_preds = [cls_ordered[idx] for idx in torch.argmax(cos_logits, dim=-1).cpu().tolist()]
                if model.classifier.ncm_prototypes.numel() > 0:
                    ncm_logits = model.classifier.predict_ncm(z)
                    ncm_preds = [cls_ordered[idx] for idx in torch.argmax(ncm_logits, dim=-1).cpu().tolist()]
                else:
                    ncm_preds = list(cos_preds)

            k_rank = min(max_k, len(ordered_rel_ids))
            ranked_probs, ranked_pos = torch.topk(rel_probs, k=k_rank, dim=-1)
            ranked_pos_list = ranked_pos.cpu().tolist()
            ranked_probs_list = ranked_probs.cpu().tolist()
            ent_list = routing_entropy.cpu().tolist()

            for idx, r_true in enumerate(true_rels):
                ranked_rels = [ordered_rel_ids[p] for p in ranked_pos_list[idx]]
                top1_r = ranked_rels[0]
                true_cid = model.tree.nodes[model.tree.relation_to_node[r_true]].parent_id
                pred_cid = model.tree.nodes[model.tree.relation_to_node[top1_r]].parent_id
                records.append({
                    "true_rel": r_true,
                    "pred_ncm": ncm_preds[idx],
                    "pred_cos": cos_preds[idx],
                    "ranked_rel_ids": ranked_rels,
                    "top1_prob": float(ranked_probs_list[idx][0]),
                    "routing_top1_correct": top1_r == r_true,
                    "cluster_top1_correct": true_cid == pred_cid,
                    "entropy": float(ent_list[idx]),
                })

    summary = compute_m7_routing_decomposition(records, model.tree, stream.relation_to_task, ks=ks)
    return summary, records
