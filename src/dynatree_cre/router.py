"""Task-Agnostic Flat and Hierarchical Soft Routers for DynaTree-CRE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.dynatree_cre.config import RoutingConfig
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.tree import DynamicSemanticTree


@dataclass
class RoutingResult:
    """Structured routing output per batch (never stores cross-task raw instances)."""

    cluster_probabilities: torch.Tensor | None  # [B, C] or None for flat
    relation_probabilities: torch.Tensor  # [B, R_seen]
    selected_relation_ids: torch.Tensor  # [B, K_eff] (relation IDs)
    selected_cluster_ids: torch.Tensor | None  # [B, K_eff] (indices into cluster_ids_ordered)
    normalized_weights: torch.Tensor  # [B, K_eff]
    composed_prompt: torch.Tensor  # [B, L, d]
    routing_entropy: torch.Tensor  # [B]
    ordered_relation_ids: list[int]
    ordered_cluster_ids: list[str]


class TaskAgnosticRouter:
    """Computes flat or hierarchical prompt routing strictly without Task IDs."""

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    @staticmethod
    def _to_tensor(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    def compute_retrieval_scores(
        self,
        queries: torch.Tensor,
        tree: DynamicSemanticTree,
        strategy_override: str | None = None,
        leaf_protos_override: np.ndarray | None = None,
        cluster_protos_override: np.ndarray | None = None,
        analytic_scores_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[int], list[str], list[str]]:
        """Compute raw retrieval scores and normalized probabilities over observed leaves."""
        strategy = strategy_override or self.config.strategy
        device = queries.device
        dtype = queries.dtype

        ordered_rel_ids = tree.seen_relation_ids()
        ordered_leaf_ids = [tree.relation_to_node[r] for r in ordered_rel_ids]
        ordered_cluster_ids = tree.cluster_ids()
        cluster_idx_map = {cid: idx for idx, cid in enumerate(ordered_cluster_ids)}

        leaf_mat = (
            leaf_protos_override
            if leaf_protos_override is not None
            else np.stack([tree.nodes[lid].prototype for lid in ordered_leaf_ids], axis=0)
        )
        leaf_protos = self._to_tensor(leaf_mat, device=device, dtype=dtype)

        norm_q = F.normalize(queries, p=2, dim=-1, eps=1e-12)
        leaf_sims = torch.mm(norm_q, leaf_protos.t())

        tau_c = max(float(self.config.cluster_temperature), 1e-3)
        tau_r = max(float(self.config.relation_temperature), 1e-3)

        cluster_probs: torch.Tensor | None = None
        if strategy in ("hierarchical_soft", "size_corrected_hierarchical"):
            cluster_mat = (
                cluster_protos_override
                if cluster_protos_override is not None
                else np.stack([tree.nodes[cid].prototype for cid in ordered_cluster_ids], axis=0)
            )
            cluster_protos = self._to_tensor(cluster_mat, device=device, dtype=dtype)
            cluster_sims = torch.mm(norm_q, cluster_protos.t())
            log_cluster_probs = F.log_softmax(cluster_sims / tau_c, dim=-1)
            cluster_probs = torch.exp(log_cluster_probs)

            eta = float(self.config.cluster_size_eta) if strategy == "size_corrected_hierarchical" else 0.0
            log_global_leaf = torch.empty_like(leaf_sims)
            for c_idx, cid in enumerate(ordered_cluster_ids):
                member_lids = tree.nodes[cid].children_ids
                member_positions = [ordered_leaf_ids.index(lid) for lid in member_lids]
                cond_logits = leaf_sims[:, member_positions] / tau_r
                log_cond = F.log_softmax(cond_logits, dim=-1)
                size_term = eta * float(np.log(max(1, len(member_positions)))) if eta != 0.0 else 0.0
                log_global_leaf[:, member_positions] = log_cluster_probs[:, c_idx : c_idx + 1] + log_cond + size_term

            raw_scores = log_global_leaf
            rel_probs = F.softmax(log_global_leaf, dim=-1)
        elif strategy in ("hybrid_tree_routing", "hybrid_analytic_tree", "analytic_retrieval"):
            if strategy in ("analytic_retrieval", "hybrid_analytic_tree"):
                if analytic_scores_override is None:
                    raise ValueError(f"{strategy} requires analytic_scores_override.")
                base_scores = analytic_scores_override
            else:
                base_scores = leaf_sims

            gamma = float(self.config.hybrid_cluster_gamma)
            if strategy != "analytic_retrieval" and ordered_cluster_ids and gamma != 0.0:
                cluster_mat = (
                    cluster_protos_override
                    if cluster_protos_override is not None
                    else np.stack([tree.nodes[cid].prototype for cid in ordered_cluster_ids], axis=0)
                )
                cluster_protos = self._to_tensor(cluster_mat, device=device, dtype=dtype)
                cluster_sims = torch.mm(norm_q, cluster_protos.t())
                cluster_probs = F.softmax(cluster_sims / tau_c, dim=-1)
                leaf_to_cidx_t = torch.tensor(
                    [cluster_idx_map[str(tree.nodes[lid].parent_id)] for lid in ordered_leaf_ids],
                    dtype=torch.long,
                    device=device,
                )
                hybrid_scores = base_scores + gamma * cluster_sims[:, leaf_to_cidx_t]
            else:
                hybrid_scores = base_scores
            raw_scores = hybrid_scores
            rel_probs = F.softmax(hybrid_scores / tau_r, dim=-1)
        elif strategy in ("flat_hard", "flat_soft_topk", "dense_soft", "oracle_route"):
            raw_scores = leaf_sims
            rel_probs = F.softmax(leaf_sims / tau_r, dim=-1)
        else:
            raise ValueError(f"Unsupported routing strategy for prompt routing: {strategy}")

        return raw_scores, rel_probs, cluster_probs, ordered_rel_ids, ordered_leaf_ids, ordered_cluster_ids

    def route(
        self,
        queries: torch.Tensor,
        tree: DynamicSemanticTree,
        prompt_bank: HierarchicalPromptBank,
        strategy_override: str | None = None,
        gold_relation_ids: torch.Tensor | None = None,
        leaf_protos_override: np.ndarray | None = None,
        cluster_protos_override: np.ndarray | None = None,
        analytic_scores_override: torch.Tensor | None = None,
    ) -> RoutingResult:
        """Route queries [B, D] over observed tree nodes and compose prompts [B, L, d]."""
        strategy = strategy_override or self.config.strategy
        device = queries.device
        dtype = queries.dtype
        batch_size = queries.size(0)

        (
            _raw_scores,
            rel_probs,
            cluster_probs,
            ordered_rel_ids,
            ordered_leaf_ids,
            ordered_cluster_ids,
        ) = self.compute_retrieval_scores(
            queries=queries,
            tree=tree,
            strategy_override=strategy,
            leaf_protos_override=leaf_protos_override,
            cluster_protos_override=cluster_protos_override,
            analytic_scores_override=analytic_scores_override,
        )
        cluster_idx_map = {cid: idx for idx, cid in enumerate(ordered_cluster_ids)}

        # Entropy over global relation probability distribution
        routing_entropy = -(rel_probs * torch.log(rel_probs.clamp(min=1e-12))).sum(dim=-1)

        # Select active leaves & renormalize weights
        n_seen = len(ordered_rel_ids)
        if strategy == "oracle_route":
            if gold_relation_ids is None:
                raise ValueError("gold_relation_ids required only for isolated oracle_route control.")
            rel_to_pos = {r: idx for idx, r in enumerate(ordered_rel_ids)}
            chosen_pos = torch.tensor(
                [[rel_to_pos[int(r.item())]] for r in gold_relation_ids],
                dtype=torch.long,
                device=device,
            )
            full_leaf_weights = torch.zeros_like(rel_probs)
            full_leaf_weights.scatter_(1, chosen_pos, 1.0)
            topk_pos = chosen_pos
            topk_weights = torch.ones((batch_size, 1), device=device, dtype=dtype)
        elif strategy == "dense_soft":
            topk_pos = torch.arange(n_seen, device=device).unsqueeze(0).expand(batch_size, -1)
            topk_weights = rel_probs
            full_leaf_weights = rel_probs
        else:
            k_eff = 1 if strategy == "flat_hard" else min(int(self.config.top_k), n_seen)
            topk_probs, topk_pos = torch.topk(rel_probs, k=k_eff, dim=-1)
            topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            full_leaf_weights = torch.zeros_like(rel_probs)
            full_leaf_weights.scatter_(1, topk_pos, topk_weights)

        # Aggregate cluster weights beta_v = sum_{r in TopK: cluster(r) = v} alpha_r
        full_cluster_weights: torch.Tensor | None = None
        selected_cluster_idx: torch.Tensor | None = None
        if prompt_bank.config.use_cluster_prompts and ordered_cluster_ids:
            full_cluster_weights = torch.zeros(
                (batch_size, len(ordered_cluster_ids)), device=device, dtype=dtype
            )
            leaf_to_cidx = torch.tensor(
                [cluster_idx_map[str(tree.nodes[lid].parent_id)] for lid in ordered_leaf_ids],
                dtype=torch.long,
                device=device,
            )
            cidx_expand = leaf_to_cidx.unsqueeze(0).expand(batch_size, -1)
            full_cluster_weights.scatter_add_(1, cidx_expand, full_leaf_weights)
            selected_cluster_idx = leaf_to_cidx[topk_pos]

        composed_prompt = prompt_bank.compose_from_weights(
            leaf_ids_ordered=ordered_leaf_ids,
            leaf_weights=full_leaf_weights.to(dtype=torch.float32),
            cluster_ids_ordered=ordered_cluster_ids,
            cluster_weights=(
                full_cluster_weights.to(dtype=torch.float32)
                if full_cluster_weights is not None
                else None
            ),
        )

        rel_id_tensor = torch.tensor(ordered_rel_ids, dtype=torch.long, device=device)
        selected_relation_ids = rel_id_tensor[topk_pos]

        return RoutingResult(
            cluster_probabilities=cluster_probs,
            relation_probabilities=rel_probs,
            selected_relation_ids=selected_relation_ids,
            selected_cluster_ids=selected_cluster_idx,
            normalized_weights=topk_weights,
            composed_prompt=composed_prompt,
            routing_entropy=routing_entropy,
            ordered_relation_ids=ordered_rel_ids,
            ordered_cluster_ids=ordered_cluster_ids,
        )
