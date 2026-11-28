"""Top-level DynaTreeCREModel tying encoder, prototype bank, tree, prompts, router, and classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dynatree_cre.analytic import IncrementalAnalyticHead
from src.dynatree_cre.classifier import ExpandableCosineClassifier
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.prototype_bank import SemanticPrototypeBank
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.query_encoder import StableQueryEncoder
from src.dynatree_cre.router import RoutingResult, TaskAgnosticRouter
from src.dynatree_cre.tree import DynamicSemanticTree


class DynaTreeCREModel(nn.Module):
    """Unified model orchestrating stationary queries, semantic prompt tree, and dual heads."""

    def __init__(self, config: DynaTreeConfig, local_files_only: bool = True) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.query_encoder = StableQueryEncoder(
            backbone_name=config.prototype.encoder,
            representation_strategy=config.prototype.representation_strategy,
            device=config.training.device,
            local_files_only=local_files_only,
        )
        self.device = self.query_encoder.target_device
        self.rep_dim = self.query_encoder.rep_dim
        self.hidden_size = self.query_encoder.hidden_size

        self.prototype_bank = SemanticPrototypeBank(
            rep_dim=self.rep_dim,
            encoder_version=config.prototype.encoder,
            representation_strategy=config.prototype.representation_strategy,
            similarity_space=config.prototype.similarity_space,
            retrieval_similarity_space=config.prototype.retrieval_similarity_space,
        )
        self.tree = DynamicSemanticTree(
            rep_dim=self.rep_dim,
            cluster_weighting=config.tree.cluster_weighting,
        )
        self.prompt_bank = HierarchicalPromptBank(
            prompt_length=config.prompt.length,
            hidden_size=self.hidden_size,
            config=config.prompt,
        ).to(self.device)
        self.router = TaskAgnosticRouter(config.routing)
        self.classifier = ExpandableCosineClassifier(
            rep_dim=self.rep_dim,
            config=config.classifier,
        ).to(self.device)
        self.analytic_head: IncrementalAnalyticHead | None = (
            IncrementalAnalyticHead(
                input_dim=self.rep_dim,
                projection_mode=config.prototype.analytic_projection_mode,
                projection_dim=config.prototype.analytic_projection_dim,
                projection_seed=config.prototype.analytic_projection_seed,
                ridge_lambda=config.prototype.analytic_ridge_lambda,
            )
            if config.prototype.use_analytic_retrieval
            else None
        )

    @property
    def is_prototype_only(self) -> bool:
        return self.config.routing.strategy == "none" or self.config.method.variant in ("DT-C0", "DT-C0.5")

    def get_retrieval_context(
        self, raw_q: torch.Tensor
    ) -> tuple[list[int], list[str], Any, Any, torch.Tensor | None]:
        """Gather retrieval prototypes and analytic scores (when enabled) cleanly."""
        ordered_rel_ids = self.tree.seen_relation_ids()
        ordered_cluster_ids = self.tree.cluster_ids()
        leaf_ret, cluster_ret = self.prototype_bank.get_retrieval_prototypes(
            self.tree, ordered_rel_ids, ordered_cluster_ids
        )
        analytic_scores: torch.Tensor | None = None
        if self.analytic_head is not None and self.analytic_head.ordered_rel_ids:
            analytic_scores, _ = self.analytic_head.predict(raw_q, target_rel_order=ordered_rel_ids)
        return ordered_rel_ids, ordered_cluster_ids, leaf_ret, cluster_ret, analytic_scores

    @torch.no_grad()
    def encode_stationary_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.query_encoder.encode_query(input_ids, attention_mask, e1_indices, e2_indices)

    def forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        strategy_override: str | None = None,
        gold_relation_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run full query -> router -> prompt composition -> prompted encoder -> cosine logits."""
        raw_q = self.encode_stationary_queries(input_ids, attention_mask, e1_indices, e2_indices)
        transformed_q = self.prototype_bank.transform_queries(raw_q)
        retrieval_q = self.prototype_bank.transform_queries_for_retrieval(raw_q)

        if self.is_prototype_only:
            if self.analytic_head is not None:
                sims, ordered_ids = self.analytic_head.predict(raw_q)
            else:
                sims, ordered_ids = self.prototype_bank.predict_nearest_prototype(raw_q)
            return {
                "raw_queries": raw_q,
                "transformed_queries": transformed_q,
                "logits": sims,
                "ncm_logits": sims,
                "ordered_relation_ids": ordered_ids,
                "routing": None,
                "representations": retrieval_q,
            }

        if (
            strategy_override is None
            and self.config.routing.inference_mode == "candidate_rerank"
        ):
            return self.forward_candidate_rerank(
                input_ids=input_ids,
                attention_mask=attention_mask,
                e1_indices=e1_indices,
                e2_indices=e2_indices,
                raw_q=raw_q,
                transformed_q=transformed_q,
                retrieval_q=retrieval_q,
            )

        _, _, leaf_ret, cluster_ret, analytic_scores = self.get_retrieval_context(raw_q)
        routing: RoutingResult = self.router.route(
            queries=retrieval_q,
            tree=self.tree,
            prompt_bank=self.prompt_bank,
            strategy_override=strategy_override,
            gold_relation_ids=gold_relation_ids,
            leaf_protos_override=leaf_ret,
            cluster_protos_override=cluster_ret,
            analytic_scores_override=analytic_scores,
        )
        z = self.query_encoder.forward_prompted(
            input_ids=input_ids,
            attention_mask=attention_mask,
            e1_indices=e1_indices,
            e2_indices=e2_indices,
            prompt_embeds=routing.composed_prompt,
        )
        logits = self.classifier(z)
        ncm_logits = (
            self.classifier.predict_ncm(z)
            if self.classifier.ncm_prototypes.numel() > 0
            else None
        )
        return {
            "raw_queries": raw_q,
            "transformed_queries": transformed_q,
            "representations": z,
            "logits": logits,
            "ncm_logits": ncm_logits,
            "ordered_relation_ids": self.classifier.seen_relation_ids,
            "routing": routing,
        }

    @staticmethod
    def normalize_candidate_scores(scores: torch.Tensor, method: str) -> torch.Tensor:
        """Normalize candidate scores [B, K] within each sample's candidate set."""
        if scores.size(-1) <= 1 or method == "none":
            return scores
        if method == "zscore":
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, unbiased=False, keepdim=True)
            return torch.where(std > 1e-6, (scores - mean) / std.clamp(min=1e-6), torch.zeros_like(scores))
        if method == "minmax":
            lo = scores.min(dim=-1, keepdim=True).values
            hi = scores.max(dim=-1, keepdim=True).values
            span = hi - lo
            return torch.where(span > 1e-6, (scores - lo) / span.clamp(min=1e-6), torch.zeros_like(scores))
        raise ValueError(f"Unsupported rerank_score_norm: {method}")

    def fuse_rerank_scores(
        self,
        ret_scores: torch.Tensor,
        prompt_scores: torch.Tensor,
        fusion_mode: str,
        fusion_lambda: float,
    ) -> torch.Tensor:
        """Fuse retrieval scores and candidate-conditioned prompted scores over [B, K]."""
        if ret_scores.size(-1) <= 1 or fusion_mode == "retrieval_only":
            return ret_scores
        if fusion_mode == "prompted_only":
            return prompt_scores
        norm_method = self.config.routing.rerank_score_norm
        r_n = self.normalize_candidate_scores(ret_scores, norm_method)
        p_n = self.normalize_candidate_scores(prompt_scores, norm_method)
        lam = float(fusion_lambda)
        return lam * r_n + (1.0 - lam) * p_n

    @torch.no_grad()
    def forward_candidate_rerank(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        top_k_override: int | None = None,
        fusion_mode_override: str | None = None,
        fusion_lambda_override: float | None = None,
        raw_q: torch.Tensor | None = None,
        transformed_q: torch.Tensor | None = None,
        retrieval_q: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Execute M7.4 Stages 1-6: Retrieve Top-K, forward per-candidate prompts, and fuse."""
        if raw_q is None:
            raw_q = self.encode_stationary_queries(input_ids, attention_mask, e1_indices, e2_indices)
        if transformed_q is None:
            transformed_q = self.prototype_bank.transform_queries(raw_q)
        if retrieval_q is None:
            retrieval_q = self.prototype_bank.transform_queries_for_retrieval(raw_q)

        ordered_rel_ids, ordered_cluster_ids, leaf_ret, cluster_ret, analytic_scores = (
            self.get_retrieval_context(raw_q)
        )
        raw_scores, rel_probs, cluster_probs, _, _, _ = self.router.compute_retrieval_scores(
            queries=retrieval_q,
            tree=self.tree,
            leaf_protos_override=leaf_ret,
            cluster_protos_override=cluster_ret,
            analytic_scores_override=analytic_scores,
        )

        batch_size = input_ids.size(0)
        n_seen = len(ordered_rel_ids)
        k_eff = min(int(top_k_override or self.config.routing.top_k), n_seen)
        cand_ret_scores, topk_pos = torch.topk(raw_scores, k=k_eff, dim=-1)
        rel_id_tensor = torch.tensor(ordered_rel_ids, dtype=torch.long, device=input_ids.device)
        cand_rel_ids = rel_id_tensor[topk_pos]  # [B, K]

        fusion_mode = fusion_mode_override or self.config.routing.rerank_fusion_mode
        fusion_lam = (
            float(fusion_lambda_override)
            if fusion_lambda_override is not None
            else float(self.config.routing.rerank_fusion_lambda)
        )

        cls_slots = torch.tensor(
            [[self.classifier.relation_to_slot[int(r)] for r in row] for row in cand_rel_ids.tolist()],
            dtype=torch.long,
            device=input_ids.device,
        )
        prompt_cos_scores = torch.zeros_like(cand_ret_scores)
        prompt_ncm_scores = torch.zeros_like(cand_ret_scores)
        need_prompted = (fusion_mode != "retrieval_only") and (k_eff > 1)
        if need_prompted:
            flat_cands = cand_rel_ids.reshape(-1)
            cand_prompts = self.prompt_bank.compose_for_relation_ids(flat_cands, self.tree)
            rep_input_ids = input_ids.repeat_interleave(k_eff, dim=0)
            rep_mask = attention_mask.repeat_interleave(k_eff, dim=0)
            rep_e1 = e1_indices.repeat_interleave(k_eff, dim=0)
            rep_e2 = e2_indices.repeat_interleave(k_eff, dim=0)

            chunk_sz = max(1, int(self.config.routing.rerank_forward_chunk_size))
            z_chunks: list[torch.Tensor] = []
            for st in range(0, rep_input_ids.size(0), chunk_sz):
                ed = st + chunk_sz
                z_chunks.append(
                    self.query_encoder.forward_prompted(
                        rep_input_ids[st:ed],
                        rep_mask[st:ed],
                        rep_e1[st:ed],
                        rep_e2[st:ed],
                        cand_prompts[st:ed],
                    )
                )
            z_cand = torch.cat(z_chunks, dim=0).view(batch_size, k_eff, self.rep_dim)
            z_norm = F.normalize(z_cand, p=2, dim=-1, eps=1e-12)

            all_w = F.normalize(self.classifier._all_weights(), p=2, dim=-1, eps=1e-12)
            cand_w = all_w[cls_slots]
            prompt_cos_scores = self.classifier.scale * (z_norm * cand_w).sum(dim=-1)

            if self.classifier.ncm_prototypes.numel() > 0:
                all_p = F.normalize(self.classifier.ncm_prototypes, p=2, dim=-1, eps=1e-12)
                cand_p = all_p[cls_slots]
                prompt_ncm_scores = (z_norm * cand_p).sum(dim=-1)
            else:
                prompt_ncm_scores = prompt_cos_scores

        fused_cos = self.fuse_rerank_scores(cand_ret_scores, prompt_cos_scores, fusion_mode, fusion_lam)
        fused_ncm = self.fuse_rerank_scores(cand_ret_scores, prompt_ncm_scores, fusion_mode, fusion_lam)

        cls_ordered = self.classifier.seen_relation_ids
        full_cos_logits = torch.full(
            (batch_size, len(cls_ordered)), float("-inf"), device=input_ids.device, dtype=fused_cos.dtype
        ).scatter_(1, cls_slots, fused_cos)
        full_ncm_logits = torch.full(
            (batch_size, len(cls_ordered)), float("-inf"), device=input_ids.device, dtype=fused_ncm.dtype
        ).scatter_(1, cls_slots, fused_ncm)

        routing_entropy = -(rel_probs * torch.log(rel_probs.clamp(min=1e-12))).sum(dim=-1)
        topk_p = rel_probs.gather(1, topk_pos)
        topk_w = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        routing = RoutingResult(
            cluster_probabilities=cluster_probs,
            relation_probabilities=rel_probs,
            selected_relation_ids=cand_rel_ids,
            selected_cluster_ids=None,
            normalized_weights=topk_w,
            composed_prompt=torch.empty(0, device=input_ids.device),
            routing_entropy=routing_entropy,
            ordered_relation_ids=ordered_rel_ids,
            ordered_cluster_ids=ordered_cluster_ids,
        )
        return {
            "raw_queries": raw_q,
            "transformed_queries": transformed_q,
            "representations": raw_q,
            "logits": full_cos_logits,
            "ncm_logits": full_ncm_logits,
            "ordered_relation_ids": cls_ordered,
            "routing": routing,
            "rerank_details": {
                "cand_rel_ids": cand_rel_ids,
                "ret_scores": cand_ret_scores,
                "prompt_cos_scores": prompt_cos_scores,
                "prompt_ncm_scores": prompt_ncm_scores,
            },
        }

    def trainable_parameter_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        p_params = self.prompt_bank.trainable_parameters()
        if p_params:
            groups.append(
                {
                    "params": p_params,
                    "lr": float(self.config.training.learning_rate),
                    "weight_decay": float(self.config.training.weight_decay),
                }
            )
        c_params = self.classifier.trainable_parameters()
        if c_params:
            groups.append(
                {
                    "params": c_params,
                    "lr": float(self.config.training.classifier_lr),
                    "weight_decay": float(self.config.training.weight_decay),
                }
            )
        return groups

    def snapshot_frozen_state(self) -> dict[str, torch.Tensor]:
        snaps: dict[str, torch.Tensor] = {}
        for name, param in self.query_encoder.named_parameters():
            snaps[f"encoder.{name}"] = param.detach().clone().cpu()
        for name, t in self.prompt_bank.snapshot_frozen_tensors().items():
            snaps[f"prompt_bank.{name}"] = t
        for name, t in self.classifier.snapshot_frozen_weights().items():
            snaps[f"classifier.{name}"] = t
        return snaps

    def assert_frozen_unchanged(self, before: dict[str, torch.Tensor]) -> None:
        current = {}
        for name, param in self.query_encoder.named_parameters():
            current[f"encoder.{name}"] = param.detach().cpu()
        for name, param in self.prompt_bank.named_parameters():
            current[f"prompt_bank.{name}"] = param.detach().cpu()
        for name, t in self.classifier.snapshot_frozen_weights().items():
            current[f"classifier.{name}"] = t

        for key, snap in before.items():
            if not torch.equal(snap, current[key]):
                raise AssertionError(f"Frozen tensor '{key}' mutated during training!")

    def memory_breakdown_bytes(self) -> dict[str, int]:
        proto_b = self.prototype_bank.memory_bytes()
        analytic_b = self.analytic_head.persistent_bytes() if self.analytic_head is not None else 0
        prompt_b = self.prompt_bank.persistent_bytes() if not self.is_prototype_only else 0
        cls_b = self.classifier.persistent_bytes() if not self.is_prototype_only else 0
        return {
            "persistent_prototype_bytes": proto_b,
            "persistent_analytic_bytes": analytic_b,
            "persistent_prompt_bytes": prompt_b,
            "classifier_bytes": cls_b,
            "total_persistent_state_bytes": proto_b + analytic_b + prompt_b + cls_b,
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "prototype_bank": self.prototype_bank.state_dict(),
            "analytic_head": self.analytic_head.state_dict() if self.analytic_head is not None else None,
            "tree": self.tree.to_dict(),
            "prompt_bank_state": self.prompt_bank.state_dict(),
            "prompt_bank_extra": self.prompt_bank.extra_state(),
            "classifier_state": self.classifier.state_dict(),
            "classifier_meta": self.classifier.metadata_state(),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.prototype_bank = SemanticPrototypeBank.from_state_dict(state["prototype_bank"])
        if state.get("analytic_head") is not None:
            self.analytic_head = IncrementalAnalyticHead.from_state_dict(state["analytic_head"])
        self.tree = DynamicSemanticTree.from_dict(state["tree"])
        self.prompt_bank.restore_from_tree_and_state(
            self.tree, state["prompt_bank_state"], state["prompt_bank_extra"]
        )
        self.prompt_bank.to(self.device)
        self.classifier.restore_metadata_and_state(
            state["classifier_meta"], state["classifier_state"]
        )
        self.classifier.to(self.device)
