"""Top-level DynaTreeCREModel tying encoder, prototype bank, tree, prompts, router, and classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

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

    @property
    def is_prototype_only(self) -> bool:
        return self.config.routing.strategy == "none" or self.config.method.variant == "DT-C0"

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

        leaf_ret, cluster_ret = self.prototype_bank.get_retrieval_prototypes(
            self.tree, self.tree.seen_relation_ids(), self.tree.cluster_ids()
        )
        routing: RoutingResult = self.router.route(
            queries=retrieval_q,
            tree=self.tree,
            prompt_bank=self.prompt_bank,
            strategy_override=strategy_override,
            gold_relation_ids=gold_relation_ids,
            leaf_protos_override=leaf_ret,
            cluster_protos_override=cluster_ret,
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
        prompt_b = self.prompt_bank.persistent_bytes() if not self.is_prototype_only else 0
        cls_b = self.classifier.persistent_bytes() if not self.is_prototype_only else 0
        return {
            "persistent_prototype_bytes": proto_b,
            "persistent_prompt_bytes": prompt_b,
            "classifier_bytes": cls_b,
            "total_persistent_state_bytes": proto_b + prompt_b + cls_b,
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "prototype_bank": self.prototype_bank.state_dict(),
            "tree": self.tree.to_dict(),
            "prompt_bank_state": self.prompt_bank.state_dict(),
            "prompt_bank_extra": self.prompt_bank.extra_state(),
            "classifier_state": self.classifier.state_dict(),
            "classifier_meta": self.classifier.metadata_state(),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.prototype_bank = SemanticPrototypeBank.from_state_dict(state["prototype_bank"])
        self.tree = DynamicSemanticTree.from_dict(state["tree"])
        self.prompt_bank.restore_from_tree_and_state(
            self.tree, state["prompt_bank_state"], state["prompt_bank_extra"]
        )
        self.prompt_bank.to(self.device)
        self.classifier.restore_metadata_and_state(
            state["classifier_meta"], state["classifier_state"]
        )
        self.classifier.to(self.device)
