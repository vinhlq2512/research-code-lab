"""Hierarchical Residual Prompt Bank with strict old-node freeze policy."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.dynatree_cre.config import PromptConfig
from src.dynatree_cre.tree import DynamicSemanticTree


class HierarchicalPromptBank(nn.Module):
    """Stores root, cluster, and leaf residual prompts of shape [L, d]."""

    def __init__(
        self,
        prompt_length: int,
        hidden_size: int,
        config: PromptConfig,
    ) -> None:
        super().__init__()
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(hidden_size)
        self.config = config

        if config.use_root_prompt:
            self.root_prompt: nn.Parameter | None = nn.Parameter(
                torch.empty(self.prompt_length, self.hidden_size)
            )
            nn.init.normal_(self.root_prompt, mean=0.0, std=0.02)
        else:
            self.root_prompt = None

        self.cluster_prompts = nn.ParameterDict()
        self.leaf_prompts = nn.ParameterDict()
        self.node_created_task: dict[str, int] = {"root": 0}
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def _init_tensor(
        self,
        tensor: torch.Tensor,
        role: str,
        word_embedding_weight: torch.Tensor | None = None,
    ) -> None:
        if self.config.init_strategy == "zero_leaf" and role == "leaf":
            nn.init.zeros_(tensor)
        elif self.config.init_strategy == "embedding_stats" and word_embedding_weight is not None:
            with torch.no_grad():
                std = float(word_embedding_weight.std().clamp(min=1e-4, max=0.05).item())
                scale = std * (0.5 if role == "leaf" else 1.0)
                tensor.normal_(mean=0.0, std=scale)
        else:
            std = 0.01 if role == "leaf" else 0.02
            nn.init.normal_(tensor, mean=0.0, std=std)

    def synchronize_with_tree(
        self,
        tree: DynamicSemanticTree,
        task_id: int,
        word_embedding_weight: torch.Tensor | None = None,
    ) -> list[str]:
        """Allocate prompt parameters only for newly created tree nodes."""
        if self.root_prompt is not None:
            device = self.root_prompt.device
        elif len(self.leaf_prompts) > 0:
            device = next(iter(self.leaf_prompts.values())).device
        elif word_embedding_weight is not None:
            device = word_embedding_weight.device
        else:
            device = self._device_anchor.device
        created_nodes: list[str] = []

        if self.config.use_cluster_prompts:
            for cid in tree.cluster_ids():
                if cid not in self.cluster_prompts:
                    param = nn.Parameter(
                        torch.empty(self.prompt_length, self.hidden_size, device=device)
                    )
                    self._init_tensor(param, role="cluster", word_embedding_weight=word_embedding_weight)
                    self.cluster_prompts[cid] = param
                    self.node_created_task[cid] = int(tree.nodes[cid].created_at_task)
                    created_nodes.append(cid)

        for lid in tree.leaf_ids():
            if lid not in self.leaf_prompts:
                param = nn.Parameter(
                    torch.empty(self.prompt_length, self.hidden_size, device=device)
                )
                self._init_tensor(param, role="leaf", word_embedding_weight=word_embedding_weight)
                self.leaf_prompts[lid] = param
                self.node_created_task[lid] = int(tree.nodes[lid].created_at_task)
                created_nodes.append(lid)

        return created_nodes

    def apply_freeze_policy(self, task_id: int) -> None:
        """Freeze old root, old clusters, and old leaves strictly by creation task."""
        if self.root_prompt is not None:
            should_freeze_root = (task_id > 0) and self.config.freeze_old_root
            self.root_prompt.requires_grad = not should_freeze_root

        for cid, param in self.cluster_prompts.items():
            created = self.node_created_task.get(cid, 0)
            should_freeze = (created < task_id) and self.config.freeze_old_clusters
            param.requires_grad = not should_freeze

        for lid, param in self.leaf_prompts.items():
            created = self.node_created_task.get(lid, 0)
            should_freeze = (created < task_id) and self.config.freeze_old_leaves
            param.requires_grad = not should_freeze

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only parameters with requires_grad=True so AdamW weight decay cannot drift old nodes."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_effective_prompt(self, relation_id: int, tree: DynamicSemanticTree) -> torch.Tensor:
        """Compute P_r^effective = P_root + P_cluster(r) + Delta P_r for a single relation."""
        lid = tree.relation_to_node[int(relation_id)]
        cid = tree.nodes[lid].parent_id
        prompt = self.leaf_prompts[lid]
        if self.config.use_cluster_prompts and cid is not None and cid in self.cluster_prompts:
            prompt = prompt + self.cluster_prompts[cid]
        if self.config.use_root_prompt and self.root_prompt is not None:
            prompt = prompt + self.root_prompt
        return prompt

    def compose_from_weights(
        self,
        leaf_ids_ordered: list[str],
        leaf_weights: torch.Tensor,
        cluster_ids_ordered: list[str] | None = None,
        cluster_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compose [B, L, d] batch prompts: P_root + sum_v beta_v P_v + sum_r alpha_r Delta P_r."""
        leaf_stack = torch.stack([self.leaf_prompts[lid] for lid in leaf_ids_ordered], dim=0)
        composed = torch.einsum("br,rld->bld", leaf_weights, leaf_stack)

        if (
            self.config.use_cluster_prompts
            and cluster_ids_ordered is not None
            and cluster_weights is not None
            and len(cluster_ids_ordered) > 0
        ):
            cluster_stack = torch.stack(
                [self.cluster_prompts[cid] for cid in cluster_ids_ordered], dim=0
            )
            composed = composed + torch.einsum("bc,cld->bld", cluster_weights, cluster_stack)

        if self.config.use_root_prompt and self.root_prompt is not None:
            composed = composed + self.root_prompt.unsqueeze(0)

        return composed

    def snapshot_frozen_tensors(self) -> dict[str, torch.Tensor]:
        """Take detached CPU snapshots of all currently frozen prompt tensors."""
        snaps: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                snaps[name] = param.detach().clone().cpu()
        return snaps

    def persistent_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.parameters()))

    def extra_state(self) -> dict[str, Any]:
        return {
            "node_created_task": dict(self.node_created_task),
            "cluster_keys": sorted(self.cluster_prompts.keys()),
            "leaf_keys": sorted(self.leaf_prompts.keys()),
        }

    def restore_from_tree_and_state(
        self, tree: DynamicSemanticTree, state_dict: dict[str, Any], extra: dict[str, Any]
    ) -> None:
        """Rebuild ParameterDict keys before loading checkpoint weights."""
        self.node_created_task = dict(extra.get("node_created_task", {"root": 0}))
        dev = self._device_anchor.device
        for cid in extra.get("cluster_keys", tree.cluster_ids()):
            if cid not in self.cluster_prompts:
                self.cluster_prompts[cid] = nn.Parameter(
                    torch.zeros(self.prompt_length, self.hidden_size, device=dev)
                )
        for lid in extra.get("leaf_keys", tree.leaf_ids()):
            if lid not in self.leaf_prompts:
                self.leaf_prompts[lid] = nn.Parameter(
                    torch.zeros(self.prompt_length, self.hidden_size, device=dev)
                )
        self.load_state_dict(state_dict, strict=True)
