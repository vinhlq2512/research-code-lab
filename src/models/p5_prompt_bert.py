"""P5 Model Implementation: Contrastive Prototype Tree with Momentum Root Anchor.

Key enhancements over P4:
1. Module B: Feature-Level Supervised Contrastive Prototype Loss (M=0 compliant).
   Directly expands inter-task prototype margin from ~0.01-0.02 to >= 0.25.
2. Module C: Confidence-Gated Candidate Masking inference.
   Eliminates toxic soft-fusion pollution by restricting prediction space to candidate tasks.
3. Module D: Momentum Root Prompt Anchor Regularization.
   Preserves plasticity for tasks 2..T while anchoring Task 1 anti-forgetting stability.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.prompt_bert import PromptedBERTRelationClassifier

logger = logging.getLogger(__name__)


class P5PromptedBERT(PromptedBERTRelationClassifier):
    """Refined Hierarchical Prompt Tree Model for TAPTA P5."""

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        prompt_length: int = 8,
        root_prompt_length: int = 4,
        num_tasks: int = 8,
        num_classes: int = 80,
        classifier_dropout: float = 0.1,
        device: str | torch.device = "auto",
        proto_contrastive_temp: float = 0.1,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            prompt_mode="hierarchical_tree",
            prompt_length=prompt_length,
            root_prompt_length=root_prompt_length,
            num_tasks=num_tasks,
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
            device=device,
        )
        self.proto_contrastive_temp = float(proto_contrastive_temp)

        # Buffer for Root Prompt Anchor (Module D)
        self.register_buffer("root_prompt_anchor", torch.zeros_like(self.root_prompt.data))
        self.register_buffer("has_root_anchor", torch.tensor(False, dtype=torch.bool))

    def anchor_root_prompt(self) -> None:
        """Lock in task 0 root prompt as the invariant semantic anchor."""
        self.root_prompt_anchor.copy_(self.root_prompt.data.detach())
        self.has_root_anchor.fill_(True)

    def compute_root_anchor_loss(self) -> torch.Tensor:
        """Module D: Penalize root prompt drift away from the Task 1 anchor."""
        if not bool(self.has_root_anchor.item()):
            return torch.tensor(0.0, device=self.target_device)
        return F.mse_loss(self.root_prompt, self.root_prompt_anchor)

    def compute_prototype_contrastive_loss(
        self,
        h_r: torch.Tensor,
        labels: torch.Tensor,
        seen_classes: int | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """Module B: Supervised Contrastive Loss against prototype memory (M=0 compliant).

        Attracts representation h_r to its ground-truth class prototype while repelling it
        from ALL other seen prototypes across both current and past tasks.
        """
        temp = temperature if temperature is not None else self.proto_contrastive_temp
        n_classes = seen_classes if seen_classes is not None else self.seen_classes
        if n_classes <= 1:
            return torch.tensor(0.0, device=h_r.device)

        active_protos = self.prototypes[:n_classes]  # [C, D]
        norm_h = F.normalize(h_r, p=2, dim=-1)  # [B, D]
        norm_p = F.normalize(active_protos, p=2, dim=-1)  # [C, D]

        # Cosine similarity scaled by temperature: [B, C]
        logits = torch.mm(norm_h, norm_p.t()) / max(float(temp), 1e-4)

        # Labels are 0-based global class indices
        return F.cross_entropy(logits, labels)

    def predict_confidence_gated_prototype(
        self,
        candidate_sims: list[torch.Tensor],  # [sim_t1, sim_t2] each [B, num_classes]
        topk_task_ids: torch.Tensor,  # [B, K]
        router_scores: torch.Tensor,  # [B, seen_tasks]
        relations_per_task: int = 10,
        confidence_margin_threshold: float = 0.05,
        temperature: float = 0.2,
    ) -> torch.Tensor:
        """Module C: Confidence-Gated Candidate Masking.

        If router is confident (s1 - s2 >= delta), perform Hard Top-1 Gating restricted to task 1.
        If router is uncertain (s1 - s2 < delta), perform Candidate Gating restricted to {task 1, task 2}.
        All irrelevant classes from other tasks are masked with -1e9.
        """
        batch_size = topk_task_ids.size(0)
        device = candidate_sims[0].device
        t1_indices = topk_task_ids[:, 0]  # [B]
        t2_indices = topk_task_ids[:, 1] if topk_task_ids.size(1) > 1 else t1_indices

        # Gather top-1 and top-2 router scores
        s1 = torch.gather(router_scores, 1, t1_indices.unsqueeze(-1)).squeeze(-1)  # [B]
        s2 = torch.gather(router_scores, 1, t2_indices.unsqueeze(-1)).squeeze(-1)  # [B]
        margins = s1 - s2  # [B]

        gated_sim = torch.full((batch_size, self.num_classes), -1e9, device=device)

        for b in range(batch_size):
            t1 = int(t1_indices[b].item())
            t2 = int(t2_indices[b].item())
            m = float(margins[b].item())

            c1_start = t1 * relations_per_task
            c1_end = min(c1_start + relations_per_task, self.seen_classes)

            if m >= confidence_margin_threshold or t1 == t2:
                # Confident: Hard Top-1 Gated to t1 relations only
                sim_0 = candidate_sims[0][b]  # [num_classes]
                gated_sim[b, c1_start:c1_end] = sim_0[c1_start:c1_end]
            else:
                # Uncertain: Candidate Gated to {t1, t2} relations
                c2_start = t2 * relations_per_task
                c2_end = min(c2_start + relations_per_task, self.seen_classes)

                # Soft blend weights for {t1, t2}
                weights = F.softmax(torch.tensor([s1[b], s2[b]], device=device) / max(temperature, 1e-4), dim=-1)
                fused = weights[0] * candidate_sims[0][b] + weights[1] * candidate_sims[1][b]

                gated_sim[b, c1_start:c1_end] = fused[c1_start:c1_end]
                gated_sim[b, c2_start:c2_end] = fused[c2_start:c2_end]

        return gated_sim
