"""P5 Router Implementation: Whitened Anisotropy-Resistant Task Router.

Key enhancements over P4:
1. Optimal Subcentroid Aggregation (alpha=0.4 gives higher accuracy than 0.7).
2. Regularized Covariance Whitening (W = U Lambda^{-1/2}) to eliminate unprompted BERT anisotropy.
3. Confidence-Gated Output: Computes confidence margins (s1 - s2) for Module C gating.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.task_router import TaskRouter

logger = logging.getLogger(__name__)


class P5TaskRouter(TaskRouter):
    """Anisotropy-Resistant Task Router with Confidence Gating for TAPTA P5."""

    def __init__(
        self,
        input_dim: int = 1536,
        num_tasks: int = 8,
        relations_per_task: int = 10,
        subcentroid_weight: float = 0.4,  # Empirical optimum from grid search (0.4 vs 0.7)
        enable_whitening: bool = True,
        whitening_dim: int = 64,
        whitening_eps: float = 1e-4,
        confidence_threshold: float = 0.05,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_tasks=num_tasks,
            relations_per_task=relations_per_task,
            router_type="centroid",
            subcentroid_weight=subcentroid_weight,
        )
        self.enable_whitening = bool(enable_whitening)
        self.whitening_dim = int(whitening_dim)
        self.whitening_eps = float(whitening_eps)
        self.confidence_threshold = float(confidence_threshold)

        # Buffer for global query mean and whitening matrix
        self.register_buffer("global_query_mean", torch.zeros(1, self.input_dim))
        self.register_buffer("whitening_matrix", torch.eye(self.input_dim))
        self.register_buffer("whitening_ready", torch.tensor(False, dtype=torch.bool))

    def update_whitening_transform(self, seen_tasks: int) -> None:
        """Compute regularized whitening transform W = U Lambda^{-1/2} from accumulated subcentroids."""
        if not self.enable_whitening or seen_tasks <= 1:
            return

        with torch.no_grad():
            active_sub = self.relation_subcentroids[:seen_tasks].view(-1, self.input_dim)  # [seen_rels, D]
            n_samples = active_sub.size(0)
            if n_samples < 2:
                return

            mu = active_sub.mean(dim=0, keepdim=True)  # [1, D]
            diff = active_sub - mu
            cov = torch.mm(diff.t(), diff) / (n_samples - 1)  # [D, D]

            # Regularized eigendecomposition
            evals, evecs = torch.linalg.eigh(cov)

            # Keep top-K significant components to preserve signal while discarding collinear noise
            k = min(self.whitening_dim, n_samples - 1, self.input_dim)
            top_evals = evals[-k:]
            top_evecs = evecs[:, -k:]

            inv_sqrt_lambda = torch.diag(1.0 / torch.sqrt(torch.clamp(top_evals, min=self.whitening_eps)))
            W = torch.mm(top_evecs, inv_sqrt_lambda)  # [D, k]

            self.global_query_mean.copy_(mu)
            # Store W padded to [D, D] or store in dynamic attribute
            self.whitening_W = W
            self.whitening_ready.fill_(True)

    def compute_p5_scores(
        self,
        query_features: torch.Tensor,
        seen_tasks: int,
    ) -> torch.Tensor:
        """Compute router scores with optional whitening and optimal subcentroid weighting."""
        norm_q = F.normalize(query_features, p=2, dim=-1)

        # 1. Standard cosine similarities
        active_task_centroids = F.normalize(self.task_centroids[:seen_tasks], p=2, dim=-1)
        task_sim = torch.mm(norm_q, active_task_centroids.t())  # [B, seen_tasks]

        active_subcentroids = F.normalize(
            self.relation_subcentroids[:seen_tasks], p=2, dim=-1
        )  # [seen_tasks, R, D]
        rel_sims = torch.einsum("bd,trd->btr", norm_q, active_subcentroids)  # [B, T, R]
        max_rel_sim, _ = rel_sims.max(dim=-1)  # [B, T]

        alpha = self.subcentroid_weight
        scores = (1.0 - alpha) * task_sim + alpha * max_rel_sim  # [B, T]

        # 2. If whitening is active, blend with whitened alignment
        if self.enable_whitening and bool(self.whitening_ready.item()) and hasattr(self, "whitening_W"):
            W = self.whitening_W.to(query_features.device)
            mu = self.global_query_mean.to(query_features.device)
            qw = torch.mm(query_features - mu, W)
            norm_qw = F.normalize(qw, p=2, dim=-1)

            sub_flat = self.relation_subcentroids[:seen_tasks].view(-1, self.input_dim).to(query_features.device)
            sub_w = torch.mm(sub_flat - mu, W).view(seen_tasks, self.relations_per_task, -1)
            norm_subw = F.normalize(sub_w, p=2, dim=-1)

            rel_sims_w = torch.einsum("bd,trd->btr", norm_qw, norm_subw)
            max_rel_w, _ = rel_sims_w.max(dim=-1)

            # Balanced ensemble of raw and whitened representations
            scores = 0.5 * scores + 0.5 * max_rel_w

        return scores

    @torch.no_grad()
    def get_confidence_routing(
        self,
        query_features: torch.Tensor,
        seen_tasks: int,
        top_k: int = 2,
    ) -> dict[str, Any]:
        """Compute top-k tasks and confidence margins for Module C."""
        self.eval()
        scores = self.compute_p5_scores(query_features, seen_tasks=seen_tasks)
        k = max(1, min(int(top_k), int(seen_tasks)))
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)

        if k > 1:
            margins = topk_scores[:, 0] - topk_scores[:, 1]
        else:
            margins = torch.full((query_features.size(0),), 1.0, device=query_features.device)

        is_confident = margins >= self.confidence_threshold

        return {
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "margins": margins,
            "is_confident": is_confident,
            "task_scores": scores,
        }
