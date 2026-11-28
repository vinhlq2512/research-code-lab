"""Lightweight Residual Contrastive Query Adapter for M7.5 Step 4 (Frozen BERT)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQueryAdapter(nn.Module):
    """Residual bottleneck MLP projecting frozen BERT [h_e1; h_e2] into a sharper retrieval space."""

    def __init__(
        self,
        rep_dim: int,
        hidden_dim: int = 256,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)

        self.fc1 = nn.Linear(self.rep_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.rep_dim)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, raw_queries: torch.Tensor) -> torch.Tensor:
        q_norm = F.normalize(raw_queries, p=2, dim=-1, eps=1e-12)
        h = self.ln(F.gelu(self.fc1(q_norm)))
        delta = self.fc2(h)
        return F.normalize(q_norm + self.residual_scale * delta, p=2, dim=-1, eps=1e-12)

    def persistent_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.parameters()))


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Batch SupCon loss on L2-normalized [B, D] features; skips singletons safely."""
    bsz = features.size(0)
    if bsz <= 1:
        return features.sum() * 0.0

    feats = F.normalize(features, p=2, dim=-1, eps=1e-12)
    tau = max(float(temperature), 1e-3)
    sim = torch.mm(feats, feats.t()) / tau

    labels_col = labels.view(-1, 1)
    same_label = torch.eq(labels_col, labels_col.t())
    not_self = ~torch.eye(bsz, dtype=torch.bool, device=features.device)
    pos_mask = same_label & not_self

    # Subtract row max excluding self for numerical stability
    sim_masked = sim.masked_fill(~not_self, float("-inf"))
    row_max = sim_masked.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    logits = sim - row_max.detach()

    exp_logits = torch.exp(logits) * not_self.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=-1, keepdim=True).clamp(min=1e-12))

    pos_counts = pos_mask.sum(dim=-1)
    valid = pos_counts > 0
    if not valid.any():
        return features.sum() * 0.0

    mean_log_pos = (pos_mask.float() * log_prob).sum(dim=-1) / pos_counts.clamp(min=1).float()
    return -mean_log_pos[valid].mean()
