"""Expandable Global Cosine Classifier and Prompted NCM Control Head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dynatree_cre.config import ClassifierConfig


class ExpandableCosineClassifier(nn.Module):
    """Class-incremental classifier predicting strictly over seen relations R_<=t."""

    def __init__(self, rep_dim: int, config: ClassifierConfig) -> None:
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.config = config
        self.scale = float(config.scale)

        self.seen_relation_ids: list[int] = []
        self.relation_to_slot: dict[int, int] = {}
        self.current_relation_ids: list[int] = []

        self.register_buffer("old_weights", torch.empty(0, self.rep_dim))
        self.current_weights: nn.Parameter | None = None
        self.register_buffer("ncm_prototypes", torch.empty(0, self.rep_dim))

    @property
    def num_seen_classes(self) -> int:
        return len(self.seen_relation_ids)

    def expand_for_task(
        self,
        new_relation_ids: list[int],
        init_vectors: torch.Tensor | None = None,
    ) -> None:
        """Expand classifier strictly for new relations while freezing prior weights."""
        ordered_new = sorted(int(r) for r in new_relation_ids)
        for r in ordered_new:
            if r in self.relation_to_slot:
                raise ValueError(f"Relation {r} already exists in classifier.")

        device = self.old_weights.device
        dtype = self.old_weights.dtype

        if self.current_weights is not None and self.current_weights.numel() > 0:
            if self.config.freeze_old_weights:
                prev = self.current_weights.detach().to(device=device, dtype=dtype)
                self.old_weights = (
                    torch.cat([self.old_weights, prev], dim=0)
                    if self.old_weights.numel() > 0
                    else prev
                )
            else:
                # If freeze_old_weights=False (ablation), keep previous weights concatenated
                pass

        if init_vectors is not None:
            if init_vectors.shape != (len(ordered_new), self.rep_dim):
                raise ValueError(
                    f"Expected init_vectors {(len(ordered_new), self.rep_dim)}, got {tuple(init_vectors.shape)}"
                )
            w_init = F.normalize(init_vectors.detach().to(device=device, dtype=dtype), p=2, dim=-1, eps=1e-12)
        else:
            w_init = torch.empty(len(ordered_new), self.rep_dim, device=device, dtype=dtype)
            nn.init.normal_(w_init, mean=0.0, std=0.02)
            w_init = F.normalize(w_init, p=2, dim=-1, eps=1e-12)

        if self.current_weights is not None and not self.config.freeze_old_weights:
            combined = torch.cat([self.current_weights.detach(), w_init], dim=0)
            self.current_weights = nn.Parameter(combined)
            self.current_relation_ids = self.current_relation_ids + ordered_new
        else:
            self.current_weights = nn.Parameter(w_init)
            self.current_relation_ids = ordered_new

        start_slot = len(self.seen_relation_ids)
        for offset, rel_id in enumerate(ordered_new):
            self.seen_relation_ids.append(rel_id)
            self.relation_to_slot[rel_id] = start_slot + offset

    def relation_ids_to_slots(self, relation_ids: torch.Tensor) -> torch.Tensor:
        slots = [self.relation_to_slot[int(r.item())] for r in relation_ids]
        return torch.tensor(slots, dtype=torch.long, device=relation_ids.device)

    def _all_weights(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.old_weights.numel() > 0:
            parts.append(self.old_weights)
        if self.current_weights is not None and self.current_weights.numel() > 0:
            parts.append(self.current_weights)
        if not parts:
            raise RuntimeError("Classifier has not been expanded with any relations yet.")
        return torch.cat(parts, dim=0)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """Return scaled cosine logits [B, |R_<=t|] with zero future-class logits."""
        norm_z = F.normalize(representations, p=2, dim=-1, eps=1e-12)
        norm_w = F.normalize(self._all_weights(), p=2, dim=-1, eps=1e-12)
        return self.scale * torch.mm(norm_z, norm_w.t())

    @torch.no_grad()
    def update_ncm_prototypes(self, rel_to_features: dict[int, torch.Tensor]) -> None:
        """Append prompted NCM class prototypes for newly learned task relations."""
        device = self.old_weights.device
        dtype = self.old_weights.dtype
        new_vecs: list[torch.Tensor] = []
        for rel_id in self.current_relation_ids:
            feats = rel_to_features[rel_id].detach().to(device=device, dtype=dtype)
            mean_f = F.normalize(feats, p=2, dim=-1, eps=1e-12).mean(dim=0)
            new_vecs.append(F.normalize(mean_f, p=2, dim=-1, eps=1e-12))

        if new_vecs:
            block = torch.stack(new_vecs, dim=0)
            self.ncm_prototypes = (
                torch.cat([self.ncm_prototypes, block], dim=0)
                if self.ncm_prototypes.numel() > 0
                else block
            )

    @torch.no_grad()
    def predict_ncm(self, representations: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity logits against frozen prompted NCM prototypes."""
        if self.ncm_prototypes.numel() == 0:
            raise RuntimeError("NCM prototypes have not been populated.")
        norm_z = F.normalize(representations, p=2, dim=-1, eps=1e-12)
        norm_p = F.normalize(self.ncm_prototypes, p=2, dim=-1, eps=1e-12)
        return torch.mm(norm_z, norm_p.t())

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.current_weights is not None and self.current_weights.requires_grad:
            return [self.current_weights]
        return []

    def snapshot_frozen_weights(self) -> dict[str, torch.Tensor]:
        snaps: dict[str, torch.Tensor] = {}
        if self.old_weights.numel() > 0:
            snaps["old_weights"] = self.old_weights.detach().clone().cpu()
        if self.ncm_prototypes.numel() > 0:
            snaps["ncm_prototypes"] = self.ncm_prototypes.detach().clone().cpu()
        return snaps

    def persistent_bytes(self) -> int:
        total = self.old_weights.numel() * self.old_weights.element_size()
        if self.current_weights is not None:
            total += self.current_weights.numel() * self.current_weights.element_size()
        total += self.ncm_prototypes.numel() * self.ncm_prototypes.element_size()
        return int(total)

    def metadata_state(self) -> dict[str, Any]:
        return {
            "seen_relation_ids": list(self.seen_relation_ids),
            "current_relation_ids": list(self.current_relation_ids),
        }

    def restore_metadata_and_state(
        self, meta: dict[str, Any], state_dict: dict[str, Any]
    ) -> None:
        self.seen_relation_ids = [int(r) for r in meta["seen_relation_ids"]]
        self.relation_to_slot = {r: idx for idx, r in enumerate(self.seen_relation_ids)}
        self.current_relation_ids = [int(r) for r in meta["current_relation_ids"]]

        old_w = state_dict["old_weights"]
        self.old_weights = torch.empty_like(old_w)
        if "current_weights" in state_dict:
            self.current_weights = nn.Parameter(torch.empty_like(state_dict["current_weights"]))
        ncm_p = state_dict["ncm_prototypes"]
        self.ncm_prototypes = torch.empty_like(ncm_p)
        self.load_state_dict(state_dict, strict=True)
