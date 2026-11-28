"""Incremental Analytic Classifier & Frozen Random Projection (RanPAC-style) for M7.5."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class IncrementalAnalyticHead:
    """Order-invariant closed-form Ridge classifier with zero old-instance storage."""

    def __init__(
        self,
        input_dim: int,
        projection_mode: str = "random_relu",
        projection_dim: int = 4096,
        projection_seed: int = 2021,
        ridge_lambda: float = 10.0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.projection_mode = projection_mode
        self.projection_seed = int(projection_seed)
        self.ridge_lambda = float(ridge_lambda)

        if projection_mode == "identity":
            self.feat_dim = self.input_dim
            self.w_rand_np: np.ndarray | None = None
        elif projection_mode == "random_relu":
            self.feat_dim = int(projection_dim)
            if self.feat_dim <= 0:
                raise ValueError("projection_dim must be > 0 for random_relu.")
            gen = torch.Generator(device="cpu").manual_seed(self.projection_seed)
            w = torch.randn(self.input_dim, self.feat_dim, generator=gen, dtype=torch.float32) / math.sqrt(
                float(self.feat_dim)
            )
            self.w_rand_np = w.numpy()
        else:
            raise ValueError(f"Unsupported analytic_projection_mode: {projection_mode}")

        self.gram_matrix = np.zeros((self.feat_dim, self.feat_dim), dtype=np.float64)
        self.class_sums: dict[int, np.ndarray] = {}
        self.ordered_rel_ids: list[int] = []
        self._weights_np: np.ndarray | None = None
        self._cached_device: torch.device | None = None
        self._cached_w_t: torch.Tensor | None = None
        self._cached_proj_t: torch.Tensor | None = None

    def _get_proj_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.w_rand_np is None:
            return None
        if self._cached_proj_t is None or self._cached_proj_t.device != device or self._cached_proj_t.dtype != dtype:
            self._cached_proj_t = torch.from_numpy(self.w_rand_np).to(device=device, dtype=dtype)
        return self._cached_proj_t

    def project(self, raw_queries: torch.Tensor) -> torch.Tensor:
        """Map raw queries [B, D] into L2-normalized analytic feature space [B, M]."""
        norm_q = F.normalize(raw_queries, p=2, dim=-1, eps=1e-12)
        if self.projection_mode == "identity":
            return norm_q
        w_t = self._get_proj_tensor(norm_q.device, norm_q.dtype)
        assert w_t is not None
        h = F.relu(torch.mm(norm_q, w_t))
        return F.normalize(h, p=2, dim=-1, eps=1e-12)

    @torch.no_grad()
    def update_task(
        self,
        rel_to_queries: dict[int, torch.Tensor],
        rel_to_desc_vec: dict[int, torch.Tensor] | None = None,
        desc_weight: float = 0.0,
    ) -> None:
        """Accumulate second-order Gram stats and class sums strictly from current-task data."""
        for rel_id in sorted(rel_to_queries.keys()):
            if rel_id in self.class_sums:
                raise ValueError(f"Relation {rel_id} already registered in IncrementalAnalyticHead.")
            q_t = rel_to_queries[rel_id].detach().cpu().float()
            phi = self.project(q_t).numpy().astype(np.float64)  # [N, M]
            self.gram_matrix += phi.T @ phi
            c_vec = phi.sum(axis=0)

            if rel_to_desc_vec is not None and rel_id in rel_to_desc_vec and 0.0 < desc_weight < 1.0:
                d_t = rel_to_desc_vec[rel_id].detach().cpu().float().reshape(1, -1)
                phi_d = self.project(d_t).numpy().astype(np.float64)  # [1, M]
                pseudo_cnt = float(phi.shape[0]) * float(desc_weight) / max(1e-6, 1.0 - float(desc_weight))
                self.gram_matrix += pseudo_cnt * (phi_d.T @ phi_d)
                c_vec = c_vec + pseudo_cnt * phi_d[0]

            self.class_sums[int(rel_id)] = c_vec

        self.ordered_rel_ids = sorted(self.class_sums.keys())
        self.recompute_weights()

    def recompute_weights(self) -> None:
        if not self.ordered_rel_ids:
            self._weights_np = None
            self._cached_w_t = None
            return
        c_mat = np.stack([self.class_sums[r] for r in self.ordered_rel_ids], axis=1)  # [M, R]
        reg = self.gram_matrix + float(self.ridge_lambda) * np.eye(self.feat_dim, dtype=np.float64)
        w = np.linalg.solve(reg, c_mat).astype(np.float32)
        self._weights_np = w
        self._cached_w_t = None

    @torch.no_grad()
    def predict(
        self, raw_queries: torch.Tensor, target_rel_order: list[int] | None = None
    ) -> tuple[torch.Tensor, list[int]]:
        """Return closed-form ridge logits [B, R_seen] with zero future classes."""
        if self._weights_np is None:
            raise RuntimeError("Analytic head has not been populated with any relations.")
        phi = self.project(raw_queries)
        if self._cached_w_t is None or self._cached_device != phi.device:
            self._cached_w_t = torch.from_numpy(self._weights_np).to(device=phi.device, dtype=phi.dtype)
            self._cached_device = phi.device
        logits = torch.mm(phi, self._cached_w_t)
        if target_rel_order is not None and target_rel_order != self.ordered_rel_ids:
            idx_map = {r: idx for idx, r in enumerate(self.ordered_rel_ids)}
            reorder = torch.tensor([idx_map[r] for r in target_rel_order], dtype=torch.long, device=phi.device)
            return logits[:, reorder], list(target_rel_order)
        return logits, list(self.ordered_rel_ids)

    def persistent_bytes(self) -> int:
        total = self.gram_matrix.astype(np.float32).nbytes
        if self.w_rand_np is not None:
            total += self.w_rand_np.nbytes
        total += sum(v.astype(np.float32).nbytes for v in self.class_sums.values())
        if self._weights_np is not None:
            total += self._weights_np.nbytes
        return int(total)

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "projection_mode": self.projection_mode,
            "projection_dim": self.feat_dim,
            "projection_seed": self.projection_seed,
            "ridge_lambda": self.ridge_lambda,
            "w_rand": self.w_rand_np.tolist() if self.w_rand_np is not None else None,
            "gram_matrix": self.gram_matrix.tolist(),
            "class_sums": {str(k): v.tolist() for k, v in sorted(self.class_sums.items())},
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> IncrementalAnalyticHead:
        obj = cls(
            input_dim=int(data["input_dim"]),
            projection_mode=str(data["projection_mode"]),
            projection_dim=int(data["projection_dim"]),
            projection_seed=int(data["projection_seed"]),
            ridge_lambda=float(data["ridge_lambda"]),
        )
        if data.get("w_rand") is not None:
            obj.w_rand_np = np.asarray(data["w_rand"], dtype=np.float32)
        obj.gram_matrix = np.asarray(data["gram_matrix"], dtype=np.float64)
        obj.class_sums = {int(k): np.asarray(v, dtype=np.float64) for k, v in data["class_sums"].items()}
        obj.ordered_rel_ids = sorted(obj.class_sums.keys())
        obj.recompute_weights()
        return obj
