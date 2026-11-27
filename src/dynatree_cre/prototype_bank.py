"""Semantic Prototype Bank with T1-anchored centering and zero raw/instance replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize_np(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Prototype vector contains NaN or Inf.")
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError("Cannot L2-normalize a zero-norm prototype vector.")
    return (arr / norm).astype(np.float32)


@dataclass
class RelationPrototype:
    """Persistent relation-level prototype metadata (zero raw samples stored)."""

    relation_id: int
    raw_prototype_vector: np.ndarray
    prototype_vector: np.ndarray
    sample_count: int
    created_at_task: int
    encoder_version: str
    representation_strategy: str
    cluster_id: str | None = None
    relation_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": int(self.relation_id),
            "raw_prototype_vector": self.raw_prototype_vector.tolist(),
            "prototype_vector": self.prototype_vector.tolist(),
            "sample_count": int(self.sample_count),
            "created_at_task": int(self.created_at_task),
            "encoder_version": self.encoder_version,
            "representation_strategy": self.representation_strategy,
            "cluster_id": self.cluster_id,
            "relation_name": self.relation_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationPrototype:
        return cls(
            relation_id=int(data["relation_id"]),
            raw_prototype_vector=np.asarray(data["raw_prototype_vector"], dtype=np.float32),
            prototype_vector=np.asarray(data["prototype_vector"], dtype=np.float32),
            sample_count=int(data["sample_count"]),
            created_at_task=int(data["created_at_task"]),
            encoder_version=str(data["encoder_version"]),
            representation_strategy=str(data["representation_strategy"]),
            cluster_id=data.get("cluster_id"),
            relation_name=str(data.get("relation_name", "")),
        )


class SemanticPrototypeBank:
    """Stores relation prototypes and applies stationary query transformation."""

    def __init__(
        self,
        rep_dim: int,
        encoder_version: str = "bert-base-uncased",
        representation_strategy: str = "entity_pair",
        similarity_space: str = "t1_centered_cosine",
        retrieval_similarity_space: str | None = None,
    ) -> None:
        self.rep_dim = int(rep_dim)
        self.encoder_version = encoder_version
        self.representation_strategy = representation_strategy
        self.similarity_space = similarity_space
        self.retrieval_similarity_space = retrieval_similarity_space
        self.prototypes: dict[int, RelationPrototype] = {}
        self.anchor_mean: np.ndarray | None = None
        self.t1_raw_anchor_mean: np.ndarray | None = None
        self.anchor_frozen: bool = False

    @property
    def effective_retrieval_space(self) -> str:
        return self.retrieval_similarity_space or self.similarity_space

    @staticmethod
    def _apply_given_space_np(
        raw_proto: np.ndarray, space: str, anchor_mean: np.ndarray | None
    ) -> np.ndarray:
        raw_norm = l2_normalize_np(raw_proto)
        if space == "raw_cosine" or anchor_mean is None:
            return raw_norm
        centered = raw_norm - anchor_mean
        if float(np.linalg.norm(centered)) < 1e-12:
            return raw_norm
        return l2_normalize_np(centered)

    def _apply_space_np(self, raw_proto: np.ndarray) -> np.ndarray:
        return self._apply_given_space_np(raw_proto, self.similarity_space, self.anchor_mean)

    def register_task_prototypes(
        self,
        relation_queries: dict[int, torch.Tensor],
        task_id: int,
        relation_names: dict[int, str] | None = None,
    ) -> dict[int, RelationPrototype]:
        """Compute L2-normalized relation prototypes from current-task query vectors."""
        if not relation_queries:
            raise ValueError("relation_queries cannot be empty.")

        sorted_rel_ids = sorted(relation_queries.keys())
        raw_protos: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}

        for rel_id in sorted_rel_ids:
            if rel_id in self.prototypes:
                raise ValueError(f"Duplicate relation_id {rel_id} already registered in PrototypeBank.")
            q_tensor = relation_queries[rel_id].detach().cpu()
            if q_tensor.ndim != 2 or q_tensor.size(1) != self.rep_dim:
                raise ValueError(
                    f"Expected query tensor [N, {self.rep_dim}] for relation {rel_id}, got {tuple(q_tensor.shape)}"
                )
            if q_tensor.size(0) == 0:
                raise ValueError(f"Zero samples provided for relation {rel_id}.")
            if torch.isnan(q_tensor).any() or torch.isinf(q_tensor).any():
                raise ValueError(f"NaN/Inf in query vectors for relation {rel_id}.")

            norm_q = F.normalize(q_tensor, p=2, dim=-1, eps=1e-12)
            mean_vec = norm_q.mean(dim=0).numpy()
            raw_protos[rel_id] = l2_normalize_np(mean_vec)
            counts[rel_id] = int(q_tensor.size(0))

        # Always capture T1 anchor mean once on raw prototypes so dual-space routing works in both directions
        if self.t1_raw_anchor_mean is None:
            stacked_t1 = np.stack([raw_protos[r] for r in sorted_rel_ids], axis=0)
            self.t1_raw_anchor_mean = stacked_t1.mean(axis=0).astype(np.float32)

        # Compute or update centering mean according to similarity_space policy
        if self.similarity_space == "t1_centered_cosine" and not self.anchor_frozen:
            self.anchor_mean = self.t1_raw_anchor_mean.copy()
            self.anchor_frozen = True
        elif self.similarity_space == "incremental_centered_cosine":
            all_raw = [p.raw_prototype_vector for p in self.prototypes.values()] + [
                raw_protos[r] for r in sorted_rel_ids
            ]
            self.anchor_mean = np.stack(all_raw, axis=0).mean(axis=0).astype(np.float32)
            for existing in self.prototypes.values():
                existing.prototype_vector = self._apply_space_np(existing.raw_prototype_vector)

        created: dict[int, RelationPrototype] = {}
        names = relation_names or {}
        for rel_id in sorted_rel_ids:
            proto = RelationPrototype(
                relation_id=rel_id,
                raw_prototype_vector=raw_protos[rel_id],
                prototype_vector=self._apply_space_np(raw_protos[rel_id]),
                sample_count=counts[rel_id],
                created_at_task=int(task_id),
                encoder_version=self.encoder_version,
                representation_strategy=self.representation_strategy,
                cluster_id=None,
                relation_name=names.get(rel_id, f"rel_{rel_id}"),
            )
            self.prototypes[rel_id] = proto
            created[rel_id] = proto

        return created

    @staticmethod
    def _transform_queries_with_space(
        raw_queries: torch.Tensor, space: str, anchor_mean: np.ndarray | None
    ) -> torch.Tensor:
        if torch.isnan(raw_queries).any() or torch.isinf(raw_queries).any():
            raise ValueError("NaN/Inf encountered in raw_queries.")
        norm_q = F.normalize(raw_queries, p=2, dim=-1, eps=1e-12)
        if space == "raw_cosine" or anchor_mean is None:
            return norm_q
        anchor = torch.from_numpy(anchor_mean).to(device=norm_q.device, dtype=norm_q.dtype)
        centered = norm_q - anchor.unsqueeze(0)
        norms = centered.norm(p=2, dim=-1, keepdim=True)
        safe = torch.where(norms > 1e-12, centered, norm_q)
        return F.normalize(safe, p=2, dim=-1, eps=1e-12)

    def _retrieval_anchor_mean(self) -> np.ndarray | None:
        space = self.effective_retrieval_space
        if space == "raw_cosine":
            return None
        if space == self.similarity_space and self.anchor_mean is not None:
            return self.anchor_mean
        if space == "t1_centered_cosine":
            return self.t1_raw_anchor_mean
        all_raw = [p.raw_prototype_vector for p in self.prototypes.values()]
        return np.stack(all_raw, axis=0).mean(axis=0).astype(np.float32) if all_raw else None

    def transform_queries(self, raw_queries: torch.Tensor) -> torch.Tensor:
        """Transform raw L2-normalized query vectors [B, D] into prototype similarity space."""
        return self._transform_queries_with_space(raw_queries, self.similarity_space, self.anchor_mean)

    def transform_queries_for_retrieval(self, raw_queries: torch.Tensor) -> torch.Tensor:
        """Transform raw queries into the configured retrieval_similarity_space."""
        return self._transform_queries_with_space(
            raw_queries, self.effective_retrieval_space, self._retrieval_anchor_mean()
        )

    def get_retrieval_prototypes(
        self,
        tree: Any,
        ordered_rel_ids: list[int],
        ordered_cluster_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return leaf and cluster prototypes in retrieval space without storing old instances."""
        ret_space = self.effective_retrieval_space
        ordered_leaf_ids = [tree.relation_to_node[r] for r in ordered_rel_ids]
        if ret_space == self.similarity_space:
            leaf_mat = np.stack([tree.nodes[lid].prototype for lid in ordered_leaf_ids], axis=0)
            cluster_mat = (
                np.stack([tree.nodes[cid].prototype for cid in ordered_cluster_ids], axis=0)
                if ordered_cluster_ids
                else None
            )
            return leaf_mat, cluster_mat

        ret_anchor = self._retrieval_anchor_mean()
        rel_ret_vecs = {
            r: self._apply_given_space_np(self.prototypes[r].raw_prototype_vector, ret_space, ret_anchor)
            for r in ordered_rel_ids
        }
        leaf_mat = np.stack([rel_ret_vecs[r] for r in ordered_rel_ids], axis=0)
        if not ordered_cluster_ids:
            return leaf_mat, None

        cluster_vecs: list[np.ndarray] = []
        for cid in ordered_cluster_ids:
            member_lids = tree.nodes[cid].children_ids
            member_rels = [int(tree.nodes[lid].relation_id) for lid in member_lids]
            mean_c = np.stack([rel_ret_vecs[r] for r in member_rels], axis=0).mean(axis=0)
            cluster_vecs.append(l2_normalize_np(mean_c))
        return leaf_mat, np.stack(cluster_vecs, axis=0)

    def predict_nearest_prototype(
        self,
        raw_queries: torch.Tensor,
        relation_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Compute cosine similarity logits against seen relation prototypes (DT-C0 control)."""
        ordered_ids = sorted(self.prototypes.keys()) if relation_ids is None else list(relation_ids)
        if not ordered_ids:
            raise ValueError("No relation prototypes registered.")
        ret_space = self.effective_retrieval_space
        ret_anchor = self._retrieval_anchor_mean()
        q_space = self.transform_queries_for_retrieval(raw_queries)
        if ret_space == self.similarity_space:
            mat = np.stack([self.prototypes[r].prototype_vector for r in ordered_ids], axis=0)
        else:
            mat = np.stack(
                [self._apply_given_space_np(self.prototypes[r].raw_prototype_vector, ret_space, ret_anchor) for r in ordered_ids],
                axis=0,
            )
        p_tensor = torch.from_numpy(mat).to(device=q_space.device, dtype=q_space.dtype)
        sims = torch.mm(q_space, p_tensor.t())
        return sims, ordered_ids

    def memory_bytes(self) -> int:
        proto_bytes = sum(
            p.raw_prototype_vector.nbytes + p.prototype_vector.nbytes for p in self.prototypes.values()
        )
        anchor_bytes = self.anchor_mean.nbytes if self.anchor_mean is not None else 0
        return int(proto_bytes + anchor_bytes)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rep_dim": self.rep_dim,
            "encoder_version": self.encoder_version,
            "representation_strategy": self.representation_strategy,
            "similarity_space": self.similarity_space,
            "retrieval_similarity_space": self.retrieval_similarity_space,
            "anchor_mean": self.anchor_mean.tolist() if self.anchor_mean is not None else None,
            "t1_raw_anchor_mean": (
                self.t1_raw_anchor_mean.tolist() if self.t1_raw_anchor_mean is not None else None
            ),
            "anchor_frozen": self.anchor_frozen,
            "prototypes": {str(k): v.to_dict() for k, v in sorted(self.prototypes.items())},
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> SemanticPrototypeBank:
        bank = cls(
            rep_dim=int(data["rep_dim"]),
            encoder_version=str(data["encoder_version"]),
            representation_strategy=str(data["representation_strategy"]),
            similarity_space=str(data["similarity_space"]),
            retrieval_similarity_space=data.get("retrieval_similarity_space"),
        )
        anchor = data.get("anchor_mean")
        t1_anchor = data.get("t1_raw_anchor_mean")
        bank.anchor_mean = np.asarray(anchor, dtype=np.float32) if anchor is not None else None
        bank.t1_raw_anchor_mean = (
            np.asarray(t1_anchor, dtype=np.float32)
            if t1_anchor is not None
            else (bank.anchor_mean.copy() if bank.anchor_mean is not None else None)
        )
        bank.anchor_frozen = bool(data.get("anchor_frozen", False))
        bank.prototypes = {
            int(k): RelationPrototype.from_dict(v) for k, v in data.get("prototypes", {}).items()
        }
        return bank
