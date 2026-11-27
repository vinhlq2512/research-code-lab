"""Dynamic Semantic Prompt Tree data structure, incremental clustering, and invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.dynatree_cre.config import TreeConfig
from src.dynatree_cre.prototype_bank import RelationPrototype, l2_normalize_np


@dataclass
class TreeNode:
    node_id: str
    node_type: str  # "root" | "cluster" | "leaf"
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    created_at_task: int = 0
    prototype: np.ndarray | None = None
    relation_id: int | None = None
    relation_name: str = ""
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "depth": int(self.depth),
            "created_at_task": int(self.created_at_task),
            "prototype": self.prototype.tolist() if self.prototype is not None else None,
            "relation_id": int(self.relation_id) if self.relation_id is not None else None,
            "relation_name": self.relation_name,
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        proto = data.get("prototype")
        return cls(
            node_id=str(data["node_id"]),
            node_type=str(data["node_type"]),
            parent_id=data.get("parent_id"),
            children_ids=list(data.get("children_ids", [])),
            depth=int(data["depth"]),
            created_at_task=int(data["created_at_task"]),
            prototype=np.asarray(proto, dtype=np.float32) if proto is not None else None,
            relation_id=int(data["relation_id"]) if data.get("relation_id") is not None else None,
            relation_name=str(data.get("relation_name", "")),
            sample_count=int(data.get("sample_count", 0)),
        )


class DynamicSemanticTree:
    """2-tier dynamic tree (Root -> Semantic Cluster -> Relation Leaf) built without Task IDs."""

    def __init__(self, rep_dim: int, cluster_weighting: str = "equal_relation") -> None:
        self.rep_dim = int(rep_dim)
        self.cluster_weighting = cluster_weighting
        self.nodes: dict[str, TreeNode] = {}
        self.relation_to_node: dict[int, str] = {}
        self.next_cluster_idx: int = 0
        self.effective_threshold: float | None = None
        self.topology_history: list[dict[str, Any]] = []
        self.initialized: bool = False

    def _allocate_cluster_id(self) -> str:
        cid = f"cluster_{self.next_cluster_idx:03d}"
        self.next_cluster_idx += 1
        return cid

    @staticmethod
    def leaf_node_id(relation_id: int) -> str:
        return f"rel_{int(relation_id):03d}"

    def cluster_ids(self) -> list[str]:
        return sorted(nid for nid, n in self.nodes.items() if n.node_type == "cluster")

    def leaf_ids(self) -> list[str]:
        return sorted(nid for nid, n in self.nodes.items() if n.node_type == "leaf")

    def seen_relation_ids(self) -> list[int]:
        return sorted(self.relation_to_node.keys())

    def recompute_cluster_prototype(self, cluster_id: str) -> None:
        cluster = self.nodes[cluster_id]
        if cluster.node_type != "cluster" or not cluster.children_ids:
            raise ValueError(f"Cluster {cluster_id} is invalid or empty.")
        weighted_sum = np.zeros(self.rep_dim, dtype=np.float32)
        for child_id in cluster.children_ids:
            leaf = self.nodes[child_id]
            assert leaf.prototype is not None
            w = float(leaf.sample_count) if self.cluster_weighting == "sample_count" else 1.0
            weighted_sum += w * leaf.prototype
        cluster.prototype = l2_normalize_np(weighted_sum)

    def _create_cluster_with_leaf(
        self, proto: RelationPrototype, task_id: int
    ) -> tuple[str, str]:
        cluster_id = self._allocate_cluster_id()
        leaf_id = self.leaf_node_id(proto.relation_id)

        cluster_node = TreeNode(
            node_id=cluster_id,
            node_type="cluster",
            parent_id="root",
            children_ids=[leaf_id],
            depth=1,
            created_at_task=int(task_id),
            prototype=proto.prototype_vector.copy(),
        )
        leaf_node = TreeNode(
            node_id=leaf_id,
            node_type="leaf",
            parent_id=cluster_id,
            children_ids=[],
            depth=2,
            created_at_task=int(task_id),
            prototype=proto.prototype_vector.copy(),
            relation_id=int(proto.relation_id),
            relation_name=proto.relation_name,
            sample_count=int(proto.sample_count),
        )
        proto.cluster_id = cluster_id

        self.nodes[cluster_id] = cluster_node
        self.nodes[leaf_id] = leaf_node
        self.nodes["root"].children_ids.append(cluster_id)
        self.relation_to_node[int(proto.relation_id)] = leaf_id
        self.recompute_cluster_prototype(cluster_id)
        return cluster_id, leaf_id

    def _add_leaf_to_cluster(
        self, cluster_id: str, proto: RelationPrototype, task_id: int
    ) -> str:
        leaf_id = self.leaf_node_id(proto.relation_id)
        leaf_node = TreeNode(
            node_id=leaf_id,
            node_type="leaf",
            parent_id=cluster_id,
            children_ids=[],
            depth=2,
            created_at_task=int(task_id),
            prototype=proto.prototype_vector.copy(),
            relation_id=int(proto.relation_id),
            relation_name=proto.relation_name,
            sample_count=int(proto.sample_count),
        )
        proto.cluster_id = cluster_id
        self.nodes[leaf_id] = leaf_node
        self.nodes[cluster_id].children_ids.append(leaf_id)
        self.relation_to_node[int(proto.relation_id)] = leaf_id
        self.recompute_cluster_prototype(cluster_id)
        return leaf_id

    def _resolve_threshold(self, protos: list[RelationPrototype], config: TreeConfig) -> float:
        if config.threshold_mode == "fixed" or len(protos) < 2:
            return float(config.insertion_threshold)
        mat = np.stack([p.prototype_vector for p in protos], axis=0)
        sim_mat = np.clip(mat @ mat.T, -1.0, 1.0)
        triu_idx = np.triu_indices(len(protos), k=1)
        pairwise_sims = sim_mat[triu_idx]
        return float(np.quantile(pairwise_sims, float(config.t1_quantile)))

    def initialize_from_t1(
        self,
        prototypes: dict[int, RelationPrototype],
        config: TreeConfig,
        task_id: int = 0,
    ) -> dict[str, Any]:
        """Initialize Root and T1 semantic clusters via average-linkage cosine HAC."""
        if self.initialized:
            raise RuntimeError("Tree is already initialized.")
        if not prototypes:
            raise ValueError("Cannot initialize tree with empty prototypes.")
        if config.distance_metric == "cosine" and config.linkage == "ward":
            raise ValueError("Ward linkage is invalid with cosine distance.")

        sorted_protos = [prototypes[r] for r in sorted(prototypes.keys())]
        for p in sorted_protos:
            if p.prototype_vector.shape != (self.rep_dim,):
                raise ValueError(f"Prototype dim mismatch for relation {p.relation_id}.")

        self.nodes["root"] = TreeNode(
            node_id="root",
            node_type="root",
            parent_id=None,
            children_ids=[],
            depth=0,
            created_at_task=int(task_id),
            prototype=None,
        )

        threshold = self._resolve_threshold(sorted_protos, config)
        self.effective_threshold = threshold

        if len(sorted_protos) == 1:
            cluster_labels = [1]
        else:
            mat = np.stack([p.prototype_vector for p in sorted_protos], axis=0)
            sim_mat = np.clip(mat @ mat.T, -1.0, 1.0)
            dist_mat = np.clip(1.0 - sim_mat, 0.0, 2.0)
            np.fill_diagonal(dist_mat, 0.0)
            condensed = squareform(dist_mat, checks=False)
            z_link = linkage(condensed, method=config.linkage)
            dist_threshold = max(0.0, 1.0 - threshold)
            cluster_labels = fcluster(z_link, t=dist_threshold, criterion="distance").tolist()

        # Group deterministically by order of appearance of sorted relation_id
        groups: dict[int, list[RelationPrototype]] = {}
        for proto, lbl in zip(sorted_protos, cluster_labels):
            groups.setdefault(int(lbl), []).append(proto)

        events: list[dict[str, Any]] = []
        for _, member_protos in groups.items():
            cid, first_lid = self._create_cluster_with_leaf(member_protos[0], task_id=task_id)
            events.append({"action": "create_cluster", "cluster_id": cid, "relation_id": member_protos[0].relation_id})
            for mp in member_protos[1:]:
                self._add_leaf_to_cluster(cid, mp, task_id=task_id)
                events.append({"action": "attach_leaf", "cluster_id": cid, "relation_id": mp.relation_id})

        self.initialized = True
        self.validate_invariants(expected_relation_ids=[p.relation_id for p in sorted_protos])

        record = {
            "task_id": int(task_id),
            "threshold": float(threshold),
            "num_clusters_after": len(self.cluster_ids()),
            "num_leaves_after": len(self.leaf_ids()),
            "events": events,
        }
        self.topology_history.append(record)
        return record

    def insert_task_relations(
        self,
        prototypes: dict[int, RelationPrototype],
        task_id: int,
        config: TreeConfig,
    ) -> dict[str, Any]:
        """Incrementally insert new task relations without modifying old leaf parents."""
        if not self.initialized:
            return self.initialize_from_t1(prototypes, config=config, task_id=task_id)

        threshold = (
            float(self.effective_threshold)
            if self.effective_threshold is not None
            else float(config.insertion_threshold)
        )
        events: list[dict[str, Any]] = []

        for rel_id in sorted(prototypes.keys()):
            if rel_id in self.relation_to_node:
                raise ValueError(f"Duplicate relation_id {rel_id} cannot be re-inserted.")
            proto = prototypes[rel_id]
            if proto.prototype_vector.shape != (self.rep_dim,):
                raise ValueError(f"Prototype dim mismatch for relation {rel_id}.")

            best_cluster: str | None = None
            best_score = -2.0
            for cid in self.cluster_ids():
                c_proto = self.nodes[cid].prototype
                assert c_proto is not None
                score = float(np.dot(proto.prototype_vector, c_proto))
                # Strict > ensures deterministic tie-break by smallest cluster_id
                if score > best_score:
                    best_score = score
                    best_cluster = cid

            if best_cluster is not None and best_score >= threshold:
                self._add_leaf_to_cluster(best_cluster, proto, task_id=task_id)
                events.append(
                    {
                        "action": "insert_existing_cluster",
                        "relation_id": int(rel_id),
                        "cluster_id": best_cluster,
                        "similarity": best_score,
                    }
                )
            else:
                new_cid, _ = self._create_cluster_with_leaf(proto, task_id=task_id)
                events.append(
                    {
                        "action": "create_cluster",
                        "relation_id": int(rel_id),
                        "cluster_id": new_cid,
                        "best_existing_similarity": best_score,
                    }
                )

        self.validate_invariants()
        record = {
            "task_id": int(task_id),
            "threshold": float(threshold),
            "num_clusters_after": len(self.cluster_ids()),
            "num_leaves_after": len(self.leaf_ids()),
            "events": events,
        }
        self.topology_history.append(record)
        return record

    def validate_invariants(self, expected_relation_ids: list[int] | None = None) -> None:
        """Verify all structural tree invariants defined in METHOD_SPEC §2.3."""
        roots = [nid for nid, n in self.nodes.items() if n.node_type == "root"]
        if roots != ["root"]:
            raise AssertionError(f"Expected single root 'root', found {roots}")

        visited: set[str] = set()
        stack = ["root"]
        while stack:
            curr = stack.pop()
            if curr in visited:
                raise AssertionError(f"Cycle detected at node {curr}")
            visited.add(curr)
            for child in self.nodes[curr].children_ids:
                if child not in self.nodes:
                    raise AssertionError(f"Missing child node {child}")
                if self.nodes[child].parent_id != curr:
                    raise AssertionError(f"Parent mismatch for {child}")
                stack.append(child)

        if visited != set(self.nodes.keys()):
            orphans = set(self.nodes.keys()) - visited
            raise AssertionError(f"Orphan nodes detected: {orphans}")

        for cid in self.cluster_ids():
            cnode = self.nodes[cid]
            if not cnode.children_ids:
                raise AssertionError(f"Empty cluster {cid} violates invariant.")
            if cnode.prototype is None or cnode.prototype.shape != (self.rep_dim,):
                raise AssertionError(f"Invalid prototype on cluster {cid}.")

        seen_rels = [self.nodes[lid].relation_id for lid in self.leaf_ids()]
        if len(seen_rels) != len(set(seen_rels)):
            raise AssertionError("Duplicate relation_id found in leaves.")
        if expected_relation_ids is not None and set(expected_relation_ids) != set(seen_rels):
            raise AssertionError("Leaf relations do not match expected seen relations.")

    def compute_metrics(
        self, prev_cluster_prototypes: dict[str, np.ndarray] | None = None
    ) -> dict[str, Any]:
        cids = self.cluster_ids()
        lids = self.leaf_ids()
        sizes = [len(self.nodes[c].children_ids) for c in cids]
        singleton_count = sum(1 for s in sizes if s == 1)

        cohesions: list[float] = []
        for cid in cids:
            c_proto = self.nodes[cid].prototype
            assert c_proto is not None
            for lid in self.nodes[cid].children_ids:
                l_proto = self.nodes[lid].prototype
                assert l_proto is not None
                cohesions.append(float(np.dot(c_proto, l_proto)))

        if len(cids) >= 2:
            c_mat = np.stack([self.nodes[c].prototype for c in cids], axis=0)
            sim = c_mat @ c_mat.T
            triu = sim[np.triu_indices(len(cids), k=1)]
            separation = float(1.0 - np.mean(triu))
        else:
            separation = 0.0

        movements: list[float] = []
        if prev_cluster_prototypes:
            for cid, old_p in prev_cluster_prototypes.items():
                if cid in self.nodes and self.nodes[cid].prototype is not None:
                    movements.append(float(np.linalg.norm(self.nodes[cid].prototype - old_p)))

        return {
            "num_nodes": len(self.nodes),
            "num_clusters": len(cids),
            "num_leaves": len(lids),
            "max_depth": max((n.depth for n in self.nodes.values()), default=0),
            "cluster_size_min": int(min(sizes)) if sizes else 0,
            "cluster_size_mean": float(np.mean(sizes)) if sizes else 0.0,
            "cluster_size_max": int(max(sizes)) if sizes else 0,
            "singleton_cluster_ratio": float(singleton_count / len(cids)) if cids else 0.0,
            "cluster_cohesion": float(np.mean(cohesions)) if cohesions else 0.0,
            "cluster_separation": separation,
            "mean_cluster_prototype_movement": float(np.mean(movements)) if movements else 0.0,
        }

    def to_ascii(self) -> str:
        lines = ["Root"]
        cids = self.cluster_ids()
        for idx, cid in enumerate(cids):
            branch = "└──" if idx == len(cids) - 1 else "├──"
            cnode = self.nodes[cid]
            lines.append(f"{branch} {cid} (task={cnode.created_at_task}, size={len(cnode.children_ids)})")
            prefix = "    " if idx == len(cids) - 1 else "│   "
            for l_idx, lid in enumerate(cnode.children_ids):
                l_branch = "└──" if l_idx == len(cnode.children_ids) - 1 else "├──"
                lnode = self.nodes[lid]
                lines.append(
                    f"{prefix}{l_branch} {lid} [{lnode.relation_name}] (task={lnode.created_at_task})"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rep_dim": self.rep_dim,
            "cluster_weighting": self.cluster_weighting,
            "next_cluster_idx": self.next_cluster_idx,
            "effective_threshold": self.effective_threshold,
            "initialized": self.initialized,
            "nodes": {k: v.to_dict() for k, v in sorted(self.nodes.items())},
            "relation_to_node": {str(k): v for k, v in sorted(self.relation_to_node.items())},
            "topology_history": list(self.topology_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicSemanticTree:
        tree = cls(
            rep_dim=int(data["rep_dim"]),
            cluster_weighting=str(data.get("cluster_weighting", "equal_relation")),
        )
        tree.next_cluster_idx = int(data["next_cluster_idx"])
        tree.effective_threshold = data.get("effective_threshold")
        tree.initialized = bool(data["initialized"])
        tree.nodes = {k: TreeNode.from_dict(v) for k, v in data["nodes"].items()}
        tree.relation_to_node = {int(k): str(v) for k, v in data["relation_to_node"].items()}
        tree.topology_history = list(data.get("topology_history", []))
        if tree.initialized:
            tree.validate_invariants()
        return tree
