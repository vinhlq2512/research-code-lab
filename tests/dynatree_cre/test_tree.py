"""Unit tests for DynamicSemanticTree (PART 21.2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.dynatree_cre.config import TreeConfig
from src.dynatree_cre.prototype_bank import RelationPrototype, l2_normalize_np
from src.dynatree_cre.tree import DynamicSemanticTree


def _make_proto(rel_id: int, vec: list[float], task_id: int = 0) -> RelationPrototype:
    v = l2_normalize_np(np.asarray(vec, dtype=np.float32))
    return RelationPrototype(
        relation_id=rel_id,
        raw_prototype_vector=v,
        prototype_vector=v,
        sample_count=5,
        created_at_task=task_id,
        encoder_version="synthetic",
        representation_strategy="entity_pair",
        relation_name=f"rel_{rel_id}",
    )


def test_tree_single_relation_and_ward_rejection() -> None:
    tree = DynamicSemanticTree(rep_dim=4)
    cfg = TreeConfig(threshold_mode="fixed", insertion_threshold=0.75)
    p1 = _make_proto(1, [1.0, 0.0, 0.0, 0.0], task_id=0)
    tree.initialize_from_t1({1: p1}, config=cfg, task_id=0)
    assert tree.cluster_ids() == ["cluster_000"]
    assert tree.leaf_ids() == ["rel_001"]

    bad_cfg = TreeConfig(linkage="ward", distance_metric="cosine")
    tree2 = DynamicSemanticTree(rep_dim=4)
    with pytest.raises(ValueError, match="Ward"):
        tree2.initialize_from_t1({1: p1}, config=bad_cfg, task_id=0)


def test_tree_clustering_insertion_threshold_tiebreak_and_serialization() -> None:
    tree = DynamicSemanticTree(rep_dim=4)
    cfg = TreeConfig(threshold_mode="fixed", insertion_threshold=0.80)

    # T1: two orthogonal relations -> 2 clusters (cluster_000, cluster_001)
    t1_protos = {
        1: _make_proto(1, [1.0, 0.0, 0.0, 0.0], task_id=0),
        2: _make_proto(2, [0.0, 1.0, 0.0, 0.0], task_id=0),
    }
    tree.initialize_from_t1(t1_protos, config=cfg, task_id=0)
    assert len(tree.cluster_ids()) == 2

    # T2: rel 3 close to rel 1 (>= 0.8), rel 4 orthogonal (< 0.8), rel 5 exact tie -> smallest cluster_000
    t2_protos = {
        3: _make_proto(3, [0.98, 0.19, 0.0, 0.0], task_id=1),
        4: _make_proto(4, [0.0, 0.0, 1.0, 0.0], task_id=1),
    }
    tree.insert_task_relations(t2_protos, task_id=1, config=cfg)
    assert tree.nodes["rel_003"].parent_id == "cluster_000"
    assert tree.nodes["rel_004"].parent_id == "cluster_002"

    # Tie-breaker check on fresh 2-cluster tree
    tie_tree = DynamicSemanticTree(rep_dim=4)
    tie_tree.initialize_from_t1(t1_protos, config=cfg, task_id=0)
    # Exactly equal cosine 0.707 with cluster_000 and cluster_001 -> threshold 0.70 attaches to cluster_000
    tie_cfg = TreeConfig(threshold_mode="fixed", insertion_threshold=0.70)
    tie_tree.effective_threshold = 0.70
    tie_tree.insert_task_relations({10: _make_proto(10, [1.0, 1.0, 0.0, 0.0], task_id=1)}, task_id=1, config=tie_cfg)
    assert tie_tree.nodes["rel_010"].parent_id == "cluster_000"

    # Duplicate relation rejected
    with pytest.raises(ValueError, match="Duplicate"):
        tree.insert_task_relations({3: _make_proto(3, [1.0, 0.0, 0.0, 0.0], task_id=2)}, task_id=2, config=cfg)

    # Round-trip JSON dict & ASCII
    restored = DynamicSemanticTree.from_dict(tree.to_dict())
    assert restored.cluster_ids() == tree.cluster_ids()
    assert restored.leaf_ids() == tree.leaf_ids()
    assert "Root" in restored.to_ascii()
