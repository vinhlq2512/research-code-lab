"""Unit tests for TaskAgnosticRouter (PART 21.3)."""

from __future__ import annotations

import numpy as np
import torch

from src.dynatree_cre.config import PromptConfig, RoutingConfig, TreeConfig
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.prototype_bank import RelationPrototype, l2_normalize_np
from src.dynatree_cre.router import TaskAgnosticRouter
from src.dynatree_cre.tree import DynamicSemanticTree


def _build_tree_and_prompts(num_rels: int = 4, threshold: float = 0.8):
    tree = DynamicSemanticTree(rep_dim=4)
    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.98, 0.20, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ][:num_rels]
    protos = {}
    for idx, vec in enumerate(vecs, start=1):
        v = l2_normalize_np(np.asarray(vec, dtype=np.float32))
        protos[idx] = RelationPrototype(
            relation_id=idx,
            raw_prototype_vector=v,
            prototype_vector=v,
            sample_count=5,
            created_at_task=0,
            encoder_version="synthetic",
            representation_strategy="entity_pair",
        )
    tree.initialize_from_t1(protos, TreeConfig(threshold_mode="fixed", insertion_threshold=threshold), task_id=0)
    pbank = HierarchicalPromptBank(prompt_length=4, hidden_size=6, config=PromptConfig(length=4))
    pbank.synchronize_with_tree(tree, task_id=0)
    return tree, pbank


def test_hierarchical_probabilities_sum_to_one_and_low_temp_stable() -> None:
    tree, pbank = _build_tree_and_prompts(4, threshold=0.8)
    router = TaskAgnosticRouter(
        RoutingConfig(strategy="hierarchical_soft", top_k=3, cluster_temperature=0.01, relation_temperature=0.01)
    )
    queries = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]], dtype=torch.float32)
    res = router.route(queries, tree, pbank)

    assert torch.allclose(res.cluster_probabilities.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(res.relation_probabilities.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(res.normalized_weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert not torch.isnan(res.composed_prompt).any()
    assert res.composed_prompt.shape == (2, 4, 6)


def test_routing_edge_cases_k_exceeds_single_cluster_and_equal_similarities() -> None:
    # Single cluster + K > R
    tree, pbank = _build_tree_and_prompts(2, threshold=0.5)
    router = TaskAgnosticRouter(RoutingConfig(strategy="hierarchical_soft", top_k=10))
    q_equal = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    res = router.route(q_equal, tree, pbank)
    assert res.selected_relation_ids.shape == (1, 2)
    assert torch.allclose(res.normalized_weights.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(res.relation_probabilities[0, 0], res.relation_probabilities[0, 1], atol=1e-5)


def test_routing_independent_of_true_labels_and_task_ids() -> None:
    tree, pbank = _build_tree_and_prompts(4, threshold=0.8)
    router = TaskAgnosticRouter(RoutingConfig(strategy="hierarchical_soft", top_k=2))
    q = torch.tensor([[0.8, 0.2, 0.0, 0.0]], dtype=torch.float32)
    r1 = router.route(q, tree, pbank, gold_relation_ids=None)
    r2 = router.route(q, tree, pbank, gold_relation_ids=torch.tensor([4]))
    assert torch.equal(r1.selected_relation_ids, r2.selected_relation_ids)
    assert torch.allclose(r1.composed_prompt, r2.composed_prompt)
