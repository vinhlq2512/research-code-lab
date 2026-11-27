"""Unit tests for HierarchicalPromptBank, residual composition, and freeze policy (PART 21.4)."""

from __future__ import annotations

import numpy as np
import torch

from src.dynatree_cre.config import PromptConfig, RoutingConfig, TreeConfig
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.prototype_bank import RelationPrototype, l2_normalize_np
from src.dynatree_cre.router import TaskAgnosticRouter
from src.dynatree_cre.tree import DynamicSemanticTree


def _proto(rel_id: int, vec: list[float], task_id: int) -> RelationPrototype:
    v = l2_normalize_np(np.asarray(vec, dtype=np.float32))
    return RelationPrototype(
        relation_id=rel_id,
        raw_prototype_vector=v,
        prototype_vector=v,
        sample_count=5,
        created_at_task=task_id,
        encoder_version="synthetic",
        representation_strategy="entity_pair",
    )


def test_residual_composition_counts_root_once_and_preserves_old_prompts() -> None:
    tree = DynamicSemanticTree(rep_dim=4)
    tree_cfg = TreeConfig(threshold_mode="fixed", insertion_threshold=0.8)
    tree.initialize_from_t1(
        {1: _proto(1, [1.0, 0.0, 0.0, 0.0], 0), 2: _proto(2, [0.0, 1.0, 0.0, 0.0], 0)},
        config=tree_cfg,
        task_id=0,
    )

    pbank = HierarchicalPromptBank(prompt_length=8, hidden_size=16, config=PromptConfig(length=8))
    pbank.synchronize_with_tree(tree, task_id=0)
    pbank.apply_freeze_policy(task_id=0)

    # Verify exact residual formula: P_root + P_cluster + Delta P_r
    router = TaskAgnosticRouter(RoutingConfig(strategy="hierarchical_soft", top_k=1))
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    res = router.route(q1, tree, pbank)
    expected = pbank.root_prompt + pbank.cluster_prompts["cluster_000"] + pbank.leaf_prompts["rel_001"]
    assert torch.allclose(res.composed_prompt[0], expected, atol=1e-6)

    # Simulate T2: add new relation 3 into new cluster_002
    tree.insert_task_relations({3: _proto(3, [0.0, 0.0, 1.0, 0.0], 1)}, task_id=1, config=tree_cfg)
    pbank.synchronize_with_tree(tree, task_id=1)
    pbank.apply_freeze_policy(task_id=1)

    trainable = pbank.trainable_parameters()
    assert len(trainable) == 2  # cluster_002 and rel_003 only
    assert not pbank.root_prompt.requires_grad
    assert not pbank.cluster_prompts["cluster_000"].requires_grad
    assert not pbank.leaf_prompts["rel_001"].requires_grad

    frozen_snap = pbank.snapshot_frozen_tensors()
    old_eff_before = pbank.get_effective_prompt(1, tree).detach().clone()

    # Step optimizer with strong weight_decay=0.1 and verify old prompts stay bit-exact!
    opt = torch.optim.AdamW(trainable, lr=0.05, weight_decay=0.1)
    q3 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    res3 = TaskAgnosticRouter(RoutingConfig(strategy="hierarchical_soft", top_k=2)).route(q3, tree, pbank)
    loss = (res3.composed_prompt ** 2).sum()
    loss.backward()
    assert pbank.leaf_prompts["rel_003"].grad is not None
    assert pbank.leaf_prompts["rel_001"].grad is None
    opt.step()

    for name, snap in frozen_snap.items():
        curr = dict(pbank.named_parameters())[name].detach().cpu()
        assert torch.equal(snap, curr), f"Frozen prompt {name} drifted!"
    assert torch.equal(old_eff_before, pbank.get_effective_prompt(1, tree).detach())


def test_flat_prompt_bank_without_root_inherits_target_device() -> None:
    tree = DynamicSemanticTree(rep_dim=4)
    tree_cfg = TreeConfig(threshold_mode="fixed", insertion_threshold=0.8)
    tree.initialize_from_t1(
        {1: _proto(1, [1.0, 0.0, 0.0, 0.0], 0)},
        config=tree_cfg,
        task_id=0,
    )
    target_dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pbank = HierarchicalPromptBank(
        prompt_length=4,
        hidden_size=8,
        config=PromptConfig(length=4, use_root_prompt=False, use_cluster_prompts=False),
    ).to(target_dev)
    word_emb = torch.randn(10, 8, device=target_dev)
    pbank.synchronize_with_tree(tree, task_id=0, word_embedding_weight=word_emb)
    assert pbank.leaf_prompts["rel_001"].device.type == target_dev.type
