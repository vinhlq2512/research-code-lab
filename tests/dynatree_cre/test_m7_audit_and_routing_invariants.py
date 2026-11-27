"""M7.0 & M7.1 audit and routing invariant tests (pre-architecture-refinement)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dynatree_cre.config import DynaTreeConfig, PromptConfig, RoutingConfig
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.router import TaskAgnosticRouter
from src.dynatree_cre.tree import DynamicSemanticTree, TreeNode

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tacred_dt_c0_config_targets_tacred_40_plus_relations() -> None:
    cfg_path = REPO_ROOT / "configs/dynatree_cre/dt_c0_prototype_control_tacred_5shot_seed2021.yaml"
    cfg = DynaTreeConfig.from_yaml(cfg_path)
    assert cfg.experiment.dataset == "tacred"
    assert cfg.experiment.base_dir == "data/protocol_v1/tacred"
    assert cfg.method.variant == "DT-C0"
    assert cfg.training.device == "cuda"
    assert cfg.training.epochs_per_task == 0


def test_hierarchical_soft_exhibits_equal_score_cluster_size_bias() -> None:
    """Verify M7.3 mathematical finding: equal leaf/cluster sims penalize larger clusters."""
    dim = 16
    q_dir = np.zeros(dim, dtype=np.float32)
    q_dir[0] = 1.0

    # Build a tree with 2 clusters: cluster_000 (size 10) vs cluster_001 (size 2)
    # Give every leaf and both clusters the EXACT same direction as query.
    tree = DynamicSemanticTree(rep_dim=dim)
    tree.initialized = True
    tree.nodes["root"] = TreeNode(
        node_id="root", node_type="root", depth=0, parent_id=None, children_ids=["cluster_000", "cluster_001"]
    )

    tree.nodes["cluster_000"] = TreeNode(
        node_id="cluster_000",
        node_type="cluster",
        depth=1,
        parent_id="root",
        children_ids=[f"leaf_rel_{r:03d}" for r in range(10)],
        prototype=q_dir.copy(),
    )
    tree.nodes["cluster_001"] = TreeNode(
        node_id="cluster_001",
        node_type="cluster",
        depth=1,
        parent_id="root",
        children_ids=[f"leaf_rel_{r:03d}" for r in range(10, 12)],
        prototype=q_dir.copy(),
    )
    for r in range(12):
        cid = "cluster_000" if r < 10 else "cluster_001"
        lid = f"leaf_rel_{r:03d}"
        tree.nodes[lid] = TreeNode(
            node_id=lid,
            node_type="leaf",
            depth=2,
            parent_id=cid,
            relation_id=r,
            prototype=q_dir.copy(),
        )
        tree.relation_to_node[r] = lid

    from src.dynatree_cre.config import PromptConfig

    prompt_bank = HierarchicalPromptBank(
        prompt_length=4,
        hidden_size=32,
        config=PromptConfig(length=4, use_root_prompt=True, use_cluster_prompts=True),
    )
    prompt_bank.synchronize_with_tree(tree, task_id=0)

    router = TaskAgnosticRouter(RoutingConfig(strategy="hierarchical_soft", top_k=3, cluster_temperature=0.2, relation_temperature=0.2))
    q_tensor = torch.from_numpy(q_dir).unsqueeze(0)
    res = router.route(q_tensor, tree, prompt_bank)

    probs = res.relation_probabilities.squeeze(0).detach().cpu().numpy()
    pos_large = res.ordered_relation_ids.index(0)
    pos_small = res.ordered_relation_ids.index(10)
    # In cluster_000 (size 10), each leaf gets 0.5 / 10 = 0.05; in cluster_001 (size 2), each gets 0.5 / 2 = 0.25 (5x ratio)
    assert probs[pos_small] == pytest.approx(5.0 * probs[pos_large], rel=1e-4)
