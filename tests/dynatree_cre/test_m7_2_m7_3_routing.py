"""TDD tests for M7.2 Dual-Space Routing, M7.3 Cluster-Size-Aware & Hybrid Routing, and M7.1 auto-logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dynatree_cre.benchmark import build_synthetic_task_stream
from src.dynatree_cre.config import DynaTreeConfig, PromptConfig, RoutingConfig
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.prototype_bank import SemanticPrototypeBank
from src.dynatree_cre.router import TaskAgnosticRouter
from src.dynatree_cre.trainer import DynaTreeContinualTrainer
from src.dynatree_cre.tree import DynamicSemanticTree, TreeNode


def _make_unbalanced_tree(sizes: list[int], dim: int = 16, equal_direction: bool = True) -> tuple[DynamicSemanticTree, np.ndarray]:
    rng = np.random.default_rng(123)
    base_dir = np.zeros(dim, dtype=np.float32)
    base_dir[0] = 1.0

    tree = DynamicSemanticTree(rep_dim=dim)
    tree.initialized = True
    cids = [f"cluster_{i:03d}" for i in range(len(sizes))]
    tree.nodes["root"] = TreeNode(
        node_id="root", node_type="root", depth=0, parent_id=None, children_ids=cids
    )

    rel_id = 0
    for c_idx, (cid, sz) in enumerate(zip(cids, sizes)):
        if equal_direction:
            c_proto = base_dir.copy()
        else:
            raw = rng.normal(size=dim).astype(np.float32)
            c_proto = raw / max(1e-12, float(np.linalg.norm(raw)))
        leaf_ids: list[str] = []
        for _ in range(sz):
            lid = f"leaf_rel_{rel_id:03d}"
            leaf_ids.append(lid)
            if equal_direction:
                l_proto = base_dir.copy()
            else:
                raw_l = c_proto + 0.15 * rng.normal(size=dim).astype(np.float32)
                l_proto = raw_l / max(1e-12, float(np.linalg.norm(raw_l)))
            tree.nodes[lid] = TreeNode(
                node_id=lid,
                node_type="leaf",
                depth=2,
                parent_id=cid,
                relation_id=rel_id,
                prototype=l_proto,
            )
            tree.relation_to_node[rel_id] = lid
            rel_id += 1

        tree.nodes[cid] = TreeNode(
            node_id=cid,
            node_type="cluster",
            depth=1,
            parent_id="root",
            children_ids=leaf_ids,
            prototype=c_proto,
        )
    return tree, base_dir


def test_hybrid_gamma_zero_equals_flat_soft_topk() -> None:
    tree, _ = _make_unbalanced_tree([5, 3, 2], dim=16, equal_direction=False)
    pb = HierarchicalPromptBank(4, 32, PromptConfig(length=4, use_root_prompt=True, use_cluster_prompts=True))
    pb.synchronize_with_tree(tree, task_id=0)

    q = torch.randn(6, 16)
    r_flat = TaskAgnosticRouter(RoutingConfig(strategy="flat_soft_topk", top_k=3, relation_temperature=0.2))
    r_hyb = TaskAgnosticRouter(RoutingConfig(strategy="hybrid_tree_routing", top_k=3, relation_temperature=0.2, hybrid_cluster_gamma=0.0))

    res_flat = r_flat.route(q, tree, pb)
    res_hyb = r_hyb.route(q, tree, pb)
    assert torch.allclose(res_flat.relation_probabilities, res_hyb.relation_probabilities, atol=1e-6)
    assert torch.equal(res_flat.selected_relation_ids, res_hyb.selected_relation_ids)
    assert torch.allclose(res_flat.composed_prompt, res_hyb.composed_prompt, atol=1e-6)


def test_size_corrected_eta_zero_equals_hierarchical_soft() -> None:
    tree, _ = _make_unbalanced_tree([6, 4, 1], dim=16, equal_direction=False)
    pb = HierarchicalPromptBank(4, 32, PromptConfig(length=4, use_root_prompt=True, use_cluster_prompts=True))
    pb.synchronize_with_tree(tree, task_id=0)

    q = torch.randn(6, 16)
    r_hier = TaskAgnosticRouter(RoutingConfig(strategy="hierarchical_soft", top_k=3))
    r_sc = TaskAgnosticRouter(RoutingConfig(strategy="size_corrected_hierarchical", top_k=3, cluster_size_eta=0.0))

    res_hier = r_hier.route(q, tree, pb)
    res_sc = r_sc.route(q, tree, pb)
    assert torch.allclose(res_hier.relation_probabilities, res_sc.relation_probabilities, atol=1e-6)
    assert torch.equal(res_hier.selected_relation_ids, res_sc.selected_relation_ids)


def test_size_corrected_eta_one_neutralizes_equal_score_bias() -> None:
    tree, q_dir = _make_unbalanced_tree([10, 2], dim=16, equal_direction=True)
    pb = HierarchicalPromptBank(4, 32, PromptConfig(length=4, use_root_prompt=True, use_cluster_prompts=True))
    pb.synchronize_with_tree(tree, task_id=0)

    q = torch.from_numpy(q_dir).unsqueeze(0)
    r_sc = TaskAgnosticRouter(RoutingConfig(strategy="size_corrected_hierarchical", top_k=3, cluster_size_eta=1.0))
    res = r_sc.route(q, tree, pb)
    probs = res.relation_probabilities.squeeze(0).detach().cpu().numpy()

    pos_large = res.ordered_relation_ids.index(0)
    pos_small = res.ordered_relation_ids.index(10)
    assert probs[pos_large] == pytest.approx(probs[pos_small], rel=1e-5)


def test_unbalanced_singleton_global_retrieval_stability_and_no_gold_label() -> None:
    tree, _ = _make_unbalanced_tree([10, 1, 4], dim=16, equal_direction=False)
    pb = HierarchicalPromptBank(4, 32, PromptConfig(length=4, use_root_prompt=True, use_cluster_prompts=True))
    pb.synchronize_with_tree(tree, task_id=0)

    # Include near-zero norm and extreme values to verify numerical stability
    q = torch.tensor([[0.0] * 16, [1e4] + [0.0] * 15, [-1e-8] * 16], dtype=torch.float32)
    for strat, kw in [
        ("size_corrected_hierarchical", {"cluster_size_eta": 1.0, "relation_temperature": 0.05}),
        ("hybrid_tree_routing", {"hybrid_cluster_gamma": 0.5, "relation_temperature": 0.05}),
    ]:
        router = TaskAgnosticRouter(RoutingConfig(strategy=strat, top_k=5, **kw))
        res = router.route(q, tree, pb, gold_relation_ids=None)
        assert not torch.isnan(res.relation_probabilities).any()
        assert not torch.isinf(res.relation_probabilities).any()
        assert res.selected_relation_ids.shape == (3, 5)
        assert torch.allclose(res.relation_probabilities.sum(dim=-1), torch.ones(3), atol=1e-5)

    # Verify global retrieval spans across multiple clusters (not restricted to single cluster)
    # Give cluster_001 (singleton, rel 10) and cluster_002 (rel 11..14) close prototypes to query
    p_mix = 0.7 * tree.nodes["leaf_rel_010"].prototype + 0.7 * tree.nodes["leaf_rel_011"].prototype
    q_mix = torch.from_numpy(p_mix).unsqueeze(0)
    hyb = TaskAgnosticRouter(RoutingConfig(strategy="hybrid_tree_routing", top_k=3, hybrid_cluster_gamma=0.25))
    res_mix = hyb.route(q_mix, tree, pb, gold_relation_ids=None)
    top3_cids = {tree.nodes[tree.relation_to_node[r]].parent_id for r in res_mix.selected_relation_ids[0].tolist()}
    assert len(top3_cids) >= 2


def test_dual_space_bank_raw_and_centered_retrieval_modes() -> None:
    bank_c = SemanticPrototypeBank(rep_dim=8, similarity_space="t1_centered_cosine", retrieval_similarity_space="raw_cosine")
    q_t0 = {0: torch.tensor([[1.0, 0.2] + [0.0] * 6]), 1: torch.tensor([[0.2, 1.0] + [0.0] * 6])}
    bank_c.register_task_prototypes(q_t0, task_id=0)
    raw_in = torch.tensor([[1.0, 0.0] + [0.0] * 6])
    # Retrieval transform must preserve raw direction rather than centering
    assert torch.allclose(bank_c.transform_queries_for_retrieval(raw_in), raw_in, atol=1e-6)

    bank_r = SemanticPrototypeBank(rep_dim=8, similarity_space="raw_cosine", retrieval_similarity_space="t1_centered_cosine")
    bank_r.register_task_prototypes(q_t0, task_id=0)
    assert bank_r.t1_raw_anchor_mean is not None
    # Centered retrieval query differs from raw when anchor_mean is non-zero
    assert not torch.allclose(bank_r.transform_queries_for_retrieval(raw_in), raw_in, atol=1e-3)


def test_dual_space_and_hybrid_e2e_and_m7_1_autolog(tmp_path: Path) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    for idx, (sim_sp, ret_sp, r_strat, extra_r) in enumerate([
        ("t1_centered_cosine", "raw_cosine", "hybrid_tree_routing", {"hybrid_cluster_gamma": 0.5}),
        ("raw_cosine", "t1_centered_cosine", "size_corrected_hierarchical", {"cluster_size_eta": 1.0}),
    ]):
        out_dir = tmp_path / f"run_{idx}"
        cfg = DynaTreeConfig.from_dict({
            "method": {"variant": "DT-C4"},
            "prototype": {
                "encoder": "synthetic_tiny",
                "max_seq_length": 32,
                "similarity_space": sim_sp,
                "retrieval_similarity_space": ret_sp,
            },
            "tree": {"threshold_mode": "t1_quantile" if sim_sp == "t1_centered_cosine" else "fixed", "t1_quantile": 0.50, "insertion_threshold": 0.75},
            "prompt": {"length": 4, "use_root_prompt": True, "use_cluster_prompts": True},
            "routing": {"strategy": r_strat, "top_k": 2, **extra_r},
            "training": {"epochs_per_task": 2, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
            "experiment": {"experiment_id": f"M7-TEST-{idx}", "dataset": "synthetic", "output_dir": str(out_dir)},
        })
        model = DynaTreeCREModel(cfg)
        metrics = DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir).run()

        last_stage = metrics["stages"][-1]
        assert "train_zero_grad_new_leaf_count" in last_stage["train_stats"]
        decomp = last_stage["eval_stats"]["m7_1_decomposition"]
        assert decomp["num_samples"] == 24
        rk = decomp["recall_at_k"]
        assert rk["R@1"] <= rk["R@3"] <= rk["R@5"] <= rk["R@10"]
        rm = json.loads((out_dir / "routing_metrics.json").read_text())
        assert "m7_1_decomposition" in rm["stages"][-1]
