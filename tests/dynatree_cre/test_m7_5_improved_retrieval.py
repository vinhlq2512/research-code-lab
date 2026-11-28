"""TDD tests for M7.5 Improved Retrieval Space (RanPAC Random Proj, Analytic Head, Relation Description)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dynatree_cre.analytic import IncrementalAnalyticHead
from src.dynatree_cre.benchmark import build_synthetic_task_stream
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.trainer import DynaTreeContinualTrainer


def test_analytic_head_order_invariance_and_checkpoint_roundtrip() -> None:
    torch.manual_seed(42)
    x_t0 = {
        1: torch.randn(5, 32),
        2: torch.randn(5, 32) + 2.0,
    }
    x_t1 = {
        3: torch.randn(5, 32) - 2.0,
        4: torch.randn(5, 32) + torch.tensor([3.0] + [0.0] * 31),
    }

    h_ab = IncrementalAnalyticHead(
        input_dim=32, projection_mode="random_relu", projection_dim=128, projection_seed=2021, ridge_lambda=5.0
    )
    h_ab.update_task(x_t0)
    h_ab.update_task(x_t1)

    h_ba = IncrementalAnalyticHead(
        input_dim=32, projection_mode="random_relu", projection_dim=128, projection_seed=2021, ridge_lambda=5.0
    )
    h_ba.update_task(x_t1)
    h_ba.update_task(x_t0)

    q = torch.randn(10, 32)
    s_ab, ids_ab = h_ab.predict(q)
    s_ba, ids_ba = h_ba.predict(q)
    assert set(ids_ab) == set(ids_ba) == {1, 2, 3, 4}
    # Align column order before comparing predictions
    reorder = [ids_ba.index(r) for r in ids_ab]
    assert torch.allclose(s_ab, s_ba[:, reorder], atol=1e-5)

    # Checkpoint state_dict roundtrip
    restored = IncrementalAnalyticHead.from_state_dict(h_ab.state_dict())
    s_res, ids_res = restored.predict(q)
    assert ids_res == ids_ab
    assert torch.allclose(s_ab, s_res, atol=1e-6)
    assert restored.persistent_bytes() > 0


def test_analytic_head_beats_raw_cosine_under_cone_anisotropy() -> None:
    rng = np.random.default_rng(7)
    dim = 32
    cone = np.zeros(dim, dtype=np.float32)
    cone[0] = 6.0  # Strong shared anisotropy direction

    by_rel_train = {}
    test_x = []
    test_y = []
    for cls_idx in range(4):
        signal = np.zeros(dim, dtype=np.float32)
        signal[cls_idx + 1] = 1.0
        train_pts = cone + signal + 0.08 * rng.normal(size=(8, dim)).astype(np.float32)
        by_rel_train[cls_idx] = torch.from_numpy(train_pts)
        test_pts = cone + signal + 0.08 * rng.normal(size=(20, dim)).astype(np.float32)
        test_x.append(torch.from_numpy(test_pts))
        test_y.extend([cls_idx] * 20)

    hx = torch.cat(test_x, dim=0)
    head = IncrementalAnalyticHead(
        input_dim=dim, projection_mode="identity", ridge_lambda=0.1
    )
    head.update_task(by_rel_train)
    logits, ord_ids = head.predict(hx)
    preds = [ord_ids[i] for i in torch.argmax(logits, dim=-1).tolist()]
    acc = sum(1 for yt, yp in zip(test_y, preds) if yt == yp) / len(test_y)
    assert acc >= 0.98


@pytest.mark.parametrize(
    "variant,strategy,inf_mode,proj_mode,proj_dim,use_desc",
    [
        ("DT-C0.5", "none", "soft_mix", "random_relu", 128, True),
        ("DT-C2.5", "analytic_retrieval", "candidate_rerank", "random_relu", 128, True),
        ("DT-C4.5", "hybrid_analytic_tree", "candidate_rerank", "random_relu", 128, True),
    ],
)
def test_m7_5_e2e_variants_and_persistence(
    tmp_path: Path,
    variant: str,
    strategy: str,
    inf_mode: str,
    proj_mode: str,
    proj_dim: int,
    use_desc: bool,
) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    out_dir = tmp_path / variant
    use_tree_prompts = variant == "DT-C4.5"
    cfg = DynaTreeConfig.from_dict({
        "method": {"variant": variant},
        "prototype": {
            "encoder": "synthetic_tiny",
            "max_seq_length": 32,
            "similarity_space": "t1_centered_cosine",
            "retrieval_similarity_space": "raw_cosine",
            "use_relation_description": use_desc,
            "description_weight": 0.25,
            "use_analytic_retrieval": True,
            "analytic_projection_mode": proj_mode,
            "analytic_projection_dim": proj_dim,
            "analytic_ridge_lambda": 1.0,
        },
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": use_tree_prompts, "use_cluster_prompts": use_tree_prompts},
        "routing": {
            "strategy": strategy,
            "inference_mode": inf_mode,
            "rerank_fusion_mode": "fusion",
            "hybrid_cluster_gamma": 0.15 if strategy == "hybrid_analytic_tree" else 0.0,
            "top_k": 3,
        },
        "training": {"epochs_per_task": 2, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": variant, "dataset": "synthetic", "output_dir": str(out_dir)},
    })
    model = DynaTreeCREModel(cfg)
    metrics = DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir).run()

    assert metrics["num_tasks_executed"] == 3
    res = json.loads((out_dir / "resource_metrics.json").read_text())
    assert res["memory_bytes"]["persistent_analytic_bytes"] > 0
    if inf_mode == "candidate_rerank":
        rr = metrics["stages"][-1]["eval_stats"]["rerank_diagnostics"]
        assert metrics["final_ncm_whole_accuracy"] == pytest.approx(
            rr["candidate_recall_at_k"] * rr["conditional_rerank_ncm_acc"], abs=1e-7
        )
        assert metrics["stages"][-1]["train_stats"]["train_zero_grad_new_leaf_count"] == 0
