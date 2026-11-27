"""Tests for M7.0 checkpoint equality and M7.1 routing error decomposition."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dynatree_cre.benchmark import build_synthetic_task_stream
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.diagnostic import (
    compare_checkpoints_exact,
    evaluate_checkpoint_diagnostic,
)
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.trainer import DynaTreeContinualTrainer, set_deterministic_seed


def test_checkpoint_exact_tensor_and_prediction_equality(tmp_path: Path) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    full_dir = tmp_path / "full"
    res_dir = tmp_path / "resumed"

    cfg_dict = {
        "method": {"variant": "DT-C4"},
        "prototype": {"encoder": "synthetic_tiny", "max_seq_length": 32},
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": True, "use_cluster_prompts": True},
        "routing": {"strategy": "hierarchical_soft", "top_k": 2},
        "training": {"epochs_per_task": 2, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": "DT-C4", "dataset": "synthetic", "output_dir": str(full_dir)},
    }
    cfg_a = DynaTreeConfig.from_dict(cfg_dict)
    set_deterministic_seed(cfg_a.experiment.seed)
    m_a = DynaTreeCREModel(cfg_a)
    DynaTreeContinualTrainer(m_a, cfg_a, stream, output_dir=full_dir).run()

    cfg_dict["experiment"]["output_dir"] = str(res_dir)
    cfg_b = DynaTreeConfig.from_dict(cfg_dict)
    set_deterministic_seed(cfg_b.experiment.seed)
    m_b = DynaTreeCREModel(cfg_b)
    DynaTreeContinualTrainer(m_b, cfg_b, stream, output_dir=res_dir).run(
        resume_from=full_dir / "checkpoints" / "after_T2" / "state.pt"
    )

    cmp_res = compare_checkpoints_exact(
        full_dir / "checkpoints" / "after_T3" / "state.pt",
        res_dir / "checkpoints" / "after_T3" / "state.pt",
    )
    assert cmp_res["all_tensors_exact_equal"] is True
    assert cmp_res["mismatched_tensors"] == []

    sum_a, rec_a = evaluate_checkpoint_diagnostic(m_a, stream, ks=(1, 3, 5))
    sum_b, rec_b = evaluate_checkpoint_diagnostic(m_b, stream, ks=(1, 3, 5))
    assert [r["pred_ncm"] for r in rec_a] == [r["pred_ncm"] for r in rec_b]
    assert [r["pred_cos"] for r in rec_a] == [r["pred_cos"] for r in rec_b]
    assert sum_a["overall_ncm_acc"] == pytest.approx(sum_b["overall_ncm_acc"])


def test_m7_1_routing_decomposition_and_zero_retrain_modes(tmp_path: Path) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    out_dir = tmp_path / "c4"
    cfg = DynaTreeConfig.from_dict({
        "method": {"variant": "DT-C4"},
        "prototype": {"encoder": "synthetic_tiny", "max_seq_length": 32},
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": True, "use_cluster_prompts": True},
        "routing": {"strategy": "hierarchical_soft", "top_k": 2},
        "training": {"epochs_per_task": 2, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": "DT-C4", "dataset": "synthetic", "output_dir": str(out_dir)},
    })
    model = DynaTreeCREModel(cfg)
    DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir).run()

    summary, _ = evaluate_checkpoint_diagnostic(model, stream, ks=(1, 3, 5, 10))
    m2x2 = summary["matrix_2x2_ncm"]
    total = (
        m2x2["routing_correct_pred_correct"]
        + m2x2["routing_correct_pred_wrong"]
        + m2x2["routing_wrong_pred_correct"]
        + m2x2["routing_wrong_pred_wrong"]
    )
    assert total == summary["num_samples"] == 24
    rk = summary["recall_at_k"]
    assert rk["R@1"] <= rk["R@3"] <= rk["R@5"] <= rk["R@10"]
    assert set(summary["per_task"].keys()) == {"T1", "T2", "T3"}

    # Zero-retrain diagnostic modes run cleanly without mutating model
    raw_sum, _ = evaluate_checkpoint_diagnostic(model, stream, use_raw_retrieval=True)
    eta_sum, _ = evaluate_checkpoint_diagnostic(
        model, stream, routing_override="size_corrected_hierarchical", eta=1.0
    )
    hyb_sum, _ = evaluate_checkpoint_diagnostic(
        model, stream, routing_override="hybrid_tree_routing", gamma=0.5
    )
    assert raw_sum["num_samples"] == 24
    assert eta_sum["num_samples"] == 24
    assert hyb_sum["num_samples"] == 24
