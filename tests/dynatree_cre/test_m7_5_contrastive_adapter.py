"""TDD tests for M7.5 Step 4 Learned Contrastive Query Adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.dynatree_cre.benchmark import build_synthetic_task_stream
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.query_adapter import ResidualQueryAdapter, supervised_contrastive_loss
from src.dynatree_cre.trainer import DynaTreeContinualTrainer, set_deterministic_seed


def test_residual_adapter_near_identity_init_and_supcon_loss() -> None:
    torch.manual_seed(0)
    adapter = ResidualQueryAdapter(rep_dim=32, hidden_dim=16, residual_scale=0.5)
    x = torch.nn.functional.normalize(torch.randn(8, 32), p=2, dim=-1)
    y = adapter(x)
    assert y.shape == x.shape
    cos = (x * y).sum(dim=-1)
    assert float(cos.min().item()) > 0.99

    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(y, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


@pytest.mark.parametrize(
    "variant,adapter_mode,strategy,inf_mode,expected_anchor_policy",
    [
        ("DT-C0.6a", "t1_frozen", "none", "soft_mix", "rehearsal_free_strict_zero_pseudo_replay"),
        ("DT-C0.6b", "continual_anchor_preserved", "none", "soft_mix", "rehearsal_free_raw_prototype_anchors_only"),
        ("DT-C4.6", "continual_anchor_preserved", "hybrid_tree_routing", "candidate_rerank", "rehearsal_free_raw_prototype_anchors_only"),
    ],
)
def test_m7_5_contrastive_adapter_e2e_drift_old_new_and_resume(
    tmp_path: Path,
    variant: str,
    adapter_mode: str,
    strategy: str,
    inf_mode: str,
    expected_anchor_policy: str,
) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    out_dir = tmp_path / variant
    use_tree = variant == "DT-C4.6"
    cfg = DynaTreeConfig.from_dict({
        "method": {"variant": variant},
        "prototype": {
            "encoder": "synthetic_tiny",
            "max_seq_length": 32,
            "similarity_space": "t1_centered_cosine",
            "retrieval_similarity_space": "raw_cosine",
            "use_query_adapter": True,
            "query_adapter_mode": adapter_mode,
            "query_adapter_hidden_dim": 32,
            "query_adapter_epochs": 4,
            "query_adapter_lr": 2e-3,
            "query_adapter_drift_weight": 5.0,
        },
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": use_tree, "use_cluster_prompts": use_tree},
        "routing": {
            "strategy": strategy,
            "inference_mode": inf_mode,
            "hybrid_cluster_gamma": 0.25 if use_tree else 0.0,
            "top_k": 3 if inf_mode == "candidate_rerank" else 1,
        },
        "training": {"epochs_per_task": 2 if use_tree else 0, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": variant, "dataset": "synthetic", "output_dir": str(out_dir)},
    })
    set_deterministic_seed(cfg.experiment.seed)
    model = DynaTreeCREModel(cfg)
    metrics = DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir).run()

    assert metrics["num_tasks_executed"] == 3
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["memory_policy"]["prototype_anchor_policy"] == expected_anchor_policy

    last_eval = metrics["stages"][-1]["eval_stats"]
    ret_ovn = last_eval["retrieval_old_vs_new"]
    assert "retrieval_all_top1_acc" in ret_ovn
    assert "retrieval_old_top1_acc" in ret_ovn
    assert "retrieval_new_top1_acc" in ret_ovn
    assert "representation_drift_l2" in ret_ovn
    if adapter_mode == "t1_frozen":
        assert ret_ovn["representation_drift_l2"] == pytest.approx(0.0, abs=1e-7)

    # Checkpoint resume roundtrip preserves adapter weights identically
    res_dir = tmp_path / f"{variant}_resume"
    cfg_r = DynaTreeConfig.from_dict({**cfg.to_dict(), "experiment": {**cfg.to_dict()["experiment"], "output_dir": str(res_dir)}})
    set_deterministic_seed(cfg_r.experiment.seed)
    m_r = DynaTreeCREModel(cfg_r)
    res_m = DynaTreeContinualTrainer(m_r, cfg_r, stream, output_dir=res_dir).run(
        resume_from=out_dir / "checkpoints" / "after_T2" / "state.pt"
    )
    assert res_m["final_ncm_whole_accuracy"] == pytest.approx(metrics["final_ncm_whole_accuracy"], abs=1e-6)
