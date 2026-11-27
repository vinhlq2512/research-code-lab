"""End-to-end synthetic integration and checkpoint/resume tests across DT-C0..DT-C4 (PART 21.7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dynatree_cre.benchmark import build_synthetic_task_stream
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.trainer import DynaTreeContinualTrainer


@pytest.mark.parametrize(
    "variant,strategy,use_root,use_cluster",
    [
        ("DT-C0", "none", False, False),
        ("DT-C1", "flat_hard", False, False),
        ("DT-C2", "flat_soft_topk", False, False),
        ("DT-C3", "flat_soft_topk", True, True),
        ("DT-C4", "hierarchical_soft", True, True),
    ],
)
def test_e2e_synthetic_all_variants(
    tmp_path: Path, variant: str, strategy: str, use_root: bool, use_cluster: bool
) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    out_dir = tmp_path / variant
    cfg = DynaTreeConfig.from_dict({
        "method": {"variant": variant},
        "prototype": {"encoder": "synthetic_tiny", "max_seq_length": 32},
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": use_root, "use_cluster_prompts": use_cluster},
        "routing": {"strategy": strategy, "top_k": 2},
        "training": {"epochs_per_task": 2, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": variant, "dataset": "synthetic", "output_dir": str(out_dir)},
    })
    model = DynaTreeCREModel(cfg)
    trainer = DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir)
    metrics = trainer.run()

    assert metrics["num_tasks_executed"] == 3
    assert (out_dir / "resolved_config.yaml").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "performance_matrix.csv").exists()
    assert (out_dir / "routing_metrics.json").exists()
    assert (out_dir / "resource_metrics.json").exists()
    assert (out_dir / "trees" / "task_03.json").exists()
    assert model.tree.seen_relation_ids() == [1, 2, 3, 4, 5, 6]


def test_checkpoint_resume_equivalence_and_no_duplicate_nodes(tmp_path: Path) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"

    cfg_dict = {
        "method": {"variant": "DT-C4"},
        "prototype": {"encoder": "synthetic_tiny", "max_seq_length": 32},
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": True, "use_cluster_prompts": True},
        "routing": {"strategy": "hierarchical_soft", "top_k": 2},
        "training": {"epochs_per_task": 2, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": "DT-C4", "dataset": "synthetic", "output_dir": str(full_dir)},
    }
    cfg_full = DynaTreeConfig.from_dict(cfg_dict)
    m_full = DynaTreeCREModel(cfg_full)
    res_full = DynaTreeContinualTrainer(m_full, cfg_full, stream, output_dir=full_dir).run()

    # Resume from after_T2 checkpoint into resumed_dir
    cfg_dict["experiment"]["output_dir"] = str(resumed_dir)
    cfg_res = DynaTreeConfig.from_dict(cfg_dict)
    m_res = DynaTreeCREModel(cfg_res)
    res_resume = DynaTreeContinualTrainer(m_res, cfg_res, stream, output_dir=resumed_dir).run(
        resume_from=full_dir / "checkpoints" / "after_T2" / "state.pt"
    )

    assert len(m_res.tree.leaf_ids()) == len(m_full.tree.leaf_ids()) == 6
    assert m_res.tree.cluster_ids() == m_full.tree.cluster_ids()
    assert pytest.approx(res_full["final_cosine_whole_accuracy"], abs=1e-6) == res_resume[
        "final_cosine_whole_accuracy"
    ]
    assert json.loads((resumed_dir / "metrics.json").read_text())["num_tasks_executed"] == 3
