"""Research invariant, data-leakage, zero-replay, and reproducibility tests (PART 21.6)."""

from __future__ import annotations

import pytest
import torch

from src.dynatree_cre.benchmark import build_synthetic_task_stream
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.trainer import DynaTreeContinualTrainer, set_deterministic_seed


def test_config_rejects_replay_and_test_time_task_id() -> None:
    with pytest.raises(ValueError, match="exemplar_budget"):
        DynaTreeConfig.from_dict({"memory": {"exemplar_budget": 1}})
    with pytest.raises(ValueError, match="Feature replay"):
        DynaTreeConfig.from_dict({"memory": {"feature_replay": True}})
    with pytest.raises(ValueError, match="generative replay"):
        DynaTreeConfig.from_dict({"memory": {"generative_replay": True}})
    with pytest.raises(ValueError, match="Task ID"):
        DynaTreeConfig.from_dict({"routing": {"use_task_id": True}})
    with pytest.raises(ValueError, match="Gold relation"):
        DynaTreeConfig.from_dict({"routing": {"use_gold_relation": True}})


def test_frozen_encoder_and_zero_future_access_invariants(tmp_path) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=3, test_shot=3)
    cfg = DynaTreeConfig.from_dict({
        "prototype": {"encoder": "synthetic_tiny", "max_seq_length": 32},
        "prompt": {"length": 4},
        "training": {"epochs_per_task": 2, "batch_size": 4, "device": "cpu"},
        "experiment": {"output_dir": str(tmp_path / "run")},
    })
    set_deterministic_seed(2021)
    model = DynaTreeCREModel(cfg)
    enc_before = {k: v.detach().clone() for k, v in model.query_encoder.named_parameters()}

    trainer = DynaTreeContinualTrainer(model, cfg, stream, output_dir=tmp_path / "run")

    # Run only T1 (task_id=0) manually to inspect zero future leakage
    task0 = stream.tasks[0]
    trainer._phase_a_validate_task(task0)
    protos0 = trainer._phase_b_build_prototypes(task0)
    model.tree.insert_task_relations(protos0, task_id=0, config=cfg.tree)
    trainer._phase_d_prepare_model(task0)
    trainer._phase_e_train_task(task0)

    # After T1: only T1 relations (1, 2) exist in tree, prototype bank, and classifier
    assert model.tree.seen_relation_ids() == [1, 2]
    assert sorted(model.prototype_bank.prototypes.keys()) == [1, 2]
    assert model.classifier.num_seen_classes == 2

    # Frozen encoder remains bit-exact unchanged and in eval mode
    assert not model.query_encoder.encoder.training
    for k, snap in enc_before.items():
        curr = dict(model.query_encoder.named_parameters())[k].detach()
        assert torch.equal(snap, curr)

    # Checkpoint contains zero raw sample strings from training
    payload_str = str(model.export_state())
    for s in task0.train_samples:
        assert s.marked_text not in payload_str
