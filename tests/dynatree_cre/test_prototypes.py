"""Unit tests for SemanticPrototypeBank (PART 21.1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dynatree_cre.prototype_bank import SemanticPrototypeBank


def test_prototype_mean_l2_and_determinism() -> None:
    bank1 = SemanticPrototypeBank(rep_dim=4, similarity_space="raw_cosine")
    bank2 = SemanticPrototypeBank(rep_dim=4, similarity_space="raw_cosine")

    q1 = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    q2 = torch.tensor([[0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 5.0, 0.0]])

    out1 = bank1.register_task_prototypes({1: q1, 2: q2}, task_id=0)
    out2 = bank2.register_task_prototypes({1: q1, 2: q2}, task_id=0)

    assert np.isclose(np.linalg.norm(out1[1].prototype_vector), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(out1[2].prototype_vector), 1.0, atol=1e-6)
    assert np.allclose(out1[1].prototype_vector, out2[1].prototype_vector)
    assert np.allclose(out1[2].prototype_vector, out2[2].prototype_vector)


def test_prototype_rejects_zero_nan_inf_and_bad_dim() -> None:
    bank = SemanticPrototypeBank(rep_dim=4)
    with pytest.raises(ValueError):
        bank.register_task_prototypes({1: torch.zeros(2, 4)}, task_id=0)
    with pytest.raises(ValueError):
        bank.register_task_prototypes({1: torch.tensor([[float("nan"), 1.0, 0.0, 0.0]])}, task_id=0)
    with pytest.raises(ValueError):
        bank.register_task_prototypes({1: torch.tensor([[float("inf"), 1.0, 0.0, 0.0]])}, task_id=0)
    with pytest.raises(ValueError):
        bank.register_task_prototypes({1: torch.ones(2, 5)}, task_id=0)


def test_t1_centered_anchor_stays_frozen_and_roundtrips() -> None:
    bank = SemanticPrototypeBank(rep_dim=4, similarity_space="t1_centered_cosine")
    bank.register_task_prototypes(
        {
            1: torch.tensor([[1.0, 0.2, 0.0, 0.0]]),
            2: torch.tensor([[1.0, -0.2, 0.0, 0.0]]),
        },
        task_id=0,
    )
    anchor_after_t1 = bank.anchor_mean.copy()
    p1_after_t1 = bank.prototypes[1].prototype_vector.copy()

    bank.register_task_prototypes({3: torch.tensor([[0.0, 0.0, 1.0, 0.5]])}, task_id=1)
    assert np.allclose(bank.anchor_mean, anchor_after_t1)
    assert np.allclose(bank.prototypes[1].prototype_vector, p1_after_t1)

    restored = SemanticPrototypeBank.from_state_dict(bank.state_dict())
    assert np.allclose(restored.anchor_mean, bank.anchor_mean)
    assert np.allclose(restored.prototypes[3].prototype_vector, bank.prototypes[3].prototype_vector)
