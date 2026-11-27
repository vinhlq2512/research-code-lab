"""Unit tests for ExpandableCosineClassifier and NCM head (PART 21.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.dynatree_cre.classifier import ExpandableCosineClassifier
from src.dynatree_cre.config import ClassifierConfig


def test_classifier_expansion_old_weight_preservation_and_zero_future_logits() -> None:
    clf = ExpandableCosineClassifier(rep_dim=8, config=ClassifierConfig(scale=20.0, freeze_old_weights=True))
    clf.expand_for_task([10, 20])
    assert clf.num_seen_classes == 2

    z = torch.randn(3, 8)
    logits_t1 = clf(z)
    assert logits_t1.shape == (3, 2)
    clf.update_ncm_prototypes({10: torch.randn(4, 8), 20: torch.randn(4, 8)})

    # Expand to T2 with relations [30, 40]
    clf.expand_for_task([30, 40])
    assert clf.num_seen_classes == 4
    frozen_snap = clf.snapshot_frozen_weights()

    opt = torch.optim.AdamW(clf.trainable_parameters(), lr=0.05, weight_decay=0.1)
    logits_t2 = clf(z)
    assert logits_t2.shape == (3, 4)  # No future-class logits!

    targets = clf.relation_ids_to_slots(torch.tensor([30, 40, 30]))
    loss = F.cross_entropy(logits_t2, targets)
    loss.backward()
    opt.step()

    for key, snap in frozen_snap.items():
        curr = clf.snapshot_frozen_weights()[key]
        assert torch.equal(snap, curr), f"Frozen classifier tensor {key} changed!"

    # Save/load round-trip
    meta = clf.metadata_state()
    state = clf.state_dict()
    clf2 = ExpandableCosineClassifier(rep_dim=8, config=ClassifierConfig(scale=20.0))
    clf2.restore_metadata_and_state(meta, state)
    assert torch.allclose(clf(z), clf2(z), atol=1e-6)
