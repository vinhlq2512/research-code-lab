"""Unit tests for TAPTA P5 Refined Modules (P5PromptedBERT & P5TaskRouter)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import pytest

from src.models.p5_prompt_bert import P5PromptedBERT
from src.models.p5_router import P5TaskRouter


def test_p5_root_anchor_regularization():
    """Verify that Module D correctly anchors Task 1 root prompt and penalizes drift."""
    model = P5PromptedBERT(num_classes=40, num_tasks=8, device="cpu")

    # Before anchoring, loss is 0.0
    loss_0 = model.compute_root_anchor_loss()
    assert float(loss_0.item()) == 0.0

    # Anchor after Task 1
    model.anchor_root_prompt()
    assert bool(model.has_root_anchor.item()) is True
    loss_after_anchor = model.compute_root_anchor_loss()
    assert float(loss_after_anchor.item()) == 0.0

    # Induce drift
    with torch.no_grad():
        model.root_prompt.add_(0.5)

    loss_drift = model.compute_root_anchor_loss()
    assert float(loss_drift.item()) > 0.1


def test_p5_prototype_contrastive_loss():
    """Verify that Module B computes valid contrastive loss against prototype memory buffer."""
    B, D, C = 4, 1536, 10
    h_r = torch.randn(B, D, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    model = P5PromptedBERT(num_classes=40, num_tasks=8, device="cpu")
    model.set_seen_classes(C)

    # Initialize non-zero prototypes for first 10 classes
    with torch.no_grad():
        model.prototypes[:C] = torch.randn(C, D)

    loss = model.compute_prototype_contrastive_loss(
        h_r=h_r,
        labels=labels,
        seen_classes=C,
        temperature=0.1,
    )

    assert loss.dim() == 0
    assert float(loss.item()) > 0.0

    loss.backward()
    assert h_r.grad is not None
    assert float(h_r.grad.norm().item()) > 0.0


def test_p5_confidence_gated_masking():
    """Verify that Module C masks all non-candidate classes and respects confidence margins."""
    B, C = 2, 40
    relations_per_task = 5
    model = P5PromptedBERT(num_classes=C, num_tasks=8, device="cpu")
    model.set_seen_classes(C)

    sim_t1 = torch.randn(B, C)
    sim_t2 = torch.randn(B, C)

    # Sample 0: Confident (margin = 0.2 >= 0.05) -> Task 0 only [0..4]
    # Sample 1: Uncertain (margin = 0.01 < 0.05) -> Tasks {1, 2} [5..14]
    topk_task_ids = torch.tensor([[0, 1], [1, 2]])
    router_scores = torch.tensor([
        [0.8, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # margin = 0.2
        [0.1, 0.75, 0.74, 0.1, 0.1, 0.1, 0.1, 0.1],  # margin = 0.01
    ])

    gated_sim = model.predict_confidence_gated_prototype(
        candidate_sims=[sim_t1, sim_t2],
        topk_task_ids=topk_task_ids,
        router_scores=router_scores,
        relations_per_task=relations_per_task,
        confidence_margin_threshold=0.05,
        temperature=0.2,
    )

    # Sample 0: relations 0..4 unmasked, relations 5..39 must be -1e9
    assert not torch.isclose(gated_sim[0, 0], torch.tensor(-1e9))
    assert not torch.isclose(gated_sim[0, 4], torch.tensor(-1e9))
    assert torch.all(gated_sim[0, 5:] <= -1e8)

    # Sample 1: relations 5..14 unmasked, relations 0..4 and 15..39 must be -1e9
    assert torch.all(gated_sim[1, 0:5] <= -1e8)
    assert not torch.isclose(gated_sim[1, 5], torch.tensor(-1e9))
    assert not torch.isclose(gated_sim[1, 14], torch.tensor(-1e9))
    assert torch.all(gated_sim[1, 15:] <= -1e8)


def test_p5_router_whitening_and_routing():
    """Verify that P5TaskRouter computes confidence margins and valid scores."""
    router = P5TaskRouter(
        input_dim=1536,
        num_tasks=8,
        relations_per_task=5,
        subcentroid_weight=0.4,
        enable_whitening=True,
    )

    # Seed mock data for tasks 0..3
    for t in range(4):
        queries = torch.randn(25, 1536)
        class_indices = list(range(t * 5, (t + 1) * 5))
        sample_labels = torch.tensor([c for c in class_indices for _ in range(5)])
        router.register_task_queries(t, queries, class_indices, sample_labels)

    # Update whitening transform
    router.update_whitening_transform(seen_tasks=4)
    assert bool(router.whitening_ready.item()) is True

    # Test confidence routing
    q_test = torch.randn(4, 1536)
    out = router.get_confidence_routing(q_test, seen_tasks=4, top_k=2)

    assert "topk_indices" in out
    assert "margins" in out
    assert "is_confident" in out
    assert out["topk_indices"].shape == (4, 2)
    assert out["margins"].shape == (4,)
    assert (out["margins"] >= 0.0).all()
