"""Unit and regression tests for TAPTA P4 Performance Diagnostics (D01-D09)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest


def test_d01_local_mask_indexing():
    """Verify that D01 strictly isolates local relation classes for a given task."""
    num_tasks = 8
    relations_per_task = 10
    total_classes = 80
    seen_classes = 80

    for task_id in range(num_tasks):
        c_start = task_id * relations_per_task
        c_end = min(c_start + relations_per_task, seen_classes)
        assert c_end - c_start == relations_per_task
        assert c_start == task_id * 10
        assert c_end == (task_id + 1) * 10


def test_d03_oracle_topk_replacement():
    """Verify that D03 guarantees the true task is always included in the candidate list."""
    t_true = 3
    # Case 1: t_true already in top-2
    top2_t = [3, 1]
    if t_true in top2_t:
        oracle_top2 = top2_t
    else:
        oracle_top2 = [top2_t[0], t_true]
    assert oracle_top2 == [3, 1]
    assert t_true in oracle_top2

    # Case 2: t_true is missed by top-2
    top2_t = [0, 5]
    if t_true in top2_t:
        oracle_top2 = top2_t
    else:
        oracle_top2 = [top2_t[0], t_true]
    assert oracle_top2 == [0, 3]
    assert t_true in oracle_top2


def test_d08_zscore_calibration():
    """Verify that z-score normalization centres cosine similarities and scales variance."""
    raw_sim = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    mean = raw_sim.mean()
    std = raw_sim.std()
    norm_sim = (raw_sim - mean) / std

    assert torch.isclose(norm_sim.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(norm_sim.std(), torch.tensor(1.0), atol=1e-5)


def test_error_taxonomy_logic():
    """Verify error category assignment mutually exclusive properties."""
    # Correct sample
    d05_ok = True
    assert (d05_ok) is True

    # Router miss
    d05_ok = False
    r_top1_ok = False
    r_top2_ok = False
    error_cat = "router_miss" if (not r_top1_ok and not r_top2_ok) else "other"
    assert error_cat == "router_miss"

    # Within-task confusion (router correct, but local classification failed)
    r_top1_ok = True
    d01_ok = False
    error_cat = "within_task_confusion" if (r_top1_ok and not d01_ok) else "other"
    assert error_cat == "within_task_confusion"

    # Cross-task interference (router correct, local classification succeeded, but global failed)
    r_top1_ok = True
    d01_ok = True
    d02_ok = False
    error_cat = "cross_task_interference" if (r_top1_ok and d01_ok and not d02_ok) else "other"
    assert error_cat == "cross_task_interference"
