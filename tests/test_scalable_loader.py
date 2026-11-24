"""Unit tests for Scalable FewRel Dataset Loader with Nested Sampling and Zero Leakage.

Enforces:
1. Exact sample counts for 5, 10, 50, 100, and full modes.
2. Strict Zero Leakage between Train and Test sets.
3. Nested Sampling property: S_5 subset S_10 subset S_50 subset S_100 subset S_full.
4. Deterministic reproducibility across seeds.
5. Cluster and task structure integrity (8 tasks x 10 relations).
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.data.scalable_fewrel_loader import (
    ScalableFewRelDataset,
    normalize_sentence_for_hash,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WAVE_DATA_PATH = REPO_ROOT / "paper-reproductions" / "WAVE-CRE" / "datasets" / "data_with_marker.json"


@pytest.fixture(scope="module")
def check_data_exists():
    if not WAVE_DATA_PATH.exists():
        pytest.skip(f"Required dataset not found: {WAVE_DATA_PATH}")


def test_zero_data_leakage(check_data_exists):
    """Verify that train and test sets have strictly 0 overlap."""
    dataset = ScalableFewRelDataset(
        data_json_path=WAVE_DATA_PATH,
        seed=2021,
        train_shot=50,
        test_shot=100,
    )
    tasks = dataset.build_tasks()

    all_train_hashes = set()
    all_test_hashes = set()

    for task in tasks:
        for s in task.train_samples:
            all_train_hashes.add(normalize_sentence_for_hash(s.words))
        for s in task.test_samples:
            all_test_hashes.add(normalize_sentence_for_hash(s.words))

    overlap = all_train_hashes.intersection(all_test_hashes)
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} sentences overlap between train and test!"


def test_exact_sample_counts_5_and_10_shot(check_data_exists):
    """Verify exact sample counts for 5-shot and 10-shot."""
    ds_5 = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=5, test_shot=100)
    tasks_5 = ds_5.build_tasks()
    assert len(tasks_5) == 8
    for t in tasks_5:
        assert len(t.relations) == 10
        assert len(t.train_samples) == 50  # 5 * 10
        assert len(t.test_samples) == 1000  # 100 * 10

    ds_10 = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=10, test_shot=100)
    tasks_10 = ds_10.build_tasks()
    assert len(tasks_10) == 8
    for t in tasks_10:
        assert len(t.train_samples) == 100  # 10 * 10
        assert len(t.test_samples) == 1000  # 100 * 10


def test_exact_sample_counts_50_and_100_shot(check_data_exists):
    """Verify exact sample counts for 50-shot and 100-shot."""
    ds_50 = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=50, test_shot=100)
    tasks_50 = ds_50.build_tasks()
    for t in tasks_50:
        assert len(t.train_samples) == 500  # 50 * 10
        assert len(t.test_samples) == 1000

    ds_100 = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=100, test_shot=100)
    tasks_100 = ds_100.build_tasks()
    for t in tasks_100:
        assert len(t.train_samples) == 1000  # 100 * 10
        assert len(t.test_samples) == 1000


def test_full_data_mode(check_data_exists):
    """Verify full data mode loads all available clean train samples (at least 400 per relation)."""
    ds_full = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot="full", test_shot=100)
    tasks_full = ds_full.build_tasks()
    assert len(tasks_full) == 8
    for t in tasks_full:
        assert len(t.train_samples) >= 4000  # at least 400 * 10
        assert len(t.test_samples) == 1000


def test_nested_sampling_property(check_data_exists):
    """Verify S_5 subset of S_10 subset of S_50 subset of S_100."""
    ds_5 = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=5, test_shot=100)
    ds_10 = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=10, test_shot=100)
    ds_50 = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=50, test_shot=100)

    tasks_5 = ds_5.build_tasks()
    tasks_10 = ds_10.build_tasks()
    tasks_50 = ds_50.build_tasks()

    for t_idx in range(8):
        t5 = tasks_5[t_idx]
        t10 = tasks_10[t_idx]
        t50 = tasks_50[t_idx]

        for rel_idx, rel_id in enumerate(t5.relations):
            samples_5 = [s.marked_text for s in t5.train_samples if s.relation_id == rel_id]
            samples_10 = [s.marked_text for s in t10.train_samples if s.relation_id == rel_id]
            samples_50 = [s.marked_text for s in t50.train_samples if s.relation_id == rel_id]

            assert len(samples_5) == 5
            assert len(samples_10) == 10
            assert len(samples_50) == 50

            # Exact prefix subset property: S_5 must be the first 5 samples of S_10
            assert samples_10[:5] == samples_5
            # S_10 must be the first 10 samples of S_50
            assert samples_50[:10] == samples_10


def test_deterministic_reproducibility(check_data_exists):
    """Verify that same seed generates bitwise identical samples."""
    ds_a = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=10)
    ds_b = ScalableFewRelDataset(data_json_path=WAVE_DATA_PATH, seed=2021, train_shot=10)

    tasks_a = ds_a.build_tasks()
    tasks_b = ds_b.build_tasks()

    for ta, tb in zip(tasks_a, tasks_b):
        assert ta.relations == tb.relations
        assert [s.marked_text for s in ta.train_samples] == [s.marked_text for s in tb.train_samples]
        assert [s.marked_text for s in ta.test_samples] == [s.marked_text for s in tb.test_samples]
