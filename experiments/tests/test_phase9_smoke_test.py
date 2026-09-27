#!/usr/bin/env python3
"""
Unit test & automated validator for Phase 9: 2-Task Smoke Test (T1 -> T2).
Ensures the continual learning protocol invariants are strictly satisfied:
1. Matrix is lower-triangular (A[0][0] is float, A[0][1] is None, A[1][0] and A[1][1] are floats).
2. Exactly 3 records in metrics.jsonl.
3. Checkpoints after_T1 and after_T2 exist and contain metadata.
4. BWT and AIA are computed correctly on the 2x2 matrix.
5. Old vs New breakdown is present for stage 1.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE_DIR = REPO_ROOT / "results" / "fewrel" / "5shot" / "B0_smoke_test" / "seed_2021"


class TestB0SmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.smoke_dir = SMOKE_DIR
        self.assertTrue(self.smoke_dir.exists(), f"Smoke test dir does not exist: {self.smoke_dir}")

    def test_performance_matrix_structure(self) -> None:
        matrix_file = self.smoke_dir / "performance_matrix.json"
        self.assertTrue(matrix_file.exists(), "performance_matrix.json must exist")

        with matrix_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("num_tasks"), 2)
        matrix = data.get("matrix")
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), 2)
        self.assertEqual(len(matrix[1]), 2)

        # Stage 0 (after T1): A_0,0 is float, A_0,1 must be None
        self.assertIsInstance(matrix[0][0], float)
        self.assertGreater(matrix[0][0], 0.0)
        self.assertIsNone(matrix[0][1], "A_0,1 must be None because T2 has not been seen yet")

        # Stage 1 (after T2): A_1,0 and A_1,1 must both be floats
        self.assertIsInstance(matrix[1][0], float)
        self.assertIsInstance(matrix[1][1], float)

    def test_metrics_jsonl_count_and_content(self) -> None:
        metrics_file = self.smoke_dir / "metrics.jsonl"
        self.assertTrue(metrics_file.exists(), "metrics.jsonl must exist")

        with metrics_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Exactly 3 evaluations: stage 0 (T1), stage 1 (T1), stage 1 (T2)
        self.assertEqual(len(lines), 3, f"Expected exactly 3 records in metrics.jsonl, got {len(lines)}")

        rec0 = json.loads(lines[0])
        rec1 = json.loads(lines[1])
        rec2 = json.loads(lines[2])

        self.assertEqual(rec0["stage"], 0)
        self.assertEqual(rec0["test_task_name"], "T1")

        self.assertEqual(rec1["stage"], 1)
        self.assertEqual(rec1["test_task_name"], "T1")

        self.assertEqual(rec2["stage"], 1)
        self.assertEqual(rec2["test_task_name"], "T2")

    def test_checkpoints(self) -> None:
        ckpt_dir = self.smoke_dir / "checkpoints"
        self.assertTrue(ckpt_dir.exists(), "checkpoints dir must exist")

        ckpt_t1 = ckpt_dir / "after_T1"
        ckpt_t2 = ckpt_dir / "after_T2"
        self.assertTrue(ckpt_t1.exists(), "checkpoint after_T1 must exist")
        self.assertTrue(ckpt_t2.exists(), "checkpoint after_T2 must exist")

        # Check metadata in after_T1
        meta_t1_path = ckpt_t1 / "stage_metadata.json"
        self.assertTrue(meta_t1_path.exists(), "stage_metadata.json must exist in after_T1")
        with meta_t1_path.open("r", encoding="utf-8") as f:
            meta1 = json.load(f)
        self.assertEqual(meta1.get("task_id"), 0)
        self.assertEqual(meta1.get("task_name"), "T1")

        # Check metadata in after_T2
        meta_t2_path = ckpt_t2 / "stage_metadata.json"
        self.assertTrue(meta_t2_path.exists(), "stage_metadata.json must exist in after_T2")
        with meta_t2_path.open("r", encoding="utf-8") as f:
            meta2 = json.load(f)
        self.assertEqual(meta2.get("task_id"), 1)
        self.assertEqual(meta2.get("task_name"), "T2")

    def test_summary_metrics(self) -> None:
        summary_file = self.smoke_dir / "summary.json"
        self.assertTrue(summary_file.exists(), "summary.json must exist")

        with summary_file.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        # AIA for 2 stages: (A_0,0 + (A_1,0 + A_1,1)/2) / 2
        a00 = summary["incremental_accuracy"]["T1"]
        a1 = summary["incremental_accuracy"]["T2"]
        expected_aia = (a00 + a1) / 2.0
        self.assertAlmostEqual(summary["average_incremental_accuracy"], expected_aia, places=6)

        # BWT for 2 stages: A_1,0 - A_0,0
        expected_bwt = summary["most_forgotten_task"]["final_score"] - summary["most_forgotten_task"]["initial_score"]
        self.assertAlmostEqual(summary["backward_transfer"], expected_bwt, places=6)

        # Old vs New for stage 1
        old_vs_new = summary.get("old_vs_new_by_stage", {}).get("stage_1", {})
        self.assertIn("old_acc", old_vs_new)
        self.assertIn("new_acc", old_vs_new)
        self.assertAlmostEqual(old_vs_new["old_acc"], summary["most_forgotten_task"]["final_score"], places=6)


if __name__ == "__main__":
    unittest.main()
