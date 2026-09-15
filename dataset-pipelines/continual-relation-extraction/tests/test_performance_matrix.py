from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.performance_matrix import PerformanceMatrix


class TestPerformanceMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = PerformanceMatrix(
            num_tasks=4,
            metric="accuracy",
            dataset="fewrel",
            seed=42,
            task_order_path="task-orders/fewrel/order_seed_42.json",
        )

    def test_initialization(self) -> None:
        self.assertEqual(self.matrix.num_tasks, 4)
        self.assertEqual(self.matrix.metric, "accuracy")
        self.assertEqual(self.matrix.seed, 42)
        # Verify all cells are None
        for t in range(4):
            for j in range(4):
                self.assertIsNone(self.matrix.get_score(t, j))

    def test_set_and_get_score(self) -> None:
        self.matrix.set_score(trained_until=0, evaluated_task=0, score=0.84)
        self.matrix.set_score(trained_until=1, evaluated_task=0, score=0.79)
        self.matrix.set_score(trained_until=1, evaluated_task=1, score=0.86)

        self.assertEqual(self.matrix.get_score(0, 0), 0.84)
        self.assertEqual(self.matrix.get_score(1, 0), 0.79)
        self.assertEqual(self.matrix.get_score(1, 1), 0.86)
        self.assertIsNone(self.matrix.get_score(0, 1))

    def test_future_eval_rejected_by_default(self) -> None:
        # j > t (evaluated_task=1, trained_until=0)
        with self.assertRaises(ValueError) as ctx:
            self.matrix.set_score(trained_until=0, evaluated_task=1, score=0.50)
        self.assertIn("cannot be greater than trained_until", str(ctx.exception))

    def test_future_eval_allowed_with_flag(self) -> None:
        self.matrix.set_score(trained_until=0, evaluated_task=1, score=0.50, allow_future_eval=True)
        self.assertEqual(self.matrix.get_score(0, 1), 0.50)

    def test_out_of_bounds_indices_rejected(self) -> None:
        with self.assertRaises(IndexError):
            self.matrix.set_score(-1, 0, 0.5)
        with self.assertRaises(IndexError):
            self.matrix.set_score(0, 4, 0.5)
        with self.assertRaises(IndexError):
            self.matrix.get_score(4, 0)
        with self.assertRaises(IndexError):
            self.matrix.get_score(0, -1)

    def test_rows_and_columns(self) -> None:
        self.matrix.set_score(0, 0, 0.84)
        self.matrix.set_score(1, 0, 0.79)
        self.matrix.set_score(1, 1, 0.86)

        row_1 = self.matrix.get_row(1)
        self.assertEqual(row_1, [0.79, 0.86, None, None])

        col_0 = self.matrix.get_column(0)
        self.assertEqual(col_0, [0.84, 0.79, None, None])

    def test_json_export_and_load(self) -> None:
        self.matrix.set_score(0, 0, 0.84)
        self.matrix.set_score(1, 0, 0.79)
        self.matrix.set_score(1, 1, 0.86)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "accuracy_matrix.json"
            self.matrix.export_json(json_path)
            self.assertTrue(json_path.exists())

            loaded = PerformanceMatrix.load_json(json_path)
            self.assertEqual(loaded.num_tasks, 4)
            self.assertEqual(loaded.dataset, "fewrel")
            self.assertEqual(loaded.seed, 42)
            self.assertEqual(loaded.get_score(0, 0), 0.84)
            self.assertEqual(loaded.get_score(1, 0), 0.79)
            self.assertEqual(loaded.get_score(1, 1), 0.86)
            self.assertIsNone(loaded.get_score(0, 1))

    def test_csv_export_and_load(self) -> None:
        self.matrix.set_score(0, 0, 0.84)
        self.matrix.set_score(1, 0, 0.79)
        self.matrix.set_score(1, 1, 0.86)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "accuracy_matrix.csv"
            self.matrix.export_csv(csv_path)
            self.assertTrue(csv_path.exists())

            csv_text = csv_path.read_text()
            # Verify missing cells are empty string, NOT zero
            lines = csv_text.strip().splitlines()
            self.assertEqual(lines[0], "trained_until,task_0,task_1,task_2,task_3")
            self.assertEqual(lines[1], "0,0.84,,,")
            self.assertEqual(lines[2], "1,0.79,0.86,,")

            loaded = PerformanceMatrix.load_csv(csv_path)
            self.assertEqual(loaded.get_score(0, 0), 0.84)
            self.assertEqual(loaded.get_score(1, 0), 0.79)
            self.assertEqual(loaded.get_score(1, 1), 0.86)
            self.assertIsNone(loaded.get_score(0, 1))


if __name__ == "__main__":
    unittest.main()
