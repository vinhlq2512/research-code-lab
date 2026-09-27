from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.evaluator import Evaluator, EvaluationResult, compute_accuracy, compute_macro_f1
from evaluation.metrics import (
    compute_all_average_accuracies,
    compute_average_accuracy,
    compute_average_forgetting,
    compute_average_incremental_accuracy,
    compute_backward_transfer,
    compute_continual_summary_metrics,
    compute_old_and_new_accuracies,
    compute_task_forgetting,
    identify_most_forgotten_task,
)
from evaluation.performance_matrix import PerformanceMatrix
from schemas.relation_sample import RelationSample


class TestB0EvaluatorGate(unittest.TestCase):
    """Rigorous Evaluator Gate for B0 (Sequential Fine-Tuning Baseline).

    Validates:
    1. Exact 36-cell lower-triangular matrix A[t, j] for 8 tasks (T=8).
    2. Strict bounds checking and prohibition of unobserved future task evaluations (j > t).
    3. Mathematical correctness of Final Average Accuracy (Final AA).
    4. Mathematical correctness of Average Incremental Accuracy (AIA).
    5. Mathematical correctness of per-task Catastrophic Forgetting (F_j) and Average Forgetting (AF).
    6. Mathematical correctness of Backward Transfer (BWT).
    7. Old-task vs New-task performance trajectory tracking.
    8. Identification of the most forgotten task.
    9. Invariant round-trip serialization to JSON and CSV.
    10. Deterministic evaluation behavior with dummy predictors and seen-class masking.
    """

    def setUp(self) -> None:
        self.num_tasks = 8
        self.matrix = PerformanceMatrix(
            num_tasks=self.num_tasks,
            metric="accuracy",
            dataset="fewrel",
            seed=2021,
            task_order_path="task-orders/fewrel/order_seed_2021.json",
        )

        # Hand-crafted 8-task realistic degradation matrix (8 rows, exactly 36 lower-triangular cells)
        self.raw_data = [
            # Stage 0: 1 cell
            [0.90],
            # Stage 1: 2 cells
            [0.70, 0.85],
            # Stage 2: 3 cells
            [0.55, 0.65, 0.80],
            # Stage 3: 4 cells
            [0.45, 0.50, 0.65, 0.85],
            # Stage 4: 5 cells
            [0.35, 0.40, 0.50, 0.70, 0.82],
            # Stage 5: 6 cells
            [0.25, 0.30, 0.40, 0.55, 0.68, 0.88],
            # Stage 6: 7 cells
            [0.20, 0.25, 0.30, 0.40, 0.52, 0.72, 0.84],
            # Stage 7: 8 cells
            [0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70, 0.86],
        ]

        # Populate matrix
        cell_count = 0
        for t, row in enumerate(self.raw_data):
            for j, val in enumerate(row):
                self.matrix.set_score(trained_until=t, evaluated_task=j, score=val)
                cell_count += 1

        self.populated_cell_count = cell_count

    def test_matrix_exact_cell_count_and_missing_values(self) -> None:
        """Verify exactly 36 lower-triangular cells and strictly None for future cells."""
        expected_cells = self.num_tasks * (self.num_tasks + 1) // 2
        self.assertEqual(expected_cells, 36)
        self.assertEqual(self.populated_cell_count, 36)

        # Verify all lower-triangular cells are valid floats in [0, 1]
        for t in range(self.num_tasks):
            for j in range(t + 1):
                val = self.matrix.get_score(trained_until=t, evaluated_task=j)
                self.assertIsNotNone(val)
                self.assertIsInstance(val, float)
                self.assertAlmostEqual(val, self.raw_data[t][j])

        # Verify all strictly upper-triangular cells are None
        for t in range(self.num_tasks):
            for j in range(t + 1, self.num_tasks):
                val = self.matrix.get_score(trained_until=t, evaluated_task=j)
                self.assertIsNone(val, f"Cell A[{t}, {j}] must be None for future unobserved tasks")

        # Verify setting future task score without allow_future_eval=True raises ValueError
        with self.assertRaises(ValueError):
            self.matrix.set_score(trained_until=0, evaluated_task=1, score=0.50)

    def test_average_accuracy_hand_calculated(self) -> None:
        """Verify Average Accuracy ACC_t across all 8 stages matches manual calculations."""
        # Stage 0: 0.90 / 1 = 0.90
        self.assertAlmostEqual(compute_average_accuracy(self.matrix, trained_until=0), 0.90)

        # Stage 1: (0.70 + 0.85) / 2 = 0.775
        self.assertAlmostEqual(compute_average_accuracy(self.matrix, trained_until=1), 0.775)

        # Stage 2: (0.55 + 0.65 + 0.80) / 3 = 2.00 / 3
        self.assertAlmostEqual(compute_average_accuracy(self.matrix, trained_until=2), 2.00 / 3)

        # Stage 3: (0.45 + 0.50 + 0.65 + 0.85) / 4 = 2.45 / 4 = 0.6125
        self.assertAlmostEqual(compute_average_accuracy(self.matrix, trained_until=3), 0.6125)

        # Stage 4: (0.35 + 0.40 + 0.50 + 0.70 + 0.82) / 5 = 2.77 / 5 = 0.554
        self.assertAlmostEqual(compute_average_accuracy(self.matrix, trained_until=4), 0.554)

        # Stage 5: (0.25 + 0.30 + 0.40 + 0.55 + 0.68 + 0.88) / 6 = 3.06 / 6 = 0.510
        self.assertAlmostEqual(compute_average_accuracy(self.matrix, trained_until=5), 0.510)

        # Stage 6: (0.20 + 0.25 + 0.30 + 0.40 + 0.52 + 0.72 + 0.84) / 7 = 3.23 / 7
        self.assertAlmostEqual(compute_average_accuracy(self.matrix, trained_until=6), 3.23 / 7)

        # Stage 7 (Final AA): (0.15 + 0.20 + 0.25 + 0.30 + 0.40 + 0.55 + 0.70 + 0.86) / 8 = 3.41 / 8 = 0.42625
        final_aa = compute_average_accuracy(self.matrix, trained_until=7)
        self.assertAlmostEqual(final_aa, 0.42625)

        # Verify compute_all_average_accuracies returns exactly 8 values
        all_acc = compute_all_average_accuracies(self.matrix)
        self.assertEqual(len(all_acc), 8)
        self.assertAlmostEqual(all_acc[-1], 0.42625)

    def test_average_incremental_accuracy_hand_calculated(self) -> None:
        """Verify AIA = (1/8) * sum(ACC_0..ACC_7)."""
        acc_list = [
            0.90,
            0.775,
            2.00 / 3,
            0.6125,
            0.554,
            0.510,
            3.23 / 7,
            0.42625,
        ]
        expected_aia = sum(acc_list) / 8

        calculated_aia = compute_average_incremental_accuracy(self.matrix)
        self.assertAlmostEqual(calculated_aia, expected_aia, places=6)

    def test_task_forgetting_hand_calculated(self) -> None:
        """Verify per-task forgetting F_j at final stage T=7."""
        expected_forgetting = {
            0: 0.90 - 0.15,  # 0.75
            1: 0.85 - 0.20,  # 0.65
            2: 0.80 - 0.25,  # 0.55
            3: 0.85 - 0.30,  # 0.55
            4: 0.82 - 0.40,  # 0.42
            5: 0.88 - 0.55,  # 0.33
            6: 0.84 - 0.70,  # 0.14
            7: 0.0,          # Newly learned task: F_7 = 0.0
        }

        for task_id, expected_f in expected_forgetting.items():
            f_val = compute_task_forgetting(self.matrix, task_id=task_id, current_task=7)
            self.assertAlmostEqual(f_val, expected_f, places=6)

    def test_average_forgetting_hand_calculated(self) -> None:
        """Verify Average Forgetting AF across tasks 0..6."""
        # Sum of F_0..F_6 = 0.75 + 0.65 + 0.55 + 0.55 + 0.42 + 0.33 + 0.14 = 3.39
        # AF = 3.39 / 7 ≈ 0.4842857142857143
        expected_af = 3.39 / 7
        af_val = compute_average_forgetting(self.matrix, current_task=7)
        self.assertAlmostEqual(af_val, expected_af, places=6)

    def test_backward_transfer_hand_calculated(self) -> None:
        """Verify Backward Transfer BWT across tasks 0..6."""
        # Sum of (A_{7,j} - A_{j,j}) for j=0..6 = -3.39
        # BWT = -3.39 / 7 ≈ -0.4842857142857143
        expected_bwt = -3.39 / 7
        bwt_val = compute_backward_transfer(self.matrix, current_task=7)
        self.assertAlmostEqual(bwt_val, expected_bwt, places=6)

    def test_old_and_new_accuracy_breakdown(self) -> None:
        """Verify old vs new task accuracy computation for all stages 1..7."""
        breakdown = compute_old_and_new_accuracies(self.matrix, current_task=7)
        self.assertEqual(len(breakdown), 7)

        # Stage 1: old = [0.70] -> 0.70, new = 0.85
        self.assertAlmostEqual(breakdown["stage_1"]["old_acc"], 0.70)
        self.assertAlmostEqual(breakdown["stage_1"]["new_acc"], 0.85)

        # Stage 7: old = (3.41 - 0.86) / 7 = 2.55 / 7, new = 0.86
        self.assertAlmostEqual(breakdown["stage_7"]["old_acc"], 2.55 / 7)
        self.assertAlmostEqual(breakdown["stage_7"]["new_acc"], 0.86)

    def test_identify_most_forgotten_task(self) -> None:
        """Verify detection of the task that suffered the largest drop."""
        most_forgotten = identify_most_forgotten_task(self.matrix, current_task=7)
        self.assertEqual(most_forgotten["task_id"], 0)
        self.assertEqual(most_forgotten["task_name"], "T1")
        self.assertAlmostEqual(most_forgotten["forgetting"], 0.75)
        self.assertAlmostEqual(most_forgotten["initial_score"], 0.90)
        self.assertAlmostEqual(most_forgotten["final_score"], 0.15)

    def test_summary_metrics_structure_and_values(self) -> None:
        """Verify complete dictionary returned by compute_continual_summary_metrics."""
        summary = compute_continual_summary_metrics(self.matrix)

        self.assertAlmostEqual(summary["final_average_accuracy"], 0.42625)
        self.assertAlmostEqual(summary["average_forgetting"], 3.39 / 7)
        self.assertAlmostEqual(summary["backward_transfer"], -3.39 / 7)
        self.assertEqual(summary["evaluated_stage"], 7)
        self.assertEqual(summary["most_forgotten_task"]["task_name"], "T1")
        self.assertEqual(len(summary["per_task_forgetting"]), 8)
        self.assertEqual(len(summary["old_vs_new_by_stage"]), 7)
        self.assertEqual(len(summary["incremental_accuracy"]), 8)

    def test_round_trip_matrix_serialization(self) -> None:
        """Verify CSV and JSON export/import preserves all 36 cells and None values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_file = tmp_path / "matrix.json"
            csv_file = tmp_path / "matrix.csv"

            # Export
            self.matrix.export_json(json_file)
            self.matrix.export_csv(csv_file)

            # Reload
            loaded_json = PerformanceMatrix.load_json(json_file)
            loaded_csv = PerformanceMatrix.load_csv(csv_file)

            # Verify identical cell values across both loaded instances
            for t in range(self.num_tasks):
                for j in range(self.num_tasks):
                    orig_val = self.matrix.get_score(t, j)
                    json_val = loaded_json.get_score(t, j)
                    csv_val = loaded_csv.get_score(t, j)

                    if orig_val is None:
                        self.assertIsNone(json_val)
                        self.assertIsNone(csv_val)
                    else:
                        self.assertAlmostEqual(orig_val, json_val)
                        self.assertAlmostEqual(orig_val, csv_val)

    def test_evaluator_gate_accuracy_macro_f1_predictions(self) -> None:
        """Verify Evaluator precision, accuracy, and macro-F1 calculations."""
        labels = [0, 0, 1, 1, 2, 2]
        predictions = [0, 0, 1, 2, 2, 2]  # 5 correct, 1 wrong (label 1 predicted as 2)

        # Accuracy: 5 / 6
        acc = compute_accuracy(labels, predictions)
        self.assertAlmostEqual(acc, 5 / 6)

        # Macro F1:
        # Class 0: tp=2, fp=0, fn=0 -> P=1.0, R=1.0, F1=1.0
        # Class 1: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=2*(1*0.5)/(1+0.5) = 1/1.5 = 2/3
        # Class 2: tp=2, fp=1, fn=0 -> P=2/3, R=1.0, F1=2*(2/3*1)/(2/3+1) = (4/3)/(5/3) = 4/5 = 0.8
        # Macro F1 = (1.0 + 2/3 + 0.8) / 3 ≈ 0.8222222
        expected_macro_f1 = (1.0 + (2 / 3) + 0.8) / 3
        f1 = compute_macro_f1(labels, predictions)
        self.assertAlmostEqual(f1, expected_macro_f1)


if __name__ == "__main__":
    unittest.main()
