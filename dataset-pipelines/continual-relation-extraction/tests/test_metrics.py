from __future__ import annotations

import unittest

from evaluation.dummy_predictor import (
    ConstantPredictor,
    DecayingPredictor,
    PerfectPredictor,
)
from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    compute_all_average_accuracies,
    compute_average_accuracy,
    compute_average_forgetting,
    compute_continual_summary_metrics,
    compute_task_forgetting,
)
from evaluation.performance_matrix import PerformanceMatrix
from schemas.relation_sample import RelationSample


class TestContinualMetrics(unittest.TestCase):
    def setUp(self) -> None:
        # Hand-calculated matrix from specification:
        # [
        #   [0.90, None, None],
        #   [0.80, 0.85, None],
        #   [0.70, 0.80, 0.88]
        # ]
        self.matrix = PerformanceMatrix(num_tasks=3, metric="accuracy", dataset="fewrel", seed=42)
        self.matrix.set_score(0, 0, 0.90)

        self.matrix.set_score(1, 0, 0.80)
        self.matrix.set_score(1, 1, 0.85)

        self.matrix.set_score(2, 0, 0.70)
        self.matrix.set_score(2, 1, 0.80)
        self.matrix.set_score(2, 2, 0.88)

    def test_average_accuracy_hand_calculated(self) -> None:
        # Stage 0: 0.90
        acc_0 = compute_average_accuracy(self.matrix, trained_until=0)
        self.assertAlmostEqual(acc_0, 0.90)

        # Stage 1: (0.80 + 0.85) / 2 = 0.825
        acc_1 = compute_average_accuracy(self.matrix, trained_until=1)
        self.assertAlmostEqual(acc_1, 0.825)

        # Stage 2: (0.70 + 0.80 + 0.88) / 3 = 2.38 / 3 ≈ 0.7933333333
        acc_2 = compute_average_accuracy(self.matrix, trained_until=2)
        self.assertAlmostEqual(acc_2, 2.38 / 3)

        # All stages
        all_acc = compute_all_average_accuracies(self.matrix)
        self.assertEqual(len(all_acc), 3)
        self.assertAlmostEqual(all_acc[0], 0.90)
        self.assertAlmostEqual(all_acc[1], 0.825)
        self.assertAlmostEqual(all_acc[2], 2.38 / 3)

    def test_task_forgetting_hand_calculated(self) -> None:
        # Task 0: max(0.90, 0.80) - 0.70 = 0.20
        f_0 = compute_task_forgetting(self.matrix, task_id=0, current_task=2)
        self.assertAlmostEqual(f_0, 0.20)

        # Task 1: max(0.85) - 0.80 = 0.05
        f_1 = compute_task_forgetting(self.matrix, task_id=1, current_task=2)
        self.assertAlmostEqual(f_1, 0.05)

        # Task 2: trained at stage 2, so forgetting is 0.0
        f_2 = compute_task_forgetting(self.matrix, task_id=2, current_task=2)
        self.assertAlmostEqual(f_2, 0.0)

    def test_average_forgetting_hand_calculated(self) -> None:
        # Avg F across tasks 0 and 1 at stage 2: (0.20 + 0.05) / 2 = 0.125
        avg_f = compute_average_forgetting(self.matrix, current_task=2)
        self.assertAlmostEqual(avg_f, 0.125)

    def test_summary_metrics(self) -> None:
        summary = compute_continual_summary_metrics(self.matrix)
        self.assertAlmostEqual(summary["final_average_accuracy"], 2.38 / 3)
        self.assertAlmostEqual(summary["average_forgetting"], 0.125)
        self.assertAlmostEqual(summary["per_task_forgetting"]["task_0"], 0.20)
        self.assertAlmostEqual(summary["per_task_forgetting"]["task_1"], 0.05)
        self.assertAlmostEqual(summary["per_task_forgetting"]["task_2"], 0.0)

    def test_missing_cell_error_handling(self) -> None:
        incomplete_matrix = PerformanceMatrix(num_tasks=2)
        incomplete_matrix.set_score(0, 0, 0.90)
        # Stage 1 has missing cell (1, 0)
        incomplete_matrix.set_score(1, 1, 0.85)

        with self.assertRaises(ValueError) as ctx:
            compute_average_accuracy(incomplete_matrix, trained_until=1)
        self.assertIn("is None", str(ctx.exception))


class TestDummyPredictors(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            RelationSample(
                sample_id=f"sample_{i}",
                tokens=["A", "rel", "B"],
                head_text="A",
                head_start=0,
                head_end=1,
                tail_text="B",
                tail_start=2,
                tail_end=3,
                relation=f"rel_{rel_id}",
                relation_id=rel_id,
            )
            for i, rel_id in enumerate([0, 1, 2, 3])
        ]
        self.evaluator = Evaluator()

    def test_perfect_predictor(self) -> None:
        predictor = PerfectPredictor()
        result = self.evaluator.evaluate(predictor, self.samples)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.macro_f1, 1.0)

    def test_constant_predictor(self) -> None:
        # Constant predictor predicts class 0 for all 4 samples
        # Only sample 0 is class 0 -> accuracy = 1 / 4 = 0.25
        predictor = ConstantPredictor(constant_id=0)
        result = self.evaluator.evaluate(predictor, self.samples)
        self.assertEqual(result.accuracy, 0.25)

    def test_decaying_predictor(self) -> None:
        predictor = DecayingPredictor(decay_rate=0.20, base_accuracy=0.90, seed=42)
        # Lag 0: evaluated_task == trained_until (no decay)
        preds_lag0 = predictor.predict_for_task(self.samples, trained_until=0, evaluated_task=0)
        self.assertEqual(len(preds_lag0), len(self.samples))

        # Lag 3: evaluated_task 0 evaluated at trained_until 3 (accuracy drops)
        preds_lag3 = predictor.predict_for_task(self.samples, trained_until=3, evaluated_task=0)
        self.assertEqual(len(preds_lag3), len(self.samples))


if __name__ == "__main__":
    unittest.main()
