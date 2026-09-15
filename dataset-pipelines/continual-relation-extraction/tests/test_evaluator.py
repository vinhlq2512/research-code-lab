from __future__ import annotations

import unittest
from typing import Sequence

from evaluation.evaluator import (
    EvaluationResult,
    Evaluator,
    compute_accuracy,
    compute_macro_f1,
)
from schemas.relation_sample import RelationSample


class DummyPredictor:
    def __init__(self, predictions: Sequence[int]) -> None:
        self.predictions = list(predictions)

    def predict(self, samples: Sequence[RelationSample]) -> Sequence[int]:
        return self.predictions


class PerfectPredictor:
    def predict(self, samples: Sequence[RelationSample]) -> Sequence[int]:
        return [s.relation_id for s in samples]


class TestEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = Evaluator()
        self.samples = [
            RelationSample(
                sample_id=f"s{i}",
                tokens=["Alice", "works", "at", "Google"],
                head_text="Alice",
                head_start=0,
                head_end=1,
                tail_text="Google",
                tail_start=3,
                tail_end=4,
                relation="works_for",
                relation_id=rel_id,
            )
            for i, rel_id in enumerate([0, 0, 1, 1])
        ]

    def test_perfect_predictor(self) -> None:
        predictor = PerfectPredictor()
        result = self.evaluator.evaluate(predictor, self.samples)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.macro_f1, 1.0)
        self.assertEqual(result.sample_count, 4)
        self.assertEqual(result.predictions, [0, 0, 1, 1])
        self.assertEqual(result.labels, [0, 0, 1, 1])

    def test_hand_calculated_metrics(self) -> None:
        # labels: [0, 0, 1, 1]
        # predictions: [0, 1, 1, 1]
        # Accuracy = 3 / 4 = 0.75
        # Class 0: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5 -> F1 = 2/3
        # Class 1: TP=2, FP=1, FN=0 -> Prec=2/3, Rec=1.0 -> F1 = 4/5
        # Macro F1 = (2/3 + 4/5) / 2 = 11/15 ≈ 0.7333333333
        predictor = DummyPredictor([0, 1, 1, 1])
        result = self.evaluator.evaluate(predictor, self.samples)

        self.assertAlmostEqual(result.accuracy, 0.75)
        self.assertAlmostEqual(result.macro_f1, 11 / 15)
        self.assertEqual(result.sample_count, 4)

    def test_empty_dataset_handling(self) -> None:
        predictor = PerfectPredictor()
        result = self.evaluator.evaluate(predictor, [])
        self.assertEqual(result.accuracy, 0.0)
        self.assertEqual(result.macro_f1, 0.0)
        self.assertEqual(result.sample_count, 0)
        self.assertEqual(result.predictions, [])
        self.assertEqual(result.labels, [])

    def test_prediction_length_mismatch_raises(self) -> None:
        # Predictor returns 3 predictions for 4 samples
        predictor = DummyPredictor([0, 1, 1])
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.evaluate(predictor, self.samples)
        self.assertIn("predictions for 4 samples", str(ctx.exception))

    def test_compute_functions_directly(self) -> None:
        self.assertEqual(compute_accuracy([], []), 0.0)
        self.assertEqual(compute_macro_f1([], []), 0.0)

        with self.assertRaises(ValueError):
            compute_accuracy([1], [1, 2])
        with self.assertRaises(ValueError):
            compute_macro_f1([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
