from __future__ import annotations

import random
from typing import Sequence

from evaluation.evaluator import RelationPredictor
from schemas.relation_sample import RelationSample


class PerfectPredictor(RelationPredictor):
    """Oracle predictor that always returns the exact ground-truth relation ID.

    Guarantees Accuracy = 1.0 and Macro F1 = 1.0 on any valid evaluation set.
    """

    def predict(self, samples: Sequence[RelationSample]) -> list[int]:
        return [sample.relation_id for sample in samples]


class ConstantPredictor(RelationPredictor):
    """Predictor that always outputs a fixed constant relation ID."""

    def __init__(self, constant_id: int = 0) -> None:
        self.constant_id = int(constant_id)

    def predict(self, samples: Sequence[RelationSample]) -> list[int]:
        return [self.constant_id for _ in samples]


class RandomPredictor(RelationPredictor):
    """Predictor that outputs uniformly random relation IDs from a known class pool."""

    def __init__(self, candidate_classes: Sequence[int], seed: int = 42) -> None:
        if not candidate_classes:
            raise ValueError("candidate_classes cannot be empty")
        self.candidate_classes = list(candidate_classes)
        self.rng = random.Random(seed)

    def predict(self, samples: Sequence[RelationSample]) -> list[int]:
        return [self.rng.choice(self.candidate_classes) for _ in samples]


class DecayingPredictor(RelationPredictor):
    """Deterministic dummy predictor that simulates forgetting across continual tasks.

    For task t, performance on task j decays as (t - j) increases.
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        base_accuracy: float = 0.90,
        seed: int = 42,
        num_classes: int = 80,
    ) -> None:
        self.decay_rate = decay_rate
        self.base_accuracy = base_accuracy
        self.num_classes = num_classes
        self.seed = seed

    def predict_for_task(
        self,
        samples: Sequence[RelationSample],
        trained_until: int,
        evaluated_task: int,
    ) -> list[int]:
        """Predict with accuracy scaled by the gap between trained_until and evaluated_task."""
        lag = max(0, trained_until - evaluated_task)
        target_acc = max(0.10, self.base_accuracy - (lag * self.decay_rate))

        rng = random.Random(self.seed + trained_until * 100 + evaluated_task)
        predictions: list[int] = []
        for s in samples:
            if rng.random() < target_acc:
                predictions.append(s.relation_id)
            else:
                # Predict a wrong relation
                wrong = (s.relation_id + 1) % self.num_classes
                predictions.append(wrong)
        return predictions

    def predict(self, samples: Sequence[RelationSample]) -> list[int]:
        # Default behavior: perfect predictions
        return [sample.relation_id for sample in samples]
