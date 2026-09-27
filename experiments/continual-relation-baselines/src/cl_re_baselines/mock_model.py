from __future__ import annotations

import random
from typing import Any, Sequence

from schemas.relation_sample import RelationSample


class MockRelationClassifier:
    """Mock Relation Classifier simulating sequential fine-tuning dynamics without PyTorch.

    Behaviors:
        - Maintains an 80-class output space.
        - Supports seen-class masking (predictions restricted to [0, seen_classes - 1]).
        - Simulates catastrophic forgetting:
            When training on task t, performance on relations in task t improves to ~85%,
            while performance on previously learned tasks decays by ~10% per subsequent task.
        - Satisfies the exact same protocol as BERTRelationClassifier:
            .predict(samples) -> Sequence[int]
            .set_seen_classes(n)
            .get_state_dict()
            .load_state_dict()
    """

    def __init__(
        self,
        num_classes: int = 80,
        base_accuracy: float = 0.88,
        forgetting_rate: float = 0.08,
        seed: int = 2021,
    ) -> None:
        self.num_classes = num_classes
        self.base_accuracy = base_accuracy
        self.forgetting_rate = forgetting_rate
        self.seed = seed
        self.rng = random.Random(seed)

        self.seen_classes_count = num_classes
        self.current_training_stage = 0

        # Mapping of relation_id -> last_trained_stage
        self.relation_last_trained_stage: dict[int, int] = {}

    def set_seen_classes(self, num_seen_classes: int) -> None:
        self.seen_classes_count = min(num_seen_classes, self.num_classes)

    def set_training_stage(self, stage: int, task_relation_ids: Sequence[int]) -> None:
        self.current_training_stage = stage
        for rel_id in task_relation_ids:
            self.relation_last_trained_stage[rel_id] = stage

    def predict(self, samples: Sequence[RelationSample]) -> Sequence[int]:
        predictions: list[int] = []

        for sample in samples:
            true_rel = sample.relation_id

            # Determine retention probability based on how many tasks ago this relation was trained
            last_stage = self.relation_last_trained_stage.get(true_rel, self.current_training_stage)
            lag = max(0, self.current_training_stage - last_stage)

            # Decay accuracy with lag, bounded below by random guess within seen classes
            min_acc = 1.0 / max(1, self.seen_classes_count)
            target_acc = max(min_acc, self.base_accuracy - (lag * self.forgetting_rate))

            if self.rng.random() < target_acc:
                predictions.append(true_rel)
            else:
                # Random wrong prediction within seen classes
                seen_candidates = [c for c in range(self.seen_classes_count) if c != true_rel]
                if seen_candidates:
                    predictions.append(self.rng.choice(seen_candidates))
                else:
                    predictions.append(true_rel)

        return predictions

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "seen_classes_count": self.seen_classes_count,
            "current_training_stage": self.current_training_stage,
            "relation_last_trained_stage": dict(self.relation_last_trained_stage),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.num_classes = state["num_classes"]
        self.seen_classes_count = state["seen_classes_count"]
        self.current_training_stage = state["current_training_stage"]
        self.relation_last_trained_stage = {
            int(k): v for k, v in state["relation_last_trained_stage"].items()
        }
