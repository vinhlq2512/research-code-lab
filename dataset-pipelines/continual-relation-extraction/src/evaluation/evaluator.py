from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from schemas.relation_sample import RelationSample


class RelationPredictor(Protocol):
    """Minimal protocol for relation extraction classifiers.

    Any model or predictor implementing this protocol can be evaluated
    without the evaluator knowing its neural architecture or implementation.
    """

    def predict(self, samples: Sequence[RelationSample]) -> Sequence[int]:
        """Predict relation IDs for the given relation samples.

        Args:
            samples: Sequence of canonical RelationSample objects.

        Returns:
            Sequence of integer predicted relation IDs matching len(samples).
        """
        ...


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics and predictions for a relation extraction evaluation run."""

    accuracy: float
    macro_f1: float
    sample_count: int
    predictions: list[int]
    labels: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Calculate classification accuracy.

    Args:
        labels: Ground-truth integer labels.
        predictions: Predicted integer labels.

    Returns:
        Accuracy float in [0.0, 1.0].
    """
    if len(labels) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(predictions)} predictions"
        )
    if len(labels) == 0:
        return 0.0

    correct = sum(1 for y, y_hat in zip(labels, predictions) if y == y_hat)
    return float(correct / len(labels))


def compute_macro_f1(
    labels: Sequence[int],
    predictions: Sequence[int],
    target_classes: Sequence[int] | None = None,
) -> float:
    """Calculate unweighted Macro F1 across target classes.

    Args:
        labels: Ground-truth integer labels.
        predictions: Predicted integer labels.
        target_classes: Optional sequence of classes to evaluate over.
                        If None, uses all unique classes present in `labels`.

    Returns:
        Macro-averaged F1 score in [0.0, 1.0].
    """
    if len(labels) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(predictions)} predictions"
        )
    if len(labels) == 0:
        return 0.0

    if target_classes is not None:
        classes = sorted(set(target_classes))
    else:
        classes = sorted(set(labels))

    if not classes:
        return 0.0

    # Count True Positives, False Positives, False Negatives per class
    tp: dict[int, int] = Counter()
    fp: dict[int, int] = Counter()
    fn: dict[int, int] = Counter()

    for y, y_hat in zip(labels, predictions):
        if y == y_hat:
            tp[y] += 1
        else:
            fn[y] += 1
            fp[y_hat] += 1

    f1_scores: list[float] = []
    for c in classes:
        c_tp = tp[c]
        c_fp = fp[c]
        c_fn = fn[c]

        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return float(sum(f1_scores) / len(f1_scores))


class Evaluator:
    """Model-agnostic evaluator for Continual Relation Extraction."""

    def evaluate(
        self,
        predictor: RelationPredictor,
        samples: Sequence[RelationSample],
        target_classes: Sequence[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate a predictor on a sequence of relation samples.

        Args:
            predictor: An object implementing RelationPredictor (exposing .predict()).
            samples: Sequence of canonical RelationSample objects.
            target_classes: Optional sequence of target relation IDs for Macro F1.
            metadata: Optional metadata to attach to the result.

        Returns:
            EvaluationResult containing accuracy, macro F1, counts, and predictions.
        """
        if len(samples) == 0:
            return EvaluationResult(
                accuracy=0.0,
                macro_f1=0.0,
                sample_count=0,
                predictions=[],
                labels=[],
                metadata=dict(metadata or {}),
            )

        labels = [s.relation_id for s in samples]
        raw_predictions = predictor.predict(samples)
        predictions = [int(p) for p in raw_predictions]

        if len(predictions) != len(labels):
            raise ValueError(
                f"Predictor returned {len(predictions)} predictions for {len(labels)} samples"
            )

        acc = compute_accuracy(labels, predictions)
        f1 = compute_macro_f1(labels, predictions, target_classes=target_classes)

        return EvaluationResult(
            accuracy=acc,
            macro_f1=f1,
            sample_count=len(samples),
            predictions=predictions,
            labels=labels,
            metadata=dict(metadata or {}),
        )
