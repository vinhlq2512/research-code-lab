"""Generic evaluator, performance matrix A[t,j], and continual learning metrics."""

from .dummy_predictor import (
    ConstantPredictor,
    DecayingPredictor,
    PerfectPredictor,
    RandomPredictor,
)
from .evaluator import (
    EvaluationResult,
    Evaluator,
    RelationPredictor,
    compute_accuracy,
    compute_macro_f1,
)
from .metrics import (
    compute_all_average_accuracies,
    compute_average_accuracy,
    compute_average_forgetting,
    compute_continual_summary_metrics,
    compute_task_forgetting,
)
from .performance_matrix import PerformanceMatrix

__all__ = [
    "RelationPredictor",
    "EvaluationResult",
    "Evaluator",
    "compute_accuracy",
    "compute_macro_f1",
    "PerformanceMatrix",
    "compute_average_accuracy",
    "compute_all_average_accuracies",
    "compute_task_forgetting",
    "compute_average_forgetting",
    "compute_continual_summary_metrics",
    "PerfectPredictor",
    "ConstantPredictor",
    "RandomPredictor",
    "DecayingPredictor",
]
