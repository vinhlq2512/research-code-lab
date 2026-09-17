from __future__ import annotations

from typing import Any

from evaluation.performance_matrix import PerformanceMatrix


def compute_average_accuracy(
    matrix: PerformanceMatrix,
    trained_until: int | None = None,
) -> float:
    """Calculate Average Accuracy ACC_t after the model has trained through task t.

    Formula:
        ACC_t = \\frac{1}{t + 1} \\sum_{j=0}^{t} A_{t, j}

    Args:
        matrix: The PerformanceMatrix containing evaluation scores A[t, j].
        trained_until: Training stage t (0..num_tasks - 1).
                       Defaults to the final task (matrix.num_tasks - 1).

    Returns:
        Average accuracy float in [0.0, 1.0].
    """
    t = matrix.num_tasks - 1 if trained_until is None else int(trained_until)
    if not (0 <= t < matrix.num_tasks):
        raise IndexError(f"trained_until stage {t} is out of bounds [0, {matrix.num_tasks})")

    scores: list[float] = []
    for j in range(t + 1):
        score = matrix.get_score(trained_until=t, evaluated_task=j)
        if score is None:
            raise ValueError(
                f"Cannot compute average accuracy at stage {t}: cell A[{t}, {j}] is None"
            )
        scores.append(score)

    return float(sum(scores) / len(scores))


def compute_all_average_accuracies(matrix: PerformanceMatrix) -> list[float]:
    """Calculate Average Accuracy for all completed training stages t = 0..T-1."""
    accuracies: list[float] = []
    for t in range(matrix.num_tasks):
        # Only compute if row is filled up to t
        row_ready = all(matrix.get_score(t, j) is not None for j in range(t + 1))
        if row_ready:
            accuracies.append(compute_average_accuracy(matrix, trained_until=t))
        else:
            break
    return accuracies


def compute_task_forgetting(
    matrix: PerformanceMatrix,
    task_id: int,
    current_task: int | None = None,
) -> float:
    """Calculate catastrophic forgetting F_j for a specific task j.

    Formula:
        F_j = \\max_{l \\in \\{j, \\dots, T-1\\}} A_{l, j} - A_{T, j}

        where T is the current or final evaluation stage.
        If task j has only been trained at stage T (j == T), F_j = 0.0.

    Args:
        matrix: The PerformanceMatrix.
        task_id: The task index j to measure forgetting for.
        current_task: Evaluation stage T. Defaults to final stage (num_tasks - 1).

    Returns:
        Forgetting float (typically >= 0.0).
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    j = int(task_id)

    if not (0 <= j < matrix.num_tasks):
        raise IndexError(f"task_id {j} out of bounds [0, {matrix.num_tasks})")
    if not (0 <= T < matrix.num_tasks):
        raise IndexError(f"current_task stage {T} out of bounds [0, {matrix.num_tasks})")
    if j > T:
        raise ValueError(
            f"task_id {j} cannot be greater than current evaluation stage {T}"
        )

    # If task j was just trained at stage T, it hasn't had subsequent stages to forget
    if j == T:
        return 0.0

    historical_scores: list[float] = []
    for l in range(j, T):
        score = matrix.get_score(trained_until=l, evaluated_task=j)
        if score is None:
            raise ValueError(f"Historical cell A[{l}, {j}] is None during forgetting computation")
        historical_scores.append(score)

    final_score = matrix.get_score(trained_until=T, evaluated_task=j)
    if final_score is None:
        raise ValueError(f"Current stage cell A[{T}, {j}] is None during forgetting computation")

    peak_score = max(historical_scores)
    return float(peak_score - final_score)


def compute_average_forgetting(
    matrix: PerformanceMatrix,
    current_task: int | None = None,
) -> float:
    """Calculate mean catastrophic forgetting across all previously observed tasks.

    Formula:
        Avg_F = \\frac{1}{T} \\sum_{j=0}^{T-1} F_j

    Args:
        matrix: The PerformanceMatrix.
        current_task: Evaluation stage T. Defaults to final stage (num_tasks - 1).

    Returns:
        Average forgetting float.
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    if T == 0:
        return 0.0

    forgettings = [compute_task_forgetting(matrix, task_id=j, current_task=T) for j in range(T)]
    return float(sum(forgettings) / len(forgettings))


def compute_continual_summary_metrics(
    matrix: PerformanceMatrix,
    current_task: int | None = None,
) -> dict[str, Any]:
    """Compute a complete summary of continual learning metrics.

    Returns:
        Dictionary with final_average_accuracy, average_forgetting,
        per_task_forgetting, and average_accuracy_by_stage.
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    acc_T = compute_average_accuracy(matrix, trained_until=T)
    avg_f = compute_average_forgetting(matrix, current_task=T)
    per_task_f = {
        f"task_{j}": compute_task_forgetting(matrix, task_id=j, current_task=T)
        for j in range(T + 1)
    }
    acc_by_stage = compute_all_average_accuracies(matrix)

    return {
        "final_average_accuracy": acc_T,
        "average_forgetting": avg_f,
        "per_task_forgetting": per_task_f,
        "average_accuracy_by_stage": acc_by_stage,
        "evaluated_stage": T,
    }
