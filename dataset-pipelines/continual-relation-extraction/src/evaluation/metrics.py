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


def compute_all_average_accuracies(
    matrix: PerformanceMatrix,
    max_stage: int | None = None,
) -> list[float]:
    """Calculate Average Accuracy for all completed training stages t = 0..max_stage.

    Args:
        matrix: The PerformanceMatrix.
        max_stage: Optional upper bound stage (inclusive). Defaults to num_tasks - 1.

    Returns:
        List of float average accuracies [ACC_0, ACC_1, ..., ACC_max_stage].
    """
    limit = matrix.num_tasks - 1 if max_stage is None else int(max_stage)
    if not (0 <= limit < matrix.num_tasks):
        raise IndexError(f"max_stage {limit} is out of bounds [0, {matrix.num_tasks})")

    accuracies: list[float] = []
    for t in range(limit + 1):
        row_ready = all(matrix.get_score(t, j) is not None for j in range(t + 1))
        if row_ready:
            accuracies.append(compute_average_accuracy(matrix, trained_until=t))
        else:
            break
    return accuracies


def compute_average_incremental_accuracy(
    matrix: PerformanceMatrix,
    current_task: int | None = None,
) -> float:
    """Calculate Average Incremental Accuracy (AIA) across all completed training stages.

    Formula:
        AIA = \\frac{1}{T + 1} \\sum_{t=0}^{T} ACC_t

    Args:
        matrix: The PerformanceMatrix.
        current_task: Current or final training stage T (0..num_tasks - 1).
                      Defaults to matrix.num_tasks - 1.

    Returns:
        Average Incremental Accuracy float in [0.0, 1.0].
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    if not (0 <= T < matrix.num_tasks):
        raise IndexError(f"current_task {T} is out of bounds [0, {matrix.num_tasks})")

    stage_accuracies = compute_all_average_accuracies(matrix, max_stage=T)
    if len(stage_accuracies) != T + 1:
        raise ValueError(
            f"Cannot compute AIA up to stage {T}: only {len(stage_accuracies)} of {T + 1} stages complete"
        )

    return float(sum(stage_accuracies) / len(stage_accuracies))


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
        current_task: Evaluation stage T (0..num_tasks - 1). Defaults to final stage.

    Returns:
        Average forgetting float.
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    if T == 0:
        return 0.0

    forgettings = [compute_task_forgetting(matrix, task_id=j, current_task=T) for j in range(T)]
    return float(sum(forgettings) / len(forgettings))


def compute_backward_transfer(
    matrix: PerformanceMatrix,
    current_task: int | None = None,
) -> float:
    """Calculate Backward Transfer (BWT) across all previously observed tasks.

    Formula:
        BWT = \\frac{1}{T} \\sum_{j=0}^{T-1} (A_{T, j} - A_{j, j})

        where A_{j, j} is the initial performance immediately after learning task j,
        and A_{T, j} is the performance on task j after learning task T.

    Interpretation:
        BWT < 0: Performance on old tasks decreased due to backward interference (forgetting).
        BWT = 0: No backward interference.
        BWT > 0: Positive backward transfer (learning new tasks improved old tasks).

    Args:
        matrix: The PerformanceMatrix.
        current_task: Evaluation stage T (0..num_tasks - 1). Defaults to final stage.

    Returns:
        Backward transfer float score.
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    if not (0 <= T < matrix.num_tasks):
        raise IndexError(f"current_task stage {T} out of bounds [0, {matrix.num_tasks})")
    if T == 0:
        return 0.0

    transfers: list[float] = []
    for j in range(T):
        initial_score = matrix.get_score(trained_until=j, evaluated_task=j)
        if initial_score is None:
            raise ValueError(f"Initial score A[{j}, {j}] is None during BWT computation")

        current_score = matrix.get_score(trained_until=T, evaluated_task=j)
        if current_score is None:
            raise ValueError(f"Current stage score A[{T}, {j}] is None during BWT computation")

        transfers.append(current_score - initial_score)

    return float(sum(transfers) / len(transfers))


def compute_old_and_new_accuracies(
    matrix: PerformanceMatrix,
    current_task: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Old-task vs New-task accuracy breakdown across completed stages t >= 1.

    Formula:
        Old_t = \\frac{1}{t} \\sum_{j=0}^{t-1} A_{t, j}
        New_t = A_{t, t}

    Args:
        matrix: The PerformanceMatrix.
        current_task: Evaluation stage limit T (1..num_tasks - 1). Defaults to final stage.

    Returns:
        Dictionary mapping stage name (e.g. 'stage_1') to:
            {'old_acc': float, 'new_acc': float}
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    if not (0 <= T < matrix.num_tasks):
        raise IndexError(f"current_task stage {T} out of bounds [0, {matrix.num_tasks})")

    breakdown: dict[str, dict[str, float]] = {}
    for t in range(1, T + 1):
        old_scores: list[float] = []
        for j in range(t):
            s = matrix.get_score(trained_until=t, evaluated_task=j)
            if s is None:
                raise ValueError(f"Cell A[{t}, {j}] is None during old-task accuracy computation")
            old_scores.append(s)

        new_score = matrix.get_score(trained_until=t, evaluated_task=t)
        if new_score is None:
            raise ValueError(f"New task cell A[{t}, {t}] is None during new-task accuracy computation")

        old_acc = float(sum(old_scores) / len(old_scores))
        breakdown[f"stage_{t}"] = {
            "old_acc": old_acc,
            "new_acc": float(new_score),
        }

    return breakdown


def identify_most_forgotten_task(
    matrix: PerformanceMatrix,
    current_task: int | None = None,
) -> dict[str, Any]:
    """Identify the previously observed task with the greatest performance degradation.

    Args:
        matrix: The PerformanceMatrix.
        current_task: Current or final evaluation stage T. Defaults to final stage.

    Returns:
        Dictionary containing:
            task_id: int index of most forgotten task (or None if T=0)
            task_name: 'T{task_id + 1}' string (or None if T=0)
            forgetting: float maximum drop
            initial_score: float peak/initial score
            final_score: float score at stage T
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    if not (0 <= T < matrix.num_tasks):
        raise IndexError(f"current_task stage {T} out of bounds [0, {matrix.num_tasks})")
    if T == 0:
        return {
            "task_id": None,
            "task_name": None,
            "forgetting": 0.0,
            "initial_score": matrix.get_score(0, 0),
            "final_score": matrix.get_score(0, 0),
        }

    max_drop = -1.0
    most_forgotten_id = 0

    for j in range(T):
        drop = compute_task_forgetting(matrix, task_id=j, current_task=T)
        if drop > max_drop:
            max_drop = drop
            most_forgotten_id = j

    initial_val = matrix.get_score(most_forgotten_id, most_forgotten_id)
    final_val = matrix.get_score(T, most_forgotten_id)

    return {
        "task_id": most_forgotten_id,
        "task_name": f"T{most_forgotten_id + 1}",
        "forgetting": float(max_drop),
        "initial_score": initial_val,
        "final_score": final_val,
    }


def compute_continual_summary_metrics(
    matrix: PerformanceMatrix,
    current_task: int | None = None,
) -> dict[str, Any]:
    """Compute a complete summary of continual learning metrics.

    Returns:
        Dictionary with final_average_accuracy, average_incremental_accuracy,
        average_forgetting, backward_transfer, most_forgotten_task,
        per_task_forgetting, old_vs_new_by_stage, and average_accuracy_by_stage.
    """
    T = matrix.num_tasks - 1 if current_task is None else int(current_task)
    acc_T = compute_average_accuracy(matrix, trained_until=T)
    aia = compute_average_incremental_accuracy(matrix, current_task=T)
    avg_f = compute_average_forgetting(matrix, current_task=T)
    bwt = compute_backward_transfer(matrix, current_task=T)
    most_forgotten = identify_most_forgotten_task(matrix, current_task=T)
    old_new = compute_old_and_new_accuracies(matrix, current_task=T) if T > 0 else {}

    per_task_f = {
        f"task_{j}": compute_task_forgetting(matrix, task_id=j, current_task=T)
        for j in range(T + 1)
    }
    acc_by_stage = compute_all_average_accuracies(matrix, max_stage=T)

    incremental_acc_dict = {
        f"T{t + 1}": acc_by_stage[t] for t in range(len(acc_by_stage))
    }

    return {
        "final_average_accuracy": acc_T,
        "average_incremental_accuracy": aia,
        "average_forgetting": avg_f,
        "backward_transfer": bwt,
        "most_forgotten_task": most_forgotten,
        "per_task_forgetting": per_task_f,
        "old_vs_new_by_stage": old_new,
        "incremental_accuracy": incremental_acc_dict,
        "average_accuracy_by_stage": acc_by_stage,
        "evaluated_stage": T,
    }
