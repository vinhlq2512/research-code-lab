from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.io import read_json, write_json


class PerformanceMatrix:
    """Stores and manages the continual learning evaluation performance matrix A[t, j].

    Semantics:
        A[t, j] denotes the evaluation score (e.g. Accuracy or Macro F1) on task j
        after the model has completed training through task t (i.e. tasks 0..t).

    Missing Values:
        Cells that have not been evaluated or represent unobserved future tasks (j > t)
        are stored as `None` (rendered as `null` in JSON and empty string in CSV).
        `None` is strictly distinguished from 0.0.
    """

    def __init__(
        self,
        num_tasks: int,
        metric: str = "accuracy",
        dataset: str = "",
        seed: int | None = None,
        task_order_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")

        self.num_tasks = int(num_tasks)
        self.metric = str(metric)
        self.dataset = str(dataset)
        self.seed = int(seed) if seed is not None else None
        self.task_order_path = str(task_order_path)
        self.metadata = dict(metadata or {})

        # Matrix initialized with None
        self._matrix: list[list[float | None]] = [
            [None for _ in range(self.num_tasks)] for _ in range(self.num_tasks)
        ]

    def _validate_indices(self, trained_until: int, evaluated_task: int, allow_future_eval: bool = False) -> None:
        if not (0 <= trained_until < self.num_tasks):
            raise IndexError(
                f"trained_until index {trained_until} out of bounds [0, {self.num_tasks})"
            )
        if not (0 <= evaluated_task < self.num_tasks):
            raise IndexError(
                f"evaluated_task index {evaluated_task} out of bounds [0, {self.num_tasks})"
            )
        if not allow_future_eval and evaluated_task > trained_until:
            raise ValueError(
                f"evaluated_task ({evaluated_task}) cannot be greater than trained_until ({trained_until}) "
                f"in standard continual learning."
            )

    def set_score(
        self,
        trained_until: int,
        evaluated_task: int,
        score: float,
        allow_future_eval: bool = False,
    ) -> None:
        """Record the performance score A[t, j].

        Args:
            trained_until: Task index t that the model has trained up to.
            evaluated_task: Task index j being evaluated.
            score: Metric value (float).
            allow_future_eval: Whether to allow evaluating future tasks (j > t).
        """
        self._validate_indices(trained_until, evaluated_task, allow_future_eval=allow_future_eval)
        self._matrix[trained_until][evaluated_task] = float(score)

    def get_score(self, trained_until: int, evaluated_task: int) -> float | None:
        """Retrieve score A[t, j].

        Returns:
            Float score or None if not evaluated.
        """
        if not (0 <= trained_until < self.num_tasks) or not (0 <= evaluated_task < self.num_tasks):
            raise IndexError(
                f"Index out of bounds: trained_until={trained_until}, evaluated_task={evaluated_task} "
                f"for matrix of size {self.num_tasks}x{self.num_tasks}"
            )
        return self._matrix[trained_until][evaluated_task]

    def get_row(self, trained_until: int) -> list[float | None]:
        """Return all task scores after training through task t."""
        if not (0 <= trained_until < self.num_tasks):
            raise IndexError(f"trained_until index {trained_until} out of bounds [0, {self.num_tasks})")
        return list(self._matrix[trained_until])

    def get_column(self, evaluated_task: int) -> list[float | None]:
        """Return the trajectory of task j performance across all training stages."""
        if not (0 <= evaluated_task < self.num_tasks):
            raise IndexError(f"evaluated_task index {evaluated_task} out of bounds [0, {self.num_tasks})")
        return [self._matrix[t][evaluated_task] for t in range(self.num_tasks)]

    def as_list(self) -> list[list[float | None]]:
        """Return a copy of the underlying 2D matrix."""
        return [list(row) for row in self._matrix]

    def is_complete(self, trained_until: int | None = None) -> bool:
        """Check if all required lower-triangular cells are evaluated."""
        max_t = self.num_tasks - 1 if trained_until is None else trained_until
        for t in range(max_t + 1):
            for j in range(t + 1):
                if self._matrix[t][j] is None:
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize matrix and metadata to dictionary."""
        data: dict[str, Any] = {
            "dataset": self.dataset,
            "metric": self.metric,
            "num_tasks": self.num_tasks,
            "matrix": self.as_list(),
        }
        if self.seed is not None:
            data["seed"] = self.seed
        if self.task_order_path:
            data["task_order_path"] = self.task_order_path
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceMatrix:
        """Construct PerformanceMatrix from dictionary."""
        num_tasks = int(data["num_tasks"])
        matrix_obj = cls(
            num_tasks=num_tasks,
            metric=str(data.get("metric", "accuracy")),
            dataset=str(data.get("dataset", "")),
            seed=data.get("seed"),
            task_order_path=str(data.get("task_order_path", "")),
            metadata=dict(data.get("metadata", {})),
        )

        raw_matrix = data["matrix"]
        if len(raw_matrix) != num_tasks:
            raise ValueError(f"Matrix row count {len(raw_matrix)} != num_tasks {num_tasks}")

        for t in range(num_tasks):
            row = raw_matrix[t]
            if len(row) != num_tasks:
                raise ValueError(f"Row {t} length {len(row)} != num_tasks {num_tasks}")
            for j in range(num_tasks):
                val = row[j]
                matrix_obj._matrix[t][j] = float(val) if val is not None else None

        return matrix_obj

    def to_csv_string(self) -> str:
        """Format matrix as CSV string with empty strings for missing/None cells."""
        output = io.StringIO()
        writer = csv.writer(output)

        header = ["trained_until"] + [f"task_{j}" for j in range(self.num_tasks)]
        writer.writerow(header)

        for t in range(self.num_tasks):
            row = [t]
            for j in range(self.num_tasks):
                val = self._matrix[t][j]
                row.append(f"{val:.6f}".rstrip("0").rstrip(".") if val is not None else "")
            writer.writerow(row)

        return output.getvalue()

    def export_csv(self, path: Path | str) -> None:
        """Export performance matrix to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(self.to_csv_string())

    def export_json(self, path: Path | str) -> None:
        """Export performance matrix to JSON file."""
        write_json(path, self.to_dict())

    @classmethod
    def load_json(cls, path: Path | str) -> PerformanceMatrix:
        """Load PerformanceMatrix from a JSON file."""
        data = read_json(path)
        return cls.from_dict(data)

    @classmethod
    def load_csv(
        cls,
        path: Path | str,
        metric: str = "accuracy",
        dataset: str = "",
        seed: int | None = None,
    ) -> PerformanceMatrix:
        """Load PerformanceMatrix from a CSV file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            num_tasks = len(header) - 1

            matrix_obj = cls(
                num_tasks=num_tasks,
                metric=metric,
                dataset=dataset,
                seed=seed,
            )

            for row in reader:
                if not row:
                    continue
                t = int(row[0])
                for j in range(num_tasks):
                    cell = row[j + 1].strip()
                    if cell != "":
                        matrix_obj.set_score(t, j, float(cell), allow_future_eval=True)

            return matrix_obj
