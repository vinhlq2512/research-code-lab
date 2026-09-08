from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from utils.io import read_json, write_json


@dataclass(frozen=True)
class TaskAssignment:
    """Assignment of a set of relation names to a specific continual task."""

    task_id: int
    relations: list[str]

    def __post_init__(self) -> None:
        if self.task_id < 0:
            raise ValueError(f"task_id must be non-negative, got {self.task_id}")
        if not self.relations:
            raise ValueError(f"Task {self.task_id} has no relations assigned")
        if len(self.relations) != len(set(self.relations)):
            raise ValueError(f"Task {self.task_id} contains duplicate relations: {self.relations}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "relations": list(self.relations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskAssignment:
        return cls(
            task_id=int(data["task_id"]),
            relations=[str(r) for r in data["relations"]],
        )


@dataclass(frozen=True)
class TaskOrder:
    """Schema and container for a complete, reproducible continual task sequence."""

    dataset: str
    seed: int
    num_tasks: int
    relation_count: int
    tasks: list[TaskAssignment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset name must be non-empty")
        if self.num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {self.num_tasks}")
        if len(self.tasks) != self.num_tasks:
            raise ValueError(
                f"num_tasks ({self.num_tasks}) does not match tasks list length ({len(self.tasks)})"
            )

        # Validate task IDs are 0..num_tasks - 1
        expected_task_ids = list(range(self.num_tasks))
        actual_task_ids = [t.task_id for t in self.tasks]
        if actual_task_ids != expected_task_ids:
            raise ValueError(f"Task IDs must be 0..{self.num_tasks - 1} in order, got: {actual_task_ids}")

        # Validate disjointness across all tasks
        all_relations: list[str] = []
        for t in self.tasks:
            all_relations.extend(t.relations)

        if len(all_relations) != len(set(all_relations)):
            seen = set()
            duplicates = set()
            for r in all_relations:
                if r in seen:
                    duplicates.add(r)
                seen.add(r)
            raise ValueError(f"Duplicate relations across tasks: {sorted(duplicates)}")

        if len(all_relations) != self.relation_count:
            raise ValueError(
                f"relation_count ({self.relation_count}) does not match total assigned relations ({len(all_relations)})"
            )

    def validate_against_expected_relations(
        self,
        expected_relations: Sequence[str],
        expected_dataset: str | None = None,
    ) -> None:
        """Validate that task order precisely matches expected relations and dataset name."""
        if expected_dataset is not None:
            if self.dataset.strip().lower() != expected_dataset.strip().lower():
                raise ValueError(
                    f"Dataset mismatch: expected '{expected_dataset}', but task order specifies '{self.dataset}'"
                )

        expected_set = set(expected_relations)
        actual_set = set(self.all_relations)

        missing = expected_set - actual_set
        if missing:
            raise ValueError(f"Task order is missing {len(missing)} expected relations: {sorted(missing)}")

        unknown = actual_set - expected_set
        if unknown:
            raise ValueError(f"Task order contains {len(unknown)} unknown relations: {sorted(unknown)}")

    @property
    def all_relations(self) -> list[str]:
        """Return flat list of all assigned relations in task order."""
        result: list[str] = []
        for t in self.tasks:
            result.extend(t.relations)
        return result

    def get_task_relations(self, task_id: int) -> list[str]:
        """Return the list of relations assigned to a specific task ID."""
        if not (0 <= task_id < self.num_tasks):
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")
        return list(self.tasks[task_id].relations)

    def get_relation_to_task_id(self) -> dict[str, int]:
        """Return dictionary mapping relation name to its assigned task ID."""
        return {rel: t.task_id for t in self.tasks for rel in t.relations}

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "num_tasks": self.num_tasks,
            "relation_count": self.relation_count,
            "tasks": [t.to_dict() for t in self.tasks],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskOrder:
        tasks = [TaskAssignment.from_dict(t) for t in data["tasks"]]
        return cls(
            dataset=str(data["dataset"]),
            seed=int(data["seed"]),
            num_tasks=int(data["num_tasks"]),
            relation_count=int(data["relation_count"]),
            tasks=tasks,
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: Path | str) -> None:
        """Save task order to JSON file."""
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> TaskOrder:
        """Load task order from JSON file."""
        data = read_json(path)
        return cls.from_dict(data)


def create_task_order(
    dataset: str,
    relations: Sequence[str],
    num_tasks: int,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> TaskOrder:
    """Create a reproducible, deterministic TaskOrder by partitioning relations using a seed.

    Args:
        dataset: Name of dataset (e.g. 'fewrel' or 'tacred').
        relations: List or sequence of relation names.
        num_tasks: Number of continual tasks to partition into.
        seed: Random seed for shuffling.
        metadata: Optional metadata to attach to task order.

    Returns:
        A validated TaskOrder instance.
    """
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be > 0, got {num_tasks}")
    if len(relations) < num_tasks:
        raise ValueError(
            f"Cannot divide {len(relations)} relations into {num_tasks} tasks (need at least 1 per task)"
        )

    # Sort relations first for cross-platform deterministic base order
    base_relations = sorted(relations)

    # Use isolated local RNG
    rng = random.Random(seed)
    shuffled = list(base_relations)
    rng.shuffle(shuffled)

    # Partition into num_tasks cleanly, distributing remainder to early tasks
    total = len(shuffled)
    base_size = total // num_tasks
    remainder = total % num_tasks

    tasks: list[TaskAssignment] = []
    curr_idx = 0
    for task_id in range(num_tasks):
        size = base_size + (1 if task_id < remainder else 0)
        task_rels = shuffled[curr_idx : curr_idx + size]
        curr_idx += size
        tasks.append(TaskAssignment(task_id=task_id, relations=task_rels))

    meta = dict(metadata or {})
    meta["generated_by"] = "src.task_generation.task_order.create_task_order"

    task_order = TaskOrder(
        dataset=dataset.strip().lower(),
        seed=seed,
        num_tasks=num_tasks,
        relation_count=total,
        tasks=tasks,
        metadata=meta,
    )
    return task_order
