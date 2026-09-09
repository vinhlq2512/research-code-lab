from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from datasets.base import ContinualRelationDataset
from schemas.relation_sample import RelationSample
from task_generation.task_order import TaskOrder


@dataclass(frozen=True)
class ContinualTask:
    """Represents a single continual learning task partitioned by relation classes."""

    task_id: int
    relations: list[str]
    train_samples: list[RelationSample]
    validation_samples: list[RelationSample]
    test_samples: list[RelationSample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate_isolation()

    def validate_isolation(self) -> None:
        """Validate that all samples strictly belong to the relations assigned to this task."""
        rel_set = set(self.relations)
        for split_name, samples in [
            ("train", self.train_samples),
            ("validation", self.validation_samples),
            ("test", self.test_samples),
        ]:
            for sample in samples:
                if sample.relation not in rel_set:
                    raise ValueError(
                        f"Relation leakage in Task {self.task_id} ({split_name} split): "
                        f"sample {sample.sample_id} with relation {sample.relation!r} "
                        f"does not belong to task relations {self.relations}"
                    )

    @property
    def train_count(self) -> int:
        return len(self.train_samples)

    @property
    def val_count(self) -> int:
        return len(self.validation_samples)

    @property
    def test_count(self) -> int:
        return len(self.test_samples)

    @property
    def total_count(self) -> int:
        return self.train_count + self.val_count + self.test_count

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "relations": list(self.relations),
            "relation_count": len(self.relations),
            "train_samples": self.train_count,
            "validation_samples": self.val_count,
            "test_samples": self.test_count,
            "total_samples": self.total_count,
        }


class ContinualTaskBuilder:
    """Partitions a ContinualRelationDataset into disjoint ContinualTask instances based on a TaskOrder."""

    def build_tasks(
        self,
        dataset: ContinualRelationDataset,
        task_order: TaskOrder,
        allow_empty_tasks: bool = False,
    ) -> list[ContinualTask]:
        """Build continual learning tasks from dataset splits and task order.

        Args:
            dataset: The source ContinualRelationDataset.
            task_order: The TaskOrder specifying which relation belongs to which task.
            allow_empty_tasks: Whether to allow tasks with 0 samples.

        Returns:
            A list of ContinualTask objects.
        """
        # Validate that task order relations match dataset relations
        dataset_relations = dataset.get_relations()
        task_order.validate_against_expected_relations(dataset_relations)

        # Load all splits
        all_train = dataset.load_train()
        all_val = dataset.load_validation()
        all_test = dataset.load_test()

        # Group samples by relation for efficient class-incremental partitioning
        train_by_rel: dict[str, list[RelationSample]] = defaultdict(list)
        val_by_rel: dict[str, list[RelationSample]] = defaultdict(list)
        test_by_rel: dict[str, list[RelationSample]] = defaultdict(list)

        for s in all_train:
            train_by_rel[s.relation].append(s)
        for s in all_val:
            val_by_rel[s.relation].append(s)
        for s in all_test:
            test_by_rel[s.relation].append(s)

        tasks: list[ContinualTask] = []
        observed_relations: list[str] = []

        for assignment in task_order.tasks:
            t_id = assignment.task_id
            t_rels = assignment.relations

            t_train: list[RelationSample] = []
            t_val: list[RelationSample] = []
            t_test: list[RelationSample] = []

            for rel in t_rels:
                t_train.extend(train_by_rel.get(rel, []))
                t_val.extend(val_by_rel.get(rel, []))
                t_test.extend(test_by_rel.get(rel, []))

            if not allow_empty_tasks:
                if len(t_train) == 0 and len(t_val) == 0 and len(t_test) == 0:
                    raise ValueError(
                        f"Task {t_id} with relations {t_rels} contains 0 samples across all splits."
                    )

            continual_task = ContinualTask(
                task_id=t_id,
                relations=list(t_rels),
                train_samples=t_train,
                validation_samples=t_val,
                test_samples=t_test,
                metadata={"seed": task_order.seed, "dataset": task_order.dataset},
            )
            tasks.append(continual_task)
            observed_relations.extend(t_rels)

        # Global Verification 1: Relation disjointness across tasks
        if len(observed_relations) != len(set(observed_relations)):
            raise ValueError("Task relations are not disjoint across continual tasks!")

        # Global Verification 2: Coverage
        if set(observed_relations) != set(dataset_relations):
            raise ValueError("Union of task relations does not equal dataset relations!")

        # Global Verification 3: Partition completeness (every sample placed in exactly one task)
        task_total_train = sum(t.train_count for t in tasks)
        task_total_val = sum(t.val_count for t in tasks)
        task_total_test = sum(t.test_count for t in tasks)

        if task_total_train != len(all_train):
            raise ValueError(f"Train sample count mismatch: {task_total_train} != {len(all_train)}")
        if task_total_val != len(all_val):
            raise ValueError(f"Validation sample count mismatch: {task_total_val} != {len(all_val)}")
        if task_total_test != len(all_test):
            raise ValueError(f"Test sample count mismatch: {task_total_test} != {len(all_test)}")

        return tasks

    @staticmethod
    def format_task_statistics(tasks: Sequence[ContinualTask]) -> str:
        """Format human-readable task statistics for inspection."""
        lines: list[str] = [
            "========================================",
            "Continual Tasks Summary",
            "========================================",
        ]
        total_train = 0
        total_val = 0
        total_test = 0

        for t in tasks:
            lines.append(f"Task {t.task_id}:")
            lines.append(f"  Relations:          {len(t.relations)} {t.relations[:5]}{'...' if len(t.relations) > 5 else ''}")
            lines.append(f"  Train samples:      {t.train_count}")
            lines.append(f"  Validation samples: {t.val_count}")
            lines.append(f"  Test samples:       {t.test_count}")
            lines.append(f"  Total samples:      {t.total_count}")
            total_train += t.train_count
            total_val += t.val_count
            total_test += t.test_count

        lines.append("----------------------------------------")
        lines.append(f"Total Tasks:          {len(tasks)}")
        lines.append(f"Total Train Samples:  {total_train}")
        lines.append(f"Total Val Samples:    {total_val}")
        lines.append(f"Total Test Samples:   {total_test}")
        lines.append(f"Grand Total Samples:  {total_train + total_val + total_test}")
        lines.append("========================================")
        return "\n".join(lines)
