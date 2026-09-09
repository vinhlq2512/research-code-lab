from __future__ import annotations

import unittest
from pathlib import Path

from datasets.base import ContinualRelationDataset
from datasets.fewrel import FewRelDataset
from schemas.relation_sample import RelationSample
from task_generation.task_builder import ContinualTask, ContinualTaskBuilder
from task_generation.task_order import TaskAssignment, TaskOrder, create_task_order


class MockDataset(ContinualRelationDataset):
    def __init__(self) -> None:
        self.relations = ["R0", "R1", "R2", "R3"]
        self._train: list[RelationSample] = []
        self._val: list[RelationSample] = []
        self._test: list[RelationSample] = []

        for rel in self.relations:
            # 2 train, 1 val, 1 test per relation
            self._train.append(self._make_sample(rel, "train", 0))
            self._train.append(self._make_sample(rel, "train", 1))
            self._val.append(self._make_sample(rel, "val", 0))
            self._test.append(self._make_sample(rel, "test", 0))

    def _make_sample(self, relation: str, split: str, idx: int) -> RelationSample:
        return RelationSample(
            sample_id=f"mock_{split}_{relation}_{idx}",
            tokens=["Entity1", relation, "Entity2"],
            head_text="Entity1",
            head_start=0,
            head_end=1,
            tail_text="Entity2",
            tail_start=2,
            tail_end=3,
            relation=relation,
            relation_id=self.relations.index(relation),
        )

    def load_train(self) -> list[RelationSample]:
        return list(self._train)

    def load_validation(self) -> list[RelationSample]:
        return list(self._val)

    def load_test(self) -> list[RelationSample]:
        return list(self._test)

    def get_relations(self) -> list[str]:
        return list(self.relations)

    def get_relation_to_id(self) -> dict[str, int]:
        return {r: i for i, r in enumerate(self.relations)}


class TestTaskBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = MockDataset()
        self.builder = ContinualTaskBuilder()

    def test_build_tasks_disjoint_and_complete(self) -> None:
        # 4 relations across 2 tasks
        order = create_task_order("mock", self.dataset.get_relations(), num_tasks=2, seed=42)
        tasks = self.builder.build_tasks(self.dataset, order)

        self.assertEqual(len(tasks), 2)

        # Task 0
        t0 = tasks[0]
        self.assertEqual(len(t0.relations), 2)
        self.assertEqual(t0.train_count, 4)  # 2 relations * 2 train samples
        self.assertEqual(t0.val_count, 2)
        self.assertEqual(t0.test_count, 2)
        self.assertTrue(all(s.relation in t0.relations for s in t0.train_samples))

        # Task 1
        t1 = tasks[1]
        self.assertEqual(len(t1.relations), 2)
        self.assertEqual(t1.train_count, 4)
        self.assertEqual(t1.val_count, 2)
        self.assertEqual(t1.test_count, 2)
        self.assertTrue(all(s.relation in t1.relations for s in t1.train_samples))

        # Disjoint relations
        self.assertEqual(set(t0.relations) & set(t1.relations), set())

        # Total sample check
        self.assertEqual(sum(t.train_count for t in tasks), len(self.dataset.load_train()))

    def test_continual_task_isolation_failure(self) -> None:
        # Manually construct a task with relation leakage
        leak_sample = self.dataset._make_sample("R0", "train", 99)
        with self.assertRaises(ValueError) as ctx:
            ContinualTask(
                task_id=1,
                relations=["R1", "R2"],  # Does NOT include R0
                train_samples=[leak_sample],
                validation_samples=[],
                test_samples=[],
            )
        self.assertIn("Relation leakage", str(ctx.exception))

    def test_empty_task_detected(self) -> None:
        # Task order asks for relation R99 which has 0 samples
        order = TaskOrder(
            dataset="mock",
            seed=42,
            num_tasks=1,
            relation_count=4,
            tasks=[TaskAssignment(task_id=0, relations=["R0", "R1", "R2", "R3"])],
        )
        # Simulate empty dataset
        empty_ds = MockDataset()
        empty_ds._train.clear()
        empty_ds._val.clear()
        empty_ds._test.clear()

        with self.assertRaises(ValueError) as ctx:
            self.builder.build_tasks(empty_ds, order, allow_empty_tasks=False)
        self.assertIn("0 samples across all splits", str(ctx.exception))

    def test_real_fewrel_continual_task_building(self) -> None:
        real_data_dir = Path("data/raw/fewrel")
        order_path = Path("task-orders/fewrel/order_seed_42.json")
        if not (real_data_dir / "train_wiki.json").exists() or not order_path.exists():
            self.skipTest("FewRel raw data or order_seed_42.json missing")

        ds = FewRelDataset(real_data_dir)
        order = TaskOrder.load(order_path)

        tasks = self.builder.build_tasks(ds, order)
        self.assertEqual(len(tasks), 8)

        for t in tasks:
            self.assertEqual(len(t.relations), 10)
            self.assertEqual(t.train_count, 4200)  # 10 * 420
            self.assertEqual(t.val_count, 1400)    # 10 * 140
            self.assertEqual(t.test_count, 1400)   # 10 * 140
            self.assertEqual(t.total_count, 7000)

        # Check total counts
        self.assertEqual(sum(t.train_count for t in tasks), 33600)
        self.assertEqual(sum(t.val_count for t in tasks), 11200)
        self.assertEqual(sum(t.test_count for t in tasks), 11200)


if __name__ == "__main__":
    unittest.main()
