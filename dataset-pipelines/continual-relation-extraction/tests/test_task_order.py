from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task_generation.task_order import (
    TaskAssignment,
    TaskOrder,
    create_task_order,
)


class TestTaskOrder(unittest.TestCase):
    def setUp(self) -> None:
        self.relations = [f"R{i}" for i in range(10)]

    def test_reproducibility_same_seed(self) -> None:
        order1 = create_task_order("fewrel", self.relations, num_tasks=3, seed=42)
        order2 = create_task_order("fewrel", self.relations, num_tasks=3, seed=42)
        self.assertEqual(order1.to_dict(), order2.to_dict())
        self.assertEqual(order1.all_relations, order2.all_relations)

    def test_different_seeds_produce_different_orders(self) -> None:
        order1 = create_task_order("fewrel", self.relations, num_tasks=3, seed=42)
        order2 = create_task_order("fewrel", self.relations, num_tasks=3, seed=43)
        self.assertNotEqual(order1.all_relations, order2.all_relations)

    def test_remainder_partitioning(self) -> None:
        # 10 relations across 3 tasks -> tasks have sizes 4, 3, 3
        order = create_task_order("fewrel", self.relations, num_tasks=3, seed=100)
        sizes = [len(t.relations) for t in order.tasks]
        self.assertEqual(sizes, [4, 3, 3])
        self.assertEqual(sum(sizes), 10)
        self.assertEqual(set(order.all_relations), set(self.relations))

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "order_42.json"
            order = create_task_order("fewrel", self.relations, num_tasks=2, seed=42)
            order.save(file_path)

            loaded = TaskOrder.load(file_path)
            self.assertEqual(order, loaded)

    def test_validation_detects_duplicate_relations(self) -> None:
        tasks = [
            TaskAssignment(task_id=0, relations=["R0", "R1"]),
            TaskAssignment(task_id=1, relations=["R1", "R2"]),  # R1 duplicate
        ]
        with self.assertRaises(ValueError) as ctx:
            TaskOrder(
                dataset="fewrel",
                seed=42,
                num_tasks=2,
                relation_count=4,
                tasks=tasks,
            )
        self.assertIn("Duplicate relations", str(ctx.exception))

    def test_validation_detects_missing_and_unknown_relations(self) -> None:
        order = create_task_order("fewrel", ["R0", "R1", "R2"], num_tasks=2, seed=42)

        # Missing relation R3
        with self.assertRaises(ValueError) as ctx:
            order.validate_against_expected_relations(["R0", "R1", "R2", "R3"])
        self.assertIn("missing", str(ctx.exception))

        # Unknown relation in order
        with self.assertRaises(ValueError) as ctx:
            order.validate_against_expected_relations(["R0", "R1"])
        self.assertIn("unknown", str(ctx.exception))

        # Dataset mismatch
        with self.assertRaises(ValueError) as ctx:
            order.validate_against_expected_relations(["R0", "R1", "R2"], expected_dataset="tacred")
        self.assertIn("Dataset mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
