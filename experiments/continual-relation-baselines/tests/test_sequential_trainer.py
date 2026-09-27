from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from schemas.relation_sample import RelationSample
from task_generation.task_builder import ContinualTask

from cl_re_baselines.checkpoint import CheckpointManager
from cl_re_baselines.data_loader import insert_entity_markers, prepare_sample_tokens
from cl_re_baselines.logger import StreamingMetricsLogger
from cl_re_baselines.mock_model import MockRelationClassifier
from cl_re_baselines.sequential_trainer import SequentialFTTrainer


class TestSequentialTrainer(unittest.TestCase):
    def test_entity_marker_insertion_head_first(self) -> None:
        tokens = ["The", "tower", "was", "built", "by", "John", "Douglas", "in", "Chester"]
        # head: "John Douglas" -> tokens[5:7] -> [start=5, end=7)
        # tail: "Chester" -> tokens[8:9] -> [start=8, end=9)
        marked, e1, e2 = insert_entity_markers(tokens, head_start=5, head_end=7, tail_start=8, tail_end=9)

        # Expected order:
        # tokens[0:5] -> "The", "tower", "was", "built", "by"
        # "[E1]", "John", "Douglas", "[/E1]"
        # "in"
        # "[E2]", "Chester", "[/E2]"
        self.assertEqual(marked[5], "[E1]")
        self.assertEqual(marked[8], "[/E1]")
        self.assertEqual(marked[10], "[E2]")
        self.assertEqual(marked[12], "[/E2]")
        self.assertEqual(e1, 5)
        self.assertEqual(e2, 10)

    def test_entity_marker_insertion_tail_first(self) -> None:
        tokens = ["In", "Paris", ",", "Marie", "Curie", "studied", "physics"]
        # tail: "Paris" -> [1:2)
        # head: "Marie Curie" -> [3:5)
        marked, e1, e2 = insert_entity_markers(tokens, head_start=3, head_end=5, tail_start=1, tail_end=2)

        self.assertEqual(marked[1], "[E2]")
        self.assertEqual(marked[3], "[/E2]")
        self.assertEqual(marked[5], "[E1]")
        self.assertEqual(marked[8], "[/E1]")
        self.assertEqual(e2, 1)
        self.assertEqual(e1, 5)

    def test_mock_model_seen_class_masking(self) -> None:
        model = MockRelationClassifier(num_classes=80, seed=42)
        model.set_seen_classes(10)  # Only classes 0..9 visible

        samples = [
            RelationSample(
                sample_id=f"s_{i}",
                tokens=["A", "rel", "B"],
                head_text="A",
                head_start=0,
                head_end=1,
                tail_text="B",
                tail_start=2,
                tail_end=3,
                relation="rel_0",
                relation_id=0,
            )
            for i in range(20)
        ]

        preds = model.predict(samples)
        self.assertEqual(len(preds), 20)
        for p in preds:
            self.assertLess(p, 10, "Prediction must be strictly within seen classes 0..9")

    def test_checkpoint_manager_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_mgr = CheckpointManager(tmp_dir)
            model = MockRelationClassifier(num_classes=80, seed=123)
            model.set_training_stage(stage=2, task_relation_ids=[0, 1, 2])

            saved_dir = ckpt_mgr.save_checkpoint(stage=2, model=model, metadata={"foo": "bar"})
            self.assertTrue(saved_dir.exists())

            # Load into fresh model
            fresh_model = MockRelationClassifier(num_classes=80, seed=999)
            meta = ckpt_mgr.load_checkpoint(stage=2, model=fresh_model)

            self.assertEqual(meta["foo"], "bar")
            self.assertEqual(meta["stage"], 2)
            self.assertEqual(fresh_model.current_training_stage, 2)
            self.assertEqual(fresh_model.relation_last_trained_stage, {0: 2, 1: 2, 2: 2})

    def test_streaming_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = StreamingMetricsLogger(tmp_dir)
            logger.log_evaluation(stage=0, test_task=0, accuracy=0.85, macro_f1=0.84, sample_count=100)
            logger.log_evaluation(stage=1, test_task=0, accuracy=0.60, macro_f1=0.58, sample_count=100)
            logger.log_evaluation(stage=1, test_task=1, accuracy=0.88, macro_f1=0.87, sample_count=100)

            # Check jsonl contents
            lines = (Path(tmp_dir) / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 3)

            rec0 = json.loads(lines[0])
            self.assertEqual(rec0["stage"], 0)
            self.assertEqual(rec0["test_task"], 0)
            self.assertAlmostEqual(rec0["accuracy"], 0.85)

            rec2 = json.loads(lines[2])
            self.assertEqual(rec2["stage"], 1)
            self.assertEqual(rec2["test_task"], 1)
            self.assertAlmostEqual(rec2["accuracy"], 0.88)

    def test_sequential_trainer_two_stage_smoke_run(self) -> None:
        """Smoke test verifying sequential trainer on 2 tasks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"

            # Create synthetic samples for 2 tasks (task 0: relations [0..4], task 1: relations [5..9])
            def make_samples(relations: list[int], split: str, count_per_rel: int = 5) -> list[RelationSample]:
                s_list: list[RelationSample] = []
                for r in relations:
                    for i in range(count_per_rel):
                        s_list.append(
                            RelationSample(
                                sample_id=f"{split}_rel_{r}_{i}",
                                tokens=["EntityA", "interacts", "with", "EntityB"],
                                head_text="EntityA",
                                head_start=0,
                                head_end=1,
                                tail_text="EntityB",
                                tail_start=3,
                                tail_end=4,
                                relation=f"P{r}",
                                relation_id=r,
                            )
                        )
                return s_list

            task_0_rels = [f"P{i}" for i in range(5)]
            task_1_rels = [f"P{i}" for i in range(5, 10)]

            task_0 = ContinualTask(
                task_id=0,
                relations=task_0_rels,
                train_samples=make_samples(list(range(5)), "train", 5),
                validation_samples=make_samples(list(range(5)), "val", 2),
                test_samples=make_samples(list(range(5)), "test", 10),
            )
            task_1 = ContinualTask(
                task_id=1,
                relations=task_1_rels,
                train_samples=make_samples(list(range(5, 10)), "train", 5),
                validation_samples=make_samples(list(range(5, 10)), "val", 2),
                test_samples=make_samples(list(range(5, 10)), "test", 10),
            )

            config = {
                "experiment": {"id": "B0_test"},
                "seed": 2021,
                "dataset": {"name": "fewrel", "train_shot": 5},
                "training": {"batch_size": 8, "epochs_per_task": 1},
                "output": {"results_dir": str(results_dir)},
            }

            model = MockRelationClassifier(num_classes=80, seed=2021)
            trainer = SequentialFTTrainer(
                config=config,
                tasks=[task_0, task_1],
                model=model,
                results_dir=results_dir,
            )

            summary = trainer.run()

            # Verify artifacts generated
            self.assertTrue((results_dir / "performance_matrix.csv").exists())
            self.assertTrue((results_dir / "performance_matrix.json").exists())
            self.assertTrue((results_dir / "metrics.jsonl").exists())
            self.assertTrue((results_dir / "summary.json").exists())
            self.assertTrue((results_dir / "conclusion.md").exists())
            self.assertTrue((results_dir / "checkpoints" / "after_T1").exists())
            self.assertTrue((results_dir / "checkpoints" / "after_T2").exists())

            # Verify exactly 3 evaluation cells (1 + 2 = 3)
            lines = (results_dir / "metrics.jsonl").read_text().strip().split("\n")
            self.assertEqual(len(lines), 3)

            # Check matrix cells
            self.assertIsNotNone(trainer.acc_matrix.get_score(0, 0))
            self.assertIsNotNone(trainer.acc_matrix.get_score(1, 0))
            self.assertIsNotNone(trainer.acc_matrix.get_score(1, 1))
            self.assertIsNone(trainer.acc_matrix.get_score(0, 1))

            # Verify summary contains required keys
            self.assertIn("final_average_accuracy", summary)
            self.assertIn("average_incremental_accuracy", summary)
            self.assertIn("average_forgetting", summary)
            self.assertIn("backward_transfer", summary)


if __name__ == "__main__":
    unittest.main()
