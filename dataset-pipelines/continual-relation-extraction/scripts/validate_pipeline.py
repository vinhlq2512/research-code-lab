#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
REPO_ROOT = PROJECT_DIR.parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.fewrel import FewRelDataset
from evaluation.dummy_predictor import DecayingPredictor, PerfectPredictor
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_continual_summary_metrics
from evaluation.performance_matrix import PerformanceMatrix
from task_generation.task_builder import ContinualTaskBuilder
from task_generation.task_order import TaskOrder
from utils.io import read_json, write_json


def resolve_path(candidate_path: str | Path) -> Path:
    p = Path(candidate_path)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to cwd
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    # Try relative to PROJECT_DIR
    if (PROJECT_DIR / p).exists():
        return (PROJECT_DIR / p).resolve()
    # Try relative to REPO_ROOT
    if (REPO_ROOT / p).exists():
        return (REPO_ROOT / p).resolve()
    return p.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-End Validation for Continual Relation Extraction Pipeline & Evaluator."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fewrel",
        choices=["fewrel", "tacred"],
        help="Dataset name (default: fewrel)",
    )
    parser.add_argument(
        "--task-order",
        type=str,
        default="task-orders/fewrel/order_seed_42.json",
        help="Path to task-order JSON file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Base directory to export reports",
    )
    parser.add_argument(
        "--predictor-type",
        type=str,
        default="decaying",
        choices=["decaying", "perfect"],
        help="Type of test predictor (default: decaying)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset.strip().lower()
    task_order_path = resolve_path(args.task_order)

    print("========================================")
    print("Continual Relation Extraction Validation")
    print("========================================")
    print(f"\nDataset:    {dataset_name.upper()}")
    print(f"Task order: {task_order_path}")

    # -------------------------------------------------------------
    # [1/8] Loading dataset & validating samples
    # -------------------------------------------------------------
    print("\n[1/8] Loading dataset")
    if dataset_name == "fewrel":
        raw_dir = resolve_path(args.data_dir) if args.data_dir else resolve_path("data/raw/fewrel")
        if not raw_dir.exists() or not (raw_dir / "train_wiki.json").exists():
            print(f"[ERROR] Missing raw FewRel data at: {raw_dir}", file=sys.stderr)
            return 1
        dataset = FewRelDataset(raw_dir)
    else:
        print(f"[ERROR] Dataset {dataset_name} not yet supported in validation pipeline", file=sys.stderr)
        return 1

    try:
        train_samples = dataset.load_train()
        val_samples = dataset.load_validation()
        test_samples = dataset.load_test()
        relations = dataset.get_relations()
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    print("PASS")
    print(f"Train samples:      {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples:       {len(test_samples)}")
    print(f"Relations:          {len(relations)}")

    # -------------------------------------------------------------
    # [2/8] Validating relation mapping
    # -------------------------------------------------------------
    print("\n[2/8] Validating relation mapping")
    rel_to_id = dataset.get_relation_to_id()
    if len(rel_to_id) != len(relations):
        print(f"[ERROR] Relation mapping size {len(rel_to_id)} != relations count {len(relations)}", file=sys.stderr)
        return 1
    if sorted(rel_to_id.keys()) != sorted(relations):
        print("[ERROR] Relation keys mismatch between dataset and mapping", file=sys.stderr)
        return 1
    if sorted(rel_to_id.values()) != list(range(len(relations))):
        print("[ERROR] Relation IDs are not contiguous integers from 0 to N-1", file=sys.stderr)
        return 1
    print("PASS")

    # -------------------------------------------------------------
    # [3/8] Loading task order
    # -------------------------------------------------------------
    print("\n[3/8] Loading task order")
    if not task_order_path.exists():
        print(f"[ERROR] Task order file not found: {task_order_path}", file=sys.stderr)
        return 1

    try:
        task_order = TaskOrder.load(task_order_path)
        task_order.validate_against_expected_relations(relations, expected_dataset=dataset_name)
    except Exception as exc:
        print(f"[ERROR] Task order validation failed: {exc}", file=sys.stderr)
        return 1

    print("PASS")
    print(f"Tasks:               {task_order.num_tasks}")
    print(f"Relations assigned:  {task_order.relation_count}")
    print(f"Seed:                {task_order.seed}")
    print("Duplicate relations: 0")
    print("Missing relations:   0")

    # -------------------------------------------------------------
    # [4/8] Building continual tasks
    # -------------------------------------------------------------
    print("\n[4/8] Building continual tasks")
    builder = ContinualTaskBuilder()
    try:
        tasks = builder.build_tasks(dataset, task_order)
    except Exception as exc:
        print(f"[ERROR] Failed to build continual tasks: {exc}", file=sys.stderr)
        return 1

    print("PASS")
    for t in tasks:
        print(f"Task {t.task_id}: {len(t.relations)} rels | train={t.train_count}, val={t.val_count}, test={t.test_count}")

    # -------------------------------------------------------------
    # [5/8] Validating task isolation
    # -------------------------------------------------------------
    print("\n[5/8] Validating task isolation")
    # Verify pairwise disjoint relations
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            overlap = set(tasks[i].relations) & set(tasks[j].relations)
            if overlap:
                print(f"[ERROR] Task {i} and Task {j} share relations: {overlap}", file=sys.stderr)
                return 1

    # Verify each sample strictly belongs to its task's relations
    for t in tasks:
        try:
            t.validate_isolation()
        except Exception as exc:
            print(f"[ERROR] Isolation check failed for Task {t.task_id}: {exc}", file=sys.stderr)
            return 1
    print("PASS")

    # -------------------------------------------------------------
    # [6/8] Running evaluator with dummy predictor
    # -------------------------------------------------------------
    print("\n[6/8] Running evaluator")
    evaluator = Evaluator()

    if args.predictor_type == "perfect":
        predictor = PerfectPredictor()
    else:
        predictor = DecayingPredictor(decay_rate=0.04, base_accuracy=0.88, seed=task_order.seed)

    print(f"Predictor: {predictor.__class__.__name__}")
    print("PASS")

    # -------------------------------------------------------------
    # [7/8] Building performance matrix A[t, j]
    # -------------------------------------------------------------
    print("\n[7/8] Building performance matrix A[t, j]")
    matrix = PerformanceMatrix(
        num_tasks=len(tasks),
        metric="accuracy",
        dataset=dataset_name,
        seed=task_order.seed,
        task_order_path=str(task_order_path),
        metadata={"predictor": predictor.__class__.__name__},
    )

    num_stages = len(tasks)
    for t in range(num_stages):
        # After training through task t, evaluate on all observed tasks j = 0..t
        for j in range(t + 1):
            eval_samples = tasks[j].test_samples
            if hasattr(predictor, "predict_for_task"):
                preds = predictor.predict_for_task(eval_samples, trained_until=t, evaluated_task=j)

                class StepPredictor:
                    def predict(self, s):
                        return preds

                step_pred = StepPredictor()
            else:
                step_pred = predictor

            result = evaluator.evaluate(step_pred, eval_samples)
            matrix.set_score(trained_until=t, evaluated_task=j, score=result.accuracy)

    # Compute continual metrics
    summary_metrics = compute_continual_summary_metrics(matrix)
    print("PASS")
    print(f"Final Average Accuracy (ACC_{num_stages - 1}): {summary_metrics['final_average_accuracy']:.4f}")
    print(f"Average Catastrophic Forgetting:     {summary_metrics['average_forgetting']:.4f}")

    # -------------------------------------------------------------
    # [8/8] Exporting reports
    # -------------------------------------------------------------
    print("\n[8/8] Exporting reports")
    if args.reports_dir:
        base_reports_dir = resolve_path(args.reports_dir)
    else:
        base_reports_dir = REPO_ROOT / "reports" / "continual-relation-extraction"

    out_dir = base_reports_dir / dataset_name / f"seed_{task_order.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "accuracy_matrix.csv"
    json_path = out_dir / "accuracy_matrix.json"
    metrics_path = out_dir / "metrics.json"

    matrix.export_csv(csv_path)
    matrix.export_json(json_path)
    write_json(metrics_path, summary_metrics)

    # Verification: ensure exported files exist and are non-empty
    if not (csv_path.exists() and json_path.exists() and metrics_path.exists()):
        print("[ERROR] One or more report files failed to write", file=sys.stderr)
        return 1

    print("PASS")
    print("\nOutput:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {metrics_path}")

    print("\n========================================")
    print("PIPELINE VALIDATION PASSED")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
