#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.fewrel import FewRelDataset
from task_generation.task_builder import ContinualTaskBuilder
from task_generation.task_order import TaskOrder


def resolve_path(candidate_path: str | Path) -> Path:
    p = Path(candidate_path)
    if p.is_absolute() and p.exists():
        return p
    if (Path.cwd() / p).exists():
        return Path.cwd() / p
    if (PROJECT_DIR / p).exists():
        return PROJECT_DIR / p
    return Path.cwd() / p


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect continual task partitioning statistics.")
    parser.add_argument("--dataset", type=str, default="fewrel", choices=["fewrel", "tacred"])
    parser.add_argument(
        "--task-order",
        type=str,
        default="task-orders/fewrel/order_seed_42.json",
        help="Path to task order JSON file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw dataset directory",
    )
    args = parser.parse_args()

    task_order_path = resolve_path(args.task_order)
    if not task_order_path.exists():
        print(f"[ERROR] Task order file not found: {task_order_path}", file=sys.stderr)
        return 1

    task_order = TaskOrder.load(task_order_path)

    if args.dataset == "fewrel":
        data_dir = resolve_path(args.data_dir) if args.data_dir else resolve_path("data/raw/fewrel")
        dataset = FewRelDataset(data_dir)
    else:
        print(f"[ERROR] Dataset {args.dataset} not supported yet", file=sys.stderr)
        return 1

    builder = ContinualTaskBuilder()
    try:
        tasks = builder.build_tasks(dataset, task_order)
    except Exception as exc:
        print(f"[ERROR] Failed to build tasks: {exc}", file=sys.stderr)
        return 1

    # Print Formatted Task Statistics
    print(builder.format_task_statistics(tasks))

    # Detailed Task Breakdown
    print("\nDetailed Per-Task Breakdown:")
    for t in tasks:
        print(f"\nTask {t.task_id}:")
        print(f"  Relations:          {len(t.relations)}")
        print(f"  Relation list:      {t.relations}")
        print(f"  Train samples:      {t.train_count}")
        print(f"  Validation samples: {t.val_count}")
        print(f"  Test samples:       {t.test_count}")
        print(f"  Total samples:      {t.total_count}")

    # Programmatic Invariant Verifications
    print("\n----------------------------------------")
    print("Verification Checks:")

    # 1. Check total unique relations
    unique_rels = set()
    for t in tasks:
        unique_rels.update(t.relations)
    dataset_rels = dataset.get_relations()
    rel_match = (len(unique_rels) == len(dataset_rels) == len(task_order.all_relations))
    print(f"  [1] Unique relations ({len(unique_rels)}) == dataset relations ({len(dataset_rels)}): {'PASS' if rel_match else 'FAIL'}")

    # 2. Check pairwise disjointness: Intersection(Task_i, Task_j) == empty
    pairwise_disjoint = True
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            overlap = set(tasks[i].relations) & set(tasks[j].relations)
            if overlap:
                print(f"  [ERROR] Overlap between Task {i} and Task {j}: {overlap}", file=sys.stderr)
                pairwise_disjoint = False
    print(f"  [2] Intersection(Task_i, Task_j) == empty for all pairs: {'PASS' if pairwise_disjoint else 'FAIL'}")

    # 3. Check sample isolation: every sample relation in task relations
    all_isolated = True
    for t in tasks:
        for s in t.train_samples + t.validation_samples + t.test_samples:
            if s.relation not in t.relations:
                all_isolated = False
                break
    print(f"  [3] Relation isolation (no cross-task contamination): {'PASS' if all_isolated else 'FAIL'}")

    passed = rel_match and pairwise_disjoint and all_isolated
    if passed:
        print("\n[PASSED] Continual task builder inspection succeeded.")
        print("========================================")
        return 0
    else:
        print("\n[FAILED] Verification checks failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
