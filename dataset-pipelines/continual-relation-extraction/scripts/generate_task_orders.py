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
from task_generation.task_order import TaskOrder, create_task_order


def resolve_path(candidate_path: str | Path) -> Path:
    p = Path(candidate_path)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to cwd
    if (Path.cwd() / p).exists():
        return Path.cwd() / p
    # Try relative to PROJECT_DIR
    if (PROJECT_DIR / p).exists():
        return PROJECT_DIR / p
    return Path.cwd() / p


def get_relations_for_dataset(dataset_name: str, data_dir: str | Path | None) -> list[str]:
    name = dataset_name.strip().lower()
    if name == "fewrel":
        target_dir = resolve_path(data_dir) if data_dir else resolve_path("data/raw/fewrel")
        ds = FewRelDataset(target_dir)
        return ds.get_relations()
    else:
        raise ValueError(f"Unsupported dataset for relation extraction: '{dataset_name}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic task order files for Continual Relation Extraction.")
    parser.add_argument("--dataset", type=str, default="fewrel", choices=["fewrel", "tacred"], help="Dataset name")
    parser.add_argument("--num-tasks", type=int, default=8, help="Number of continual learning tasks (default: 8)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to generate")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="List of seeds to generate (default: 42 43 44)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory (default: task-orders/<dataset>)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional path to raw dataset directory",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else args.seeds
    dataset_name = args.dataset.strip().lower()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_DIR / "task-orders" / dataset_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print(f"Task Order Generator: {dataset_name.upper()}")
    print(f"Tasks:      {args.num_tasks}")
    print(f"Seeds:      {seeds}")
    print(f"Output dir: {output_dir}")
    print("========================================")

    try:
        relations = get_relations_for_dataset(dataset_name, args.data_dir)
    except Exception as exc:
        print(f"[ERROR] Failed to discover relations: {exc}", file=sys.stderr)
        return 1

    print(f"Discovered {len(relations)} relations.")

    generated_files: list[Path] = []
    for seed in seeds:
        # Reproducibility check: run 1 vs run 2
        order_run1 = create_task_order(dataset_name, relations, args.num_tasks, seed)
        order_run2 = create_task_order(dataset_name, relations, args.num_tasks, seed)

        if order_run1.to_dict() != order_run2.to_dict():
            print(f"[ERROR] Reproducibility failure for seed {seed}! Run 1 != Run 2", file=sys.stderr)
            return 1

        order_file = output_dir / f"order_seed_{seed}.json"
        order_run1.save(order_file)
        generated_files.append(order_file)

        # Verification: load back
        loaded = TaskOrder.load(order_file)
        loaded.validate_against_expected_relations(relations, expected_dataset=dataset_name)

        per_task_sizes = [len(t.relations) for t in loaded.tasks]
        print(f"  [OK] Seed {seed} -> {order_file.name} (task sizes: {per_task_sizes})")

    # If multiple seeds, verify distinctness
    if len(seeds) > 1:
        orders = [TaskOrder.load(f).all_relations for f in generated_files]
        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):
                if orders[i] == orders[j]:
                    print(f"[WARNING] Seed {seeds[i]} and seed {seeds[j]} generated identical order!", file=sys.stderr)
                else:
                    print(f"  [OK] Verified seed {seeds[i]} != seed {seeds[j]}")

    print("\n[PASSED] All task orders generated and verified successfully.")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
