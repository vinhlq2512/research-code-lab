from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, resolve_path, sha256_file, write_json, write_jsonl


def group_by_relation(rows: list[dict[str, Any]], include_no_relation: bool) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        relation = str(row["relation"])
        if relation == "no_relation" and not include_no_relation:
            continue
        grouped[relation].append(row)
    return dict(sorted(grouped.items()))


def split_relation_order(relations: list[str], config: dict[str, Any]) -> list[list[str]]:
    rng = random.Random(int(config["seed"]))
    ordered = list(relations)
    rng.shuffle(ordered)
    if config.get("relations_per_task"):
        size = int(config["relations_per_task"])
        return [ordered[index : index + size] for index in range(0, len(ordered), size)]
    num_tasks = int(config["num_tasks"])
    size = math.ceil(len(ordered) / num_tasks)
    return [ordered[index * size : (index + 1) * size] for index in range(num_tasks) if ordered[index * size : (index + 1) * size]]


def _sample_relation(
    relation: str,
    examples: list[dict[str, Any]],
    rng: random.Random,
    train_shots: int,
    val_shots: int,
    max_test_per_relation: int | None,
    allow_undersampled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(examples)
    rng.shuffle(shuffled)
    needed = train_shots + val_shots + 1
    if len(shuffled) < needed and not allow_undersampled:
        raise ValueError(
            f"Relation {relation!r} has {len(shuffled)} examples, needs at least {needed}. "
            "Use lower shots or set allow_undersampled=true."
        )
    train = shuffled[:train_shots]
    val = shuffled[train_shots : train_shots + val_shots]
    test = shuffled[train_shots + val_shots :]
    if max_test_per_relation is not None:
        test = test[:max_test_per_relation]
    return train, val, test


def build(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    base_dir = config_path.parent.parent
    input_path = resolve_path(config["input_jsonl"], base_dir)
    output_dir = resolve_path(config["output_dir"], base_dir)
    rows = read_jsonl(input_path)
    grouped = group_by_relation(rows, bool(config.get("include_no_relation", False)))
    tasks = split_relation_order(list(grouped), config)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(config["seed"]))
    train_shots = int(config["train_shots"])
    val_shots = int(config.get("val_shots", train_shots))
    max_test = config.get("max_test_per_relation")
    max_test_per_relation = None if max_test is None else int(max_test)
    allow_undersampled = bool(config.get("allow_undersampled", False))

    task_summaries = []
    seen_test_rows: list[dict[str, Any]] = []
    output_files: list[Path] = []
    for task_index, relations in enumerate(tasks):
        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []
        for relation in relations:
            train, val, test = _sample_relation(
                relation,
                grouped[relation],
                rng,
                train_shots,
                val_shots,
                max_test_per_relation,
                allow_undersampled,
            )
            train_rows.extend(train)
            val_rows.extend(val)
            test_rows.extend(test)
        seen_test_rows.extend(test_rows)

        task_dir = output_dir / f"task_{task_index:03d}"
        files = {
            "train": task_dir / "train.jsonl",
            "val": task_dir / "val.jsonl",
            "test": task_dir / "test.jsonl",
            "cumulative_test": task_dir / "cumulative_test.jsonl",
        }
        write_jsonl(files["train"], train_rows)
        write_jsonl(files["val"], val_rows)
        write_jsonl(files["test"], test_rows)
        write_jsonl(files["cumulative_test"], seen_test_rows)
        output_files.extend(files.values())
        task_summaries.append(
            {
                "task_index": task_index,
                "relations": relations,
                "counts": {
                    "train": len(train_rows),
                    "val": len(val_rows),
                    "test": len(test_rows),
                    "cumulative_test": len(seen_test_rows),
                },
            }
        )

    relation_order_path = output_dir / "relation_order.json"
    write_json(relation_order_path, {"tasks": tasks, "flat_order": [relation for task in tasks for relation in task]})
    output_files.append(relation_order_path)

    manifest = {
        "dataset": config["dataset"],
        "config_path": str(config_path),
        "config": config,
        "input_jsonl": str(input_path),
        "input_sha256": sha256_file(input_path),
        "num_examples": len(rows),
        "num_relations": len(grouped),
        "num_tasks": len(tasks),
        "task_summaries": task_summaries,
        "output_sha256": {str(path.relative_to(output_dir)): sha256_file(path) for path in output_files},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic continual relation extraction tasks.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args.config)
    print(f"Wrote {manifest['num_tasks']} tasks to {manifest['config']['output_dir']}")


if __name__ == "__main__":
    main()
