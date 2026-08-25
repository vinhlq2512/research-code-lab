from __future__ import annotations

import json
from pathlib import Path

from cl_re_pipeline.build_tasks import build
from cl_re_pipeline.io_utils import read_json, read_jsonl, write_json, write_jsonl


def make_rows() -> list[dict]:
    rows = []
    for relation_index in range(6):
        relation = f"rel_{relation_index}"
        for example_index in range(8):
            rows.append(
                {
                    "uid": f"{relation}:{example_index}",
                    "dataset": "fixture",
                    "source_split": "all",
                    "relation": relation,
                    "tokens": ["head", relation, "tail"],
                    "text": f"head {relation} tail",
                    "head": {"text": "head", "start": 0, "end": 0, "entity_id": None, "entity_type": None},
                    "tail": {"text": "tail", "start": 2, "end": 2, "entity_id": None, "entity_type": None},
                    "metadata": {},
                }
            )
    return rows


def test_build_tasks_is_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "fixture.jsonl"
    config_path = tmp_path / "configs" / "fixture.json"
    output_dir = tmp_path / "tasks"
    write_jsonl(input_path, make_rows())
    write_json(
        config_path,
        {
            "dataset": "fixture",
            "input_jsonl": str(input_path),
            "output_dir": str(output_dir),
            "seed": 2021,
            "num_tasks": 3,
            "train_shots": 2,
            "val_shots": 1,
            "include_no_relation": False,
            "allow_undersampled": False,
            "max_test_per_relation": 2,
        },
    )

    first = build(config_path)
    first_order = read_json(output_dir / "relation_order.json")
    first_manifest = json.dumps(first["output_sha256"], sort_keys=True)
    second = build(config_path)

    assert read_json(output_dir / "relation_order.json") == first_order
    assert json.dumps(second["output_sha256"], sort_keys=True) == first_manifest
    assert second["num_tasks"] == 3
    assert len(read_jsonl(output_dir / "task_000" / "train.jsonl")) == 4
    assert len(read_jsonl(output_dir / "task_000" / "val.jsonl")) == 2
    assert len(read_jsonl(output_dir / "task_000" / "test.jsonl")) == 4


def test_no_relation_excluded_by_default(tmp_path: Path) -> None:
    rows = make_rows()
    rows.append({**rows[0], "uid": "no_relation:0", "relation": "no_relation"})
    input_path = tmp_path / "fixture.jsonl"
    config_path = tmp_path / "configs" / "fixture.json"
    output_dir = tmp_path / "tasks"
    write_jsonl(input_path, rows)
    write_json(
        config_path,
        {
            "dataset": "fixture",
            "input_jsonl": str(input_path),
            "output_dir": str(output_dir),
            "seed": 7,
            "num_tasks": 2,
            "train_shots": 1,
            "val_shots": 1,
            "include_no_relation": False,
            "allow_undersampled": False,
            "max_test_per_relation": 1,
        },
    )

    manifest = build(config_path)

    assert manifest["num_relations"] == 6
    relation_order = read_json(output_dir / "relation_order.json")
    assert "no_relation" not in relation_order["flat_order"]
