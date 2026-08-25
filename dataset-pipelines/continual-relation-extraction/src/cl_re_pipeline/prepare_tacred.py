from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json, write_jsonl
from .schema import EntitySpan, RelationExample


REQUIRED_SPLITS = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json",
}


def _entity_text(tokens: list[str], start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return ""
    return " ".join(tokens[start : end + 1])


def normalize_item(item: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    tokens = list(item["token"])
    subj_start = item.get("subj_start")
    subj_end = item.get("subj_end")
    obj_start = item.get("obj_start")
    obj_end = item.get("obj_end")
    relation = str(item["relation"])
    uid = str(item.get("id") or f"tacred:{split}:{index}")
    example = RelationExample(
        uid=f"tacred:{uid}",
        dataset="TACRED",
        source_split=split,
        relation=relation,
        tokens=tokens,
        head=EntitySpan(
            text=_entity_text(tokens, subj_start, subj_end),
            start=subj_start,
            end=subj_end,
            entity_type=item.get("subj_type"),
        ),
        tail=EntitySpan(
            text=_entity_text(tokens, obj_start, obj_end),
            start=obj_start,
            end=obj_end,
            entity_type=item.get("obj_type"),
        ),
        metadata={key: item[key] for key in sorted(item) if key not in {"token", "relation"}},
    )
    return example.as_dict()


def prepare(raw_dir: Path, out: Path) -> None:
    missing = [name for name in REQUIRED_SPLITS.values() if not (raw_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing TACRED licensed files in {raw_dir}: {joined}. "
            "Download TACRED from LDC/Stanford and place train.json, dev.json, test.json here."
        )

    rows = []
    split_counts = {}
    for split, filename in REQUIRED_SPLITS.items():
        data = read_json(raw_dir / filename)
        split_counts[split] = len(data)
        rows.extend(normalize_item(item, split, index) for index, item in enumerate(data))
    write_jsonl(out, rows)
    write_json(out.with_suffix(".meta.json"), {"examples": len(rows), "split_counts": split_counts})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize local licensed TACRED data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tacred"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/tacred.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
