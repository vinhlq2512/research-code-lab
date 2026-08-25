from __future__ import annotations

import argparse
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import certifi

from .io_utils import read_json, write_json, write_jsonl
from .schema import EntitySpan, RelationExample


FEWREL_URLS = {
    "train_wiki.json": "https://raw.githubusercontent.com/thunlp/FewRel/master/data/train_wiki.json",
    "val_wiki.json": "https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json",
    "pid2name.json": "https://raw.githubusercontent.com/thunlp/FewRel/master/data/pid2name.json",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "cl-re-pipeline/0.1"})
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        path.write_bytes(response.read())


def _span_from_fewrel(entity: list[Any]) -> EntitySpan:
    name = str(entity[0])
    entity_id = str(entity[1]) if len(entity) > 1 else None
    positions = entity[2] if len(entity) > 2 else []
    first_span = positions[0] if positions else []
    start = int(first_span[0]) if len(first_span) > 0 else None
    end = int(first_span[-1]) if len(first_span) > 0 else None
    return EntitySpan(text=name, start=start, end=end, entity_id=entity_id)


def normalize_split(dataset: dict[str, list[dict[str, Any]]], split: str, rel_info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for relation, examples in sorted(dataset.items()):
        for index, item in enumerate(examples):
            example = RelationExample(
                uid=f"fewrel:{split}:{relation}:{index}",
                dataset="FewRel",
                source_split=split,
                relation=relation,
                tokens=list(item["tokens"]),
                head=_span_from_fewrel(item["h"]),
                tail=_span_from_fewrel(item["t"]),
                metadata={"relation_info": rel_info.get(relation, [])},
            )
            rows.append(example.as_dict())
    return rows


def prepare(raw_dir: Path, out: Path) -> None:
    for filename, url in FEWREL_URLS.items():
        download(url, raw_dir / filename)

    rel_info = read_json(raw_dir / "pid2name.json")
    all_rows = []
    all_rows.extend(normalize_split(read_json(raw_dir / "train_wiki.json"), "train_wiki", rel_info))
    all_rows.extend(normalize_split(read_json(raw_dir / "val_wiki.json"), "val_wiki", rel_info))
    write_jsonl(out, all_rows)
    write_json(out.with_suffix(".meta.json"), {"source_urls": FEWREL_URLS, "examples": len(all_rows)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize public FewRel data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/fewrel"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/fewrel.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
