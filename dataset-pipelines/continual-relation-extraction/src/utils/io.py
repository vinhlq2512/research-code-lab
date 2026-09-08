from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path | str) -> Any:
    """Read a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, value: Any, indent: int = 2) -> None:
    """Write a JSON file with safe directory creation and deterministic sort_keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to a JSON Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
