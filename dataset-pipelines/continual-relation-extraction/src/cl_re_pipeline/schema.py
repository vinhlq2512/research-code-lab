from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntitySpan:
    text: str
    start: int | None = None
    end: int | None = None
    entity_id: str | None = None
    entity_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
        }


@dataclass(frozen=True)
class RelationExample:
    uid: str
    dataset: str
    source_split: str
    relation: str
    tokens: list[str]
    head: EntitySpan
    tail: EntitySpan
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "dataset": self.dataset,
            "source_split": self.source_split,
            "relation": self.relation,
            "tokens": self.tokens,
            "text": self.text if self.text is not None else " ".join(self.tokens),
            "head": self.head.as_dict(),
            "tail": self.tail.as_dict(),
            "metadata": self.metadata,
        }
