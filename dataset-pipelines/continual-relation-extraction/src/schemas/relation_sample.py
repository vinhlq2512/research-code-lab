from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RelationSample:
    """Canonical representation of a relation extraction instance.

    Span Semantics:
        Entity spans follow half-open interval semantics: [start, end)
        - `head_start`: 0-indexed token index of entity start (inclusive)
        - `head_end`: 0-indexed token index of entity end (exclusive)
        - Slice tokens as: tokens[head_start:head_end]

    Invariants:
        - 0 <= head_start < head_end <= len(tokens)
        - 0 <= tail_start < tail_end <= len(tokens)
        - relation is not empty
        - relation_id >= 0
    """

    sample_id: str
    tokens: list[str]

    head_text: str
    head_start: int
    head_end: int

    tail_text: str
    tail_start: int
    tail_end: int

    relation: str
    relation_id: int

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise ValueError(f"sample_id must be a non-empty string, got: {self.sample_id!r}")

        if not isinstance(self.tokens, list):
            raise ValueError(f"tokens must be a list of strings, got: {type(self.tokens)}")

        num_tokens = len(self.tokens)
        if num_tokens == 0:
            raise ValueError(f"tokens cannot be empty for sample_id: {self.sample_id}")

        # Validate head span: [start, end)
        if not (0 <= self.head_start < self.head_end <= num_tokens):
            raise ValueError(
                f"Invalid head span [{self.head_start}, {self.head_end}) for tokens of length {num_tokens} "
                f"in sample_id: {self.sample_id}"
            )

        # Validate tail span: [start, end)
        if not (0 <= self.tail_start < self.tail_end <= num_tokens):
            raise ValueError(
                f"Invalid tail span [{self.tail_start}, {self.tail_end}) for tokens of length {num_tokens} "
                f"in sample_id: {self.sample_id}"
            )

        if not self.relation or not isinstance(self.relation, str):
            raise ValueError(f"relation must be a non-empty string, got: {self.relation!r}")

        if not isinstance(self.relation_id, int) or self.relation_id < 0:
            raise ValueError(f"relation_id must be an integer >= 0, got: {self.relation_id!r}")

    @property
    def head_tokens(self) -> list[str]:
        """Return the list of tokens belonging to the head entity."""
        return self.tokens[self.head_start : self.head_end]

    @property
    def tail_tokens(self) -> list[str]:
        """Return the list of tokens belonging to the tail entity."""
        return self.tokens[self.tail_start : self.tail_end]

    def to_dict(self) -> dict[str, Any]:
        """Serialize sample to dictionary."""
        return {
            "sample_id": self.sample_id,
            "tokens": list(self.tokens),
            "head_text": self.head_text,
            "head_start": self.head_start,
            "head_end": self.head_end,
            "tail_text": self.tail_text,
            "tail_start": self.tail_start,
            "tail_end": self.tail_end,
            "relation": self.relation,
            "relation_id": self.relation_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationSample:
        """Construct a RelationSample from dictionary representation."""
        return cls(
            sample_id=str(data["sample_id"]),
            tokens=list(data["tokens"]),
            head_text=str(data["head_text"]),
            head_start=int(data["head_start"]),
            head_end=int(data["head_end"]),
            tail_text=str(data["tail_text"]),
            tail_start=int(data["tail_start"]),
            tail_end=int(data["tail_end"]),
            relation=str(data["relation"]),
            relation_id=int(data["relation_id"]),
            metadata=dict(data.get("metadata", {})),
        )
