from __future__ import annotations

from typing import Any, Sequence

from schemas.relation_sample import RelationSample

# Special marker tokens for Relation Extraction
E1_START = "[E1]"
E1_END = "[/E1]"
E2_START = "[E2]"
E2_END = "[/E2]"
SPECIAL_MARKERS = [E1_START, E1_END, E2_START, E2_END]


def insert_entity_markers(
    tokens: list[str],
    head_start: int,
    head_end: int,
    tail_start: int,
    tail_end: int,
) -> tuple[list[str], int, int]:
    """Insert entity marker tokens [E1]...[/E1] and [E2]...[/E2] into token sequence.

    Inserts tokens in descending index order so that previous insertions
    do not alter earlier position offsets.

    Args:
        tokens: Original list of word tokens.
        head_start, head_end: Half-open interval [start, end) for head entity.
        tail_start, tail_end: Half-open interval [start, end) for tail entity.

    Returns:
        tuple (marked_tokens, e1_start_idx, e2_start_idx)
    """
    insertions: list[tuple[int, str, str]] = [
        (head_start, E1_START, "e1_start"),
        (head_end, E1_END, "e1_end"),
        (tail_start, E2_START, "e2_start"),
        (tail_end, E2_END, "e2_end"),
    ]

    # Sort descending by token index; if indices match, insert end tag before start tag
    # Priority for tie-breaking: [/E*] before [E*]
    def sort_key(item: tuple[int, str, str]) -> tuple[int, int]:
        idx, tag, tag_type = item
        is_end = 1 if "end" in tag_type else 0
        return (idx, is_end)

    sorted_insertions = sorted(insertions, key=sort_key, reverse=True)

    result_tokens = list(tokens)
    for idx, tag, _ in sorted_insertions:
        result_tokens.insert(idx, tag)

    # Locate new indices of [E1] and [E2]
    e1_idx = result_tokens.index(E1_START)
    e2_idx = result_tokens.index(E2_START)

    return result_tokens, e1_idx, e2_idx


def prepare_sample_tokens(sample: RelationSample) -> tuple[list[str], int, int]:
    """Prepare a RelationSample with inserted entity markers.

    Returns:
        (tokens_with_markers, e1_start_idx, e2_start_idx)
    """
    return insert_entity_markers(
        tokens=sample.tokens,
        head_start=sample.head_start,
        head_end=sample.head_end,
        tail_start=sample.tail_start,
        tail_end=sample.tail_end,
    )


def batch_samples(
    samples: Sequence[RelationSample],
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[list[RelationSample]]:
    """Partition a list of RelationSample objects into batches."""
    sample_list = list(samples)
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(sample_list)

    batches: list[list[RelationSample]] = []
    for i in range(0, len(sample_list), batch_size):
        batches.append(sample_list[i : i + batch_size])
    return batches
