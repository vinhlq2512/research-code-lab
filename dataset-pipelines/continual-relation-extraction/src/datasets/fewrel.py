from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.base import ContinualRelationDataset
from schemas.relation_sample import RelationSample
from utils.io import read_json, write_json


class FewRelDataset(ContinualRelationDataset):
    """FewRel dataset loader producing canonical RelationSample objects.

    FewRel contains 80 relation types in total (64 from train_wiki.json and 16 from val_wiki.json),
    each with 700 instances.

    Span Normalization:
        FewRel positions are provided as 0-indexed token indices (e.g. [[13, 14]]).
        These are normalized to half-open interval [start, end) where:
            start = positions[0]
            end = positions[-1] + 1
        For instances with multiple mention spans in the sentence, the first mention
        is chosen as the primary canonical span, while all alternative spans are
        faithfully preserved in metadata["head_all_mentions"] / metadata["tail_all_mentions"].
    """

    def __init__(
        self,
        data_dir: Path | str,
        train_val_test_counts: tuple[int, int, int] = (420, 140, 140),
    ) -> None:
        """Initialize FewRel dataset loader.

        Args:
            data_dir: Path to directory containing raw FewRel JSON files
                      (train_wiki.json, val_wiki.json, and pid2name.json).
            train_val_test_counts: Tuple specifying (train, val, test) sample counts
                                  per relation. Default is (420, 140, 140) which sums to 700.
        """
        self.data_dir = Path(data_dir)
        self.split_counts = train_val_test_counts

        self._train_samples: list[RelationSample] | None = None
        self._val_samples: list[RelationSample] | None = None
        self._test_samples: list[RelationSample] | None = None

        self._raw_data: dict[str, list[dict[str, Any]]] | None = None
        self._pid2name: dict[str, Any] | None = None
        self._relations: list[str] | None = None
        self._relation_to_id: dict[str, int] | None = None

    def _ensure_data_loaded(self) -> None:
        if self._raw_data is not None:
            return

        train_file = self.data_dir / "train_wiki.json"
        val_file = self.data_dir / "val_wiki.json"
        pid_file = self.data_dir / "pid2name.json"

        if not train_file.exists():
            raise FileNotFoundError(f"Missing train_wiki.json at {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Missing val_wiki.json at {val_file}")

        train_raw: dict[str, list[dict[str, Any]]] = read_json(train_file)
        val_raw: dict[str, list[dict[str, Any]]] = read_json(val_file)

        merged: dict[str, list[dict[str, Any]]] = {}
        merged.update(train_raw)
        merged.update(val_raw)
        self._raw_data = merged

        if pid_file.exists():
            self._pid2name = read_json(pid_file)
        else:
            self._pid2name = {}

        # Deterministic relation ordering: sorted alphabetically
        self._relations = sorted(self._raw_data.keys())
        self._relation_to_id = {rel: idx for idx, rel in enumerate(self._relations)}

    def get_relations(self) -> list[str]:
        self._ensure_data_loaded()
        assert self._relations is not None
        return list(self._relations)

    def get_relation_to_id(self) -> dict[str, int]:
        self._ensure_data_loaded()
        assert self._relation_to_id is not None
        return dict(self._relation_to_id)

    def save_relation_mapping(self, path: Path | str) -> None:
        """Persist the deterministic relation-to-id mapping to disk."""
        mapping = self.get_relation_to_id()
        write_json(path, mapping)

    def _parse_entity_span(self, entity_data: list[Any], num_tokens: int) -> tuple[int, int]:
        """Convert FewRel entity representation to half-open [start, end) span."""
        # entity_data structure: [text, entity_id, positions_list]
        positions = entity_data[2] if len(entity_data) > 2 else []
        if not positions or not positions[0]:
            raise ValueError(f"Entity missing valid position data: {entity_data}")

        first_span = positions[0]
        start = int(first_span[0])
        end = int(first_span[-1]) + 1  # convert to half-open [start, end)

        if not (0 <= start < end <= num_tokens):
            raise ValueError(
                f"Computed entity span [{start}, {end}) is out of bounds for {num_tokens} tokens"
            )
        return start, end

    def _build_samples(self) -> None:
        self._ensure_data_loaded()
        assert self._raw_data is not None
        assert self._relation_to_id is not None
        assert self._pid2name is not None

        train_count, val_count, test_count = self.split_counts
        train_samples: list[RelationSample] = []
        val_samples: list[RelationSample] = []
        test_samples: list[RelationSample] = []

        for relation in self.get_relations():
            rel_id = self._relation_to_id[relation]
            items = self._raw_data[relation]
            rel_meta = self._pid2name.get(relation, [])
            rel_name = rel_meta[0] if len(rel_meta) > 0 else relation
            rel_desc = rel_meta[1] if len(rel_meta) > 1 else ""

            for idx, item in enumerate(items):
                tokens = list(item["tokens"])
                num_tokens = len(tokens)
                h = item["h"]
                t = item["t"]

                head_text = str(h[0])
                head_start, head_end = self._parse_entity_span(h, num_tokens)

                tail_text = str(t[0])
                tail_start, tail_end = self._parse_entity_span(t, num_tokens)

                # Determine split deterministically by index
                if idx < train_count:
                    split_name = "train"
                    split_idx = idx
                elif idx < train_count + val_count:
                    split_name = "val"
                    split_idx = idx - train_count
                else:
                    split_name = "test"
                    split_idx = idx - (train_count + val_count)

                sample_id = f"fewrel_{split_name}_{relation}_{split_idx}"
                metadata = {
                    "source_split": split_name,
                    "head_entity_id": h[1] if len(h) > 1 else None,
                    "tail_entity_id": t[1] if len(t) > 1 else None,
                    "head_all_mentions": h[2] if len(h) > 2 else [],
                    "tail_all_mentions": t[2] if len(t) > 2 else [],
                    "relation_name": rel_name,
                    "relation_desc": rel_desc,
                }

                sample = RelationSample(
                    sample_id=sample_id,
                    tokens=tokens,
                    head_text=head_text,
                    head_start=head_start,
                    head_end=head_end,
                    tail_text=tail_text,
                    tail_start=tail_start,
                    tail_end=tail_end,
                    relation=relation,
                    relation_id=rel_id,
                    metadata=metadata,
                )

                if split_name == "train":
                    train_samples.append(sample)
                elif split_name == "val":
                    val_samples.append(sample)
                else:
                    test_samples.append(sample)

        self._train_samples = train_samples
        self._val_samples = val_samples
        self._test_samples = test_samples

    def load_train(self) -> list[RelationSample]:
        if self._train_samples is None:
            self._build_samples()
        assert self._train_samples is not None
        return list(self._train_samples)

    def load_validation(self) -> list[RelationSample]:
        if self._val_samples is None:
            self._build_samples()
        assert self._val_samples is not None
        return list(self._val_samples)

    def load_test(self) -> list[RelationSample]:
        if self._test_samples is None:
            self._build_samples()
        assert self._test_samples is not None
        return list(self._test_samples)
