from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.base import ContinualRelationDataset
from schemas.relation_sample import RelationSample
from utils.io import read_json, write_json

TACRED_REQUIRED_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json",
}

# Canonical TACRED relation list (42 relations: 41 positive relations + no_relation)
TACRED_STANDARD_RELATIONS = [
    "no_relation",
    "org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence",
    "per:title",
]


def normalize_tacred_record(
    item: dict[str, Any],
    relation_to_id: dict[str, int],
    split_name: str = "train",
    sample_index: int = 0,
) -> RelationSample:
    """Normalize a raw TACRED record into a canonical RelationSample.

    Span Semantics:
        TACRED raw positions (`subj_start`, `subj_end`, `obj_start`, `obj_end`)
        are 0-indexed INCLUSIVE intervals [start, end].
        They are normalized to half-open intervals [start, end + 1) in RelationSample.
    """
    tokens = list(item["token"])
    num_tokens = len(tokens)

    subj_start = int(item["subj_start"])
    subj_end_incl = int(item["subj_end"])
    head_start = subj_start
    head_end = subj_end_incl + 1  # Convert [start, end] -> [start, end)

    obj_start = int(item["obj_start"])
    obj_end_incl = int(item["obj_end"])
    tail_start = obj_start
    tail_end = obj_end_incl + 1  # Convert [start, end] -> [start, end)

    head_text = " ".join(tokens[head_start:head_end])
    tail_text = " ".join(tokens[tail_start:tail_end])

    relation = str(item["relation"])
    if relation not in relation_to_id:
        raise ValueError(f"Unknown TACRED relation: {relation!r}")
    relation_id = relation_to_id[relation]

    raw_id = item.get("id")
    sample_id = f"tacred_{split_name}_{raw_id}" if raw_id else f"tacred_{split_name}_{sample_index}"

    metadata = {
        "source_split": split_name,
        "raw_id": raw_id,
        "head_type": item.get("subj_type"),
        "tail_type": item.get("obj_type"),
        "raw_subj_span_inclusive": [subj_start, subj_end_incl],
        "raw_obj_span_inclusive": [obj_start, obj_end_incl],
    }
    if "stanford_ner" in item:
        metadata["stanford_ner"] = item["stanford_ner"]
    if "stanford_pos" in item:
        metadata["stanford_pos"] = item["stanford_pos"]

    return RelationSample(
        sample_id=sample_id,
        tokens=tokens,
        head_text=head_text,
        head_start=head_start,
        head_end=head_end,
        tail_text=tail_text,
        tail_start=tail_start,
        tail_end=tail_end,
        relation=relation,
        relation_id=relation_id,
        metadata=metadata,
    )


class TACREDDataset(ContinualRelationDataset):
    """TACRED (Stanford / LDC) dataset loader and normalizer.

    TACRED contains 42 relations (41 positive relations + 'no_relation')
    across ~106,264 sentences.

    Because TACRED is licensed by LDC, raw JSON files (train.json, dev.json, test.json)
    must be acquired via LDC license and placed under `data/raw/tacred/`.
    """

    def __init__(
        self,
        data_dir: Path | str,
        include_no_relation: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.include_no_relation = include_no_relation

        self._train_samples: list[RelationSample] | None = None
        self._val_samples: list[RelationSample] | None = None
        self._test_samples: list[RelationSample] | None = None

        # Build relation mapping (sorted alphabetically, optionally excluding no_relation)
        if self.include_no_relation:
            self._relations = sorted(TACRED_STANDARD_RELATIONS)
        else:
            self._relations = sorted(r for r in TACRED_STANDARD_RELATIONS if r != "no_relation")
        self._relation_to_id = {r: i for i, r in enumerate(self._relations)}

    @property
    def is_available(self) -> bool:
        """Check if required raw TACRED files are locally available."""
        return all((self.data_dir / filename).exists() for filename in TACRED_REQUIRED_FILES.values())

    def get_relations(self) -> list[str]:
        return list(self._relations)

    def get_relation_to_id(self) -> dict[str, int]:
        return dict(self._relation_to_id)

    def _load_raw_split(self, split: str) -> list[RelationSample]:
        filename = TACRED_REQUIRED_FILES[split]
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing TACRED file: {file_path}. "
                "TACRED requires an LDC license (LDC2018T24). "
                "Place train.json, dev.json, and test.json in data/raw/tacred/"
            )

        raw_records: list[dict[str, Any]] = read_json(file_path)
        samples: list[RelationSample] = []

        for idx, item in enumerate(raw_records):
            rel = str(item["relation"])
            if rel == "no_relation" and not self.include_no_relation:
                continue
            sample = normalize_tacred_record(
                item=item,
                relation_to_id=self._relation_to_id,
                split_name=split,
                sample_index=idx,
            )
            samples.append(sample)
        return samples

    def load_train(self) -> list[RelationSample]:
        if self._train_samples is None:
            self._train_samples = self._load_raw_split("train")
        return list(self._train_samples)

    def load_validation(self) -> list[RelationSample]:
        if self._val_samples is None:
            self._val_samples = self._load_raw_split("dev")
        return list(self._val_samples)

    def load_test(self) -> list[RelationSample]:
        if self._test_samples is None:
            self._test_samples = self._load_raw_split("test")
        return list(self._test_samples)
