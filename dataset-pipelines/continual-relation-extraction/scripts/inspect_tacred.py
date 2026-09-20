#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.tacred import (
    TACRED_REQUIRED_FILES,
    TACRED_STANDARD_RELATIONS,
    TACREDDataset,
    normalize_tacred_record,
)
from utils.io import read_json


def resolve_path(candidate_path: str | Path) -> Path:
    p = Path(candidate_path)
    if p.is_absolute() and p.exists():
        return p
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    if (PROJECT_DIR / p).exists():
        return (PROJECT_DIR / p).resolve()
    return p.resolve()


def get_representative_tacred_example() -> dict:
    """Representative raw TACRED sample matching Stanford/LDC schema."""
    return {
        "id": "e77ee47f482d881e191b",
        "relation": "org:founded_by",
        "token": [
            "At", "the", "same", "time", ",", "Chief", "Executive", "Steve", "Jobs",
            "co-founded", "Apple", "Computer", "in", "1976", "."
        ],
        "subj_start": 10,
        "subj_end": 11,
        "subj_type": "ORGANIZATION",
        "obj_start": 7,
        "obj_end": 8,
        "obj_type": "PERSON",
        "stanford_pos": [
            "IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP",
            "VBD", "NNP", "NNP", "IN", "CD", "."
        ],
        "stanford_ner": [
            "O", "O", "O", "O", "O", "TITLE", "TITLE", "PERSON", "PERSON",
            "O", "ORGANIZATION", "ORGANIZATION", "O", "DATE", "O"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TACRED dataset access status and canonical mapping feasibility.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/tacred",
        help="Path to directory containing raw TACRED files",
    )
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)

    print("========================================")
    print("TACRED Feasibility & Preprocessing Audit")
    print("========================================")

    missing_files = [
        fname for fname in TACRED_REQUIRED_FILES.values()
        if not (data_dir / fname).exists()
    ]

    dataset = TACREDDataset(data_dir=data_dir, include_no_relation=True)
    rel_to_id = dataset.get_relation_to_id()

    if not missing_files:
        print("\nTACRED access status: AVAILABLE (Local raw files found)")
        print(f"Data directory:       {data_dir}")

        train_data = read_json(data_dir / "train.json")
        dev_data = read_json(data_dir / "dev.json")
        test_data = read_json(data_dir / "test.json")

        all_records = train_data + dev_data + test_data
        no_rel_count = sum(1 for r in all_records if r.get("relation") == "no_relation")

        print(f"Train:                {len(train_data)} samples")
        print(f"Dev:                  {len(dev_data)} samples")
        print(f"Test:                 {len(test_data)} samples")
        print(f"Total samples:        {len(all_records)}")
        print(f"Relations:            {len(TACRED_STANDARD_RELATIONS)} (41 positive + no_relation)")
        print(f"no_relation count:    {no_rel_count} ({no_rel_count / len(all_records) * 100:.1f}%)")

        sample_raw = train_data[0]
        test_sample = normalize_tacred_record(sample_raw, relation_to_id=rel_to_id, split_name="train", sample_index=0)
    else:
        print("\nTACRED access status: MANUAL ACCESS REQUIRED")
        print("Reason:               TACRED is distributed under LDC license (LDC2018T24).")
        print("Source:               Linguistic Data Consortium (LDC) / The Stanford NLP Group")
        print(f"Missing local files:  {', '.join(missing_files)} in {data_dir}")
        print("\nExpected split structure when acquired:")
        print("  Train:              68,124 samples (train.json)")
        print("  Dev:                22,631 samples (dev.json)")
        print("  Test:               15,509 samples (test.json)")
        print("  Total samples:      106,264")
        print(f"  Total relations:    42 (41 positive + no_relation)")
        print("  no_relation share:  ~79.5% of total dataset")

        sample_raw = get_representative_tacred_example()
        test_sample = normalize_tacred_record(sample_raw, relation_to_id=rel_to_id, split_name="audit", sample_index=0)

    # Test Canonical Conversion
    conversion_status = "BLOCKED"
    try:
        # Check invariants
        assert test_sample.sample_id.startswith("tacred_")
        assert 0 <= test_sample.head_start < test_sample.head_end <= len(test_sample.tokens)
        assert 0 <= test_sample.tail_start < test_sample.tail_end <= len(test_sample.tokens)
        assert test_sample.relation in rel_to_id
        conversion_status = "SUPPORTED"
    except Exception as exc:
        conversion_status = f"BLOCKED ({exc})"

    print(f"\nCanonical conversion: {conversion_status}")
    print("\nExample normalized sample:")
    print(json.dumps(test_sample.to_dict(), indent=2))

    print("\nSpan Mapping Check:")
    print(f"  Raw subject span (inclusive): {sample_raw.get('subj_start')}..{sample_raw.get('subj_end')}")
    print(f"  Normalized head span:         [{test_sample.head_start}, {test_sample.head_end}) -> {test_sample.head_tokens}")
    print(f"  Raw object span (inclusive):  {sample_raw.get('obj_start')}..{sample_raw.get('obj_end')}")
    print(f"  Normalized tail span:         [{test_sample.tail_start}, {test_sample.tail_end}) -> {test_sample.tail_tokens}")

    print("\n========================================")
    print("TACRED AUDIT COMPLETE")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
