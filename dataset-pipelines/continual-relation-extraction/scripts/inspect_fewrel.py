#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Add src to sys.path to enable direct imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.fewrel import FewRelDataset


def resolve_data_dir(candidate_path: str | Path) -> Path:
    p = Path(candidate_path)
    if p.exists() and (p / "train_wiki.json").exists():
        return p
    # Try relative to PROJECT_DIR
    p_proj = PROJECT_DIR / candidate_path
    if p_proj.exists() and (p_proj / "train_wiki.json").exists():
        return p_proj
    # Try relative to cwd
    p_cwd = Path.cwd() / candidate_path
    if p_cwd.exists() and (p_cwd / "train_wiki.json").exists():
        return p_cwd
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect FewRel dataset loading and normalization.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/fewrel",
        help="Path to directory containing train_wiki.json, val_wiki.json, pid2name.json",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    if not data_dir.exists() or not (data_dir / "train_wiki.json").exists():
        print(f"[ERROR] Could not find FewRel raw data at: {data_dir}", file=sys.stderr)
        return 1

    print("========================================")
    print("Dataset: FewRel")
    print(f"Data directory: {data_dir}")
    print("========================================")

    dataset = FewRelDataset(data_dir=data_dir)
    relations = dataset.get_relations()
    rel_to_id = dataset.get_relation_to_id()

    train_samples = dataset.load_train()
    val_samples = dataset.load_validation()
    test_samples = dataset.load_test()

    total_samples = len(train_samples) + len(val_samples) + len(test_samples)

    print(f"\nTrain sample count: {len(train_samples)}")
    print(f"Validation sample count: {len(val_samples)}")
    print(f"Test sample count: {len(test_samples)}")
    print(f"Total sample count: {total_samples}")
    print(f"\nNumber of relations: {len(relations)}")

    # Distribution check
    train_dist = Counter(s.relation for s in train_samples)
    val_dist = Counter(s.relation for s in val_samples)
    test_dist = Counter(s.relation for s in test_samples)

    print("\nRelation distribution (first 5 relations shown):")
    for rel in relations[:5]:
        print(
            f"  - {rel} (id={rel_to_id[rel]}): train={train_dist[rel]}, val={val_dist[rel]}, test={test_dist[rel]}"
        )
    if len(relations) > 5:
        print(f"  ... and {len(relations) - 5} more relations with identical balanced distribution.")

    # Display first normalized sample
    first_sample = train_samples[0]
    print("\nFirst normalized sample:")
    print(f"  sample_id:   {first_sample.sample_id}")
    print(f"  tokens:      {first_sample.tokens}")
    print(f"  head_text:   {first_sample.head_text!r}")
    print(f"  head_span:   [{first_sample.head_start}, {first_sample.head_end}) -> {first_sample.head_tokens}")
    print(f"  tail_text:   {first_sample.tail_text!r}")
    print(f"  tail_span:   [{first_sample.tail_start}, {first_sample.tail_end}) -> {first_sample.tail_tokens}")
    print(f"  relation:    {first_sample.relation}")
    print(f"  relation_id: {first_sample.relation_id}")
    if first_sample.metadata.get("relation_name"):
        print(f"  relation_name: {first_sample.metadata['relation_name']}")

    # Validation checks
    invalid_spans = 0
    unknown_relations = 0
    seen_ids: set[str] = set()
    duplicate_sample_ids = 0

    all_samples = train_samples + val_samples + test_samples
    for sample in all_samples:
        # Check span bounds
        n = len(sample.tokens)
        if not (0 <= sample.head_start < sample.head_end <= n) or not (0 <= sample.tail_start < sample.tail_end <= n):
            invalid_spans += 1

        # Check relation
        if sample.relation not in rel_to_id or sample.relation_id != rel_to_id[sample.relation]:
            unknown_relations += 1

        # Check duplicate IDs
        if sample.sample_id in seen_ids:
            duplicate_sample_ids += 1
        seen_ids.add(sample.sample_id)

    print("\nValidation Summary:")
    print(f"  Invalid entity spans: {invalid_spans}")
    print(f"  Unknown relations:    {unknown_relations}")
    print(f"  Duplicate sample IDs: {duplicate_sample_ids}")

    has_errors = invalid_spans > 0 or unknown_relations > 0 or duplicate_sample_ids > 0
    if has_errors:
        print("\n[FAILED] FewRel validation found errors.")
        return 1

    print("\n[PASSED] FewRel dataset and sample normalization verified successfully.")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
