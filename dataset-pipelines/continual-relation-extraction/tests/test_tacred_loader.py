from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.tacred import (
    TACRED_STANDARD_RELATIONS,
    TACREDDataset,
    normalize_tacred_record,
)
from utils.io import write_json


class TestTACREDLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_raw = {
            "id": "e77ee47f482d881e191b",
            "relation": "org:founded_by",
            "token": [
                "Chief", "Executive", "Steve", "Jobs", "co-founded", "Apple", "Computer", "in", "1976", "."
            ],
            "subj_start": 5,
            "subj_end": 6,
            "subj_type": "ORGANIZATION",
            "obj_start": 2,
            "obj_end": 3,
            "obj_type": "PERSON",
        }
        self.rel_to_id = {r: i for i, r in enumerate(sorted(TACRED_STANDARD_RELATIONS))}

    def test_normalize_tacred_record_span_semantics(self) -> None:
        sample = normalize_tacred_record(
            self.sample_raw,
            relation_to_id=self.rel_to_id,
            split_name="train",
            sample_index=0,
        )

        self.assertEqual(sample.sample_id, "tacred_train_e77ee47f482d881e191b")
        self.assertEqual(sample.relation, "org:founded_by")

        # In raw: subj_start=5, subj_end=6 -> tokens[5] = "Apple", tokens[6] = "Computer"
        # In normalized: half-open [5, 7)
        self.assertEqual(sample.head_start, 5)
        self.assertEqual(sample.head_end, 7)
        self.assertEqual(sample.head_text, "Apple Computer")
        self.assertEqual(sample.head_tokens, ["Apple", "Computer"])

        # In raw: obj_start=2, obj_end=3 -> tokens[2] = "Steve", tokens[3] = "Jobs"
        # In normalized: half-open [2, 4)
        self.assertEqual(sample.tail_start, 2)
        self.assertEqual(sample.tail_end, 4)
        self.assertEqual(sample.tail_text, "Steve Jobs")
        self.assertEqual(sample.tail_tokens, ["Steve", "Jobs"])

        # Check metadata
        self.assertEqual(sample.metadata["head_type"], "ORGANIZATION")
        self.assertEqual(sample.metadata["tail_type"], "PERSON")

    def test_tacred_dataset_relation_configurations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without no_relation
            ds_no_neg = TACREDDataset(tmpdir, include_no_relation=False)
            rels_no_neg = ds_no_neg.get_relations()
            self.assertEqual(len(rels_no_neg), 41)
            self.assertNotIn("no_relation", rels_no_neg)

            # With no_relation
            ds_with_neg = TACREDDataset(tmpdir, include_no_relation=True)
            rels_with_neg = ds_with_neg.get_relations()
            self.assertEqual(len(rels_with_neg), 42)
            self.assertIn("no_relation", rels_with_neg)
            self.assertEqual(ds_with_neg.is_available, False)

    def test_unknown_relation_raises(self) -> None:
        bad_item = dict(self.sample_raw)
        bad_item["relation"] = "invalid_unknown_relation"
        with self.assertRaises(ValueError) as ctx:
            normalize_tacred_record(bad_item, relation_to_id=self.rel_to_id)
        self.assertIn("Unknown TACRED relation", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
