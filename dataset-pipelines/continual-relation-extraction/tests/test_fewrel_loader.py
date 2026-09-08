from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.fewrel import FewRelDataset
from utils.io import read_json, write_json


class TestFewRelLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        # Create mock raw FewRel files
        self.mock_train = {
            "P17": [
                {
                    "tokens": ["London", "is", "in", "the", "UK", "."],
                    "h": ["London", "Q84", [[0]]],
                    "t": ["UK", "Q145", [[4]]],
                },
                {
                    "tokens": ["Paris", "is", "the", "capital", "of", "France", "."],
                    "h": ["Paris", "Q90", [[0]]],
                    "t": ["France", "Q142", [[5]]],
                },
            ],
            "P19": [
                {
                    "tokens": ["John", "was", "born", "in", "Berlin", "."],
                    "h": ["John", "Q1", [[0]]],
                    "t": ["Berlin", "Q64", [[4]]],
                },
            ],
        }
        self.mock_val = {
            "P20": [
                {
                    "tokens": ["Mary", "died", "in", "Rome", "."],
                    "h": ["Mary", "Q2", [[0]]],
                    "t": ["Rome", "Q220", [[3]]],
                },
            ]
        }
        self.mock_pid = {
            "P17": ["country", "sovereign state of this item"],
            "P19": ["place of birth", "most specific known place of birth"],
            "P20": ["place of death", "most specific known place of death"],
        }

        write_json(self.data_dir / "train_wiki.json", self.mock_train)
        write_json(self.data_dir / "val_wiki.json", self.mock_val)
        write_json(self.data_dir / "pid2name.json", self.mock_pid)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fewrel_mock_loading_and_spans(self) -> None:
        # Split counts: 1 train, 0 val, 1 test (for relations with 2 items) or 1, 0, 0
        ds = FewRelDataset(self.data_dir, train_val_test_counts=(1, 0, 1))
        relations = ds.get_relations()
        self.assertEqual(relations, ["P17", "P19", "P20"])

        rel_to_id = ds.get_relation_to_id()
        self.assertEqual(rel_to_id, {"P17": 0, "P19": 1, "P20": 2})

        train_samples = ds.load_train()
        self.assertEqual(len(train_samples), 3)

        # Check sample 0 (P17)
        s0 = train_samples[0]
        self.assertEqual(s0.sample_id, "fewrel_train_P17_0")
        self.assertEqual(s0.head_tokens, ["London"])
        self.assertEqual(s0.head_start, 0)
        self.assertEqual(s0.head_end, 1)
        self.assertEqual(s0.tail_tokens, ["UK"])
        self.assertEqual(s0.tail_start, 4)
        self.assertEqual(s0.tail_end, 5)
        self.assertEqual(s0.relation, "P17")
        self.assertEqual(s0.relation_id, 0)
        self.assertEqual(s0.metadata["relation_name"], "country")

        test_samples = ds.load_test()
        # Only P17 had 2 examples, so index 1 went to test
        self.assertEqual(len(test_samples), 1)
        self.assertEqual(test_samples[0].sample_id, "fewrel_test_P17_0")
        self.assertEqual(test_samples[0].head_tokens, ["Paris"])
        self.assertEqual(test_samples[0].tail_tokens, ["France"])

    def test_save_relation_mapping(self) -> None:
        ds = FewRelDataset(self.data_dir)
        out_path = self.data_dir / "mapping.json"
        ds.save_relation_mapping(out_path)
        self.assertTrue(out_path.exists())
        loaded = read_json(out_path)
        self.assertEqual(loaded, {"P17": 0, "P19": 1, "P20": 2})

    def test_real_fewrel_data_if_available(self) -> None:
        real_data_dir = Path("data/raw/fewrel")
        if not (real_data_dir / "train_wiki.json").exists():
            self.skipTest("Real FewRel data not available")

        # Test with a small slice count to ensure fast testing
        ds = FewRelDataset(real_data_dir, train_val_test_counts=(10, 2, 2))
        relations = ds.get_relations()
        self.assertEqual(len(relations), 80)

        # Verify deterministic alphabetical sorting
        self.assertEqual(relations, sorted(relations))

        rel_to_id = ds.get_relation_to_id()
        self.assertEqual(len(rel_to_id), 80)
        self.assertEqual(list(rel_to_id.values()), list(range(80)))

        train_samples = ds.load_train()
        self.assertEqual(len(train_samples), 80 * 10)
        self.assertTrue(all(s.sample_id.startswith("fewrel_train_") for s in train_samples))


if __name__ == "__main__":
    unittest.main()
