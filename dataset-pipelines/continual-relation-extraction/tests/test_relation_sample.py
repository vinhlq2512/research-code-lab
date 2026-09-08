from __future__ import annotations

import unittest
from schemas.relation_sample import RelationSample


class TestRelationSample(unittest.TestCase):
    def test_valid_relation_sample(self) -> None:
        sample = RelationSample(
            sample_id="fewrel_train_P17_0",
            tokens=["Steve", "Jobs", "founded", "Apple", "."],
            head_text="Steve Jobs",
            head_start=0,
            head_end=2,
            tail_text="Apple",
            tail_start=3,
            tail_end=4,
            relation="founder_of",
            relation_id=0,
            metadata={"source": "test"},
        )
        self.assertEqual(sample.head_tokens, ["Steve", "Jobs"])
        self.assertEqual(sample.tail_tokens, ["Apple"])
        self.assertEqual(sample.sample_id, "fewrel_train_P17_0")
        self.assertEqual(sample.relation_id, 0)

        data = sample.to_dict()
        reconstructed = RelationSample.from_dict(data)
        self.assertEqual(sample, reconstructed)

    def test_invalid_head_span_start_negative(self) -> None:
        with self.assertRaises(ValueError):
            RelationSample(
                sample_id="s1",
                tokens=["A", "B"],
                head_text="A",
                head_start=-1,
                head_end=1,
                tail_text="B",
                tail_start=1,
                tail_end=2,
                relation="r",
                relation_id=0,
            )

    def test_invalid_head_span_start_equal_end(self) -> None:
        with self.assertRaises(ValueError):
            RelationSample(
                sample_id="s2",
                tokens=["A", "B"],
                head_text="A",
                head_start=1,
                head_end=1,
                tail_text="B",
                tail_start=1,
                tail_end=2,
                relation="r",
                relation_id=0,
            )

    def test_invalid_head_span_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RelationSample(
                sample_id="s3",
                tokens=["A", "B"],
                head_text="A",
                head_start=0,
                head_end=3,
                tail_text="B",
                tail_start=1,
                tail_end=2,
                relation="r",
                relation_id=0,
            )

    def test_invalid_tail_span(self) -> None:
        with self.assertRaises(ValueError):
            RelationSample(
                sample_id="s4",
                tokens=["A", "B"],
                head_text="A",
                head_start=0,
                head_end=1,
                tail_text="B",
                tail_start=2,
                tail_end=1,
                relation="r",
                relation_id=0,
            )

    def test_empty_relation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RelationSample(
                sample_id="s5",
                tokens=["A", "B"],
                head_text="A",
                head_start=0,
                head_end=1,
                tail_text="B",
                tail_start=1,
                tail_end=2,
                relation="",
                relation_id=0,
            )

    def test_negative_relation_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RelationSample(
                sample_id="s6",
                tokens=["A", "B"],
                head_text="A",
                head_start=0,
                head_end=1,
                tail_text="B",
                tail_start=1,
                tail_end=2,
                relation="r",
                relation_id=-1,
            )


if __name__ == "__main__":
    unittest.main()
