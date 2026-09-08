from __future__ import annotations

import unittest
from datasets.base import ContinualRelationDataset
from schemas.relation_sample import RelationSample


class ConcreteTestDataset(ContinualRelationDataset):
    def __init__(self) -> None:
        self.sample = RelationSample(
            sample_id="dummy_0",
            tokens=["Alice", "knows", "Bob"],
            head_text="Alice",
            head_start=0,
            head_end=1,
            tail_text="Bob",
            tail_start=2,
            tail_end=3,
            relation="knows",
            relation_id=0,
        )

    def load_train(self) -> list[RelationSample]:
        return [self.sample]

    def load_validation(self) -> list[RelationSample]:
        return [self.sample]

    def load_test(self) -> list[RelationSample]:
        return [self.sample]

    def get_relations(self) -> list[str]:
        return ["knows"]

    def get_relation_to_id(self) -> dict[str, int]:
        return {"knows": 0}


class TestBaseDataset(unittest.TestCase):
    def test_cannot_instantiate_abc(self) -> None:
        with self.assertRaises(TypeError):
            ContinualRelationDataset()  # type: ignore

    def test_concrete_implementation(self) -> None:
        ds = ConcreteTestDataset()
        self.assertEqual(ds.num_relations, 1)
        self.assertEqual(ds.get_relations(), ["knows"])
        self.assertEqual(ds.get_relation_to_id(), {"knows": 0})
        self.assertEqual(ds.get_id_to_relation(), {0: "knows"})

        train = ds.load_train()
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0].relation, "knows")

    def test_load_split_aliases(self) -> None:
        ds = ConcreteTestDataset()
        self.assertEqual(len(ds.load_split("train")), 1)
        self.assertEqual(len(ds.load_split("training")), 1)
        self.assertEqual(len(ds.load_split("val")), 1)
        self.assertEqual(len(ds.load_split("validation")), 1)
        self.assertEqual(len(ds.load_split("dev")), 1)
        self.assertEqual(len(ds.load_split("test")), 1)
        self.assertEqual(len(ds.load_split("testing")), 1)

    def test_load_split_invalid(self) -> None:
        ds = ConcreteTestDataset()
        with self.assertRaises(ValueError):
            ds.load_split("unknown_split")


if __name__ == "__main__":
    unittest.main()
