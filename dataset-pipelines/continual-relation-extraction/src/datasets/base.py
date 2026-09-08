from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from schemas.relation_sample import RelationSample


class ContinualRelationDataset(ABC):
    """Abstract base class for relation extraction datasets in continual learning.

    Provides a uniform, model-agnostic interface to access training, validation,
    and test splits as canonical `RelationSample` instances, along with
    stable relation mappings.
    """

    @abstractmethod
    def load_train(self) -> list[RelationSample]:
        """Load and return all training samples."""
        raise NotImplementedError

    @abstractmethod
    def load_validation(self) -> list[RelationSample]:
        """Load and return all validation/development samples."""
        raise NotImplementedError

    @abstractmethod
    def load_test(self) -> list[RelationSample]:
        """Load and return all test samples."""
        raise NotImplementedError

    @abstractmethod
    def get_relations(self) -> list[str]:
        """Return the complete list of unique relation names in deterministic order."""
        raise NotImplementedError

    @abstractmethod
    def get_relation_to_id(self) -> dict[str, int]:
        """Return a mapping from relation name to stable numeric ID."""
        raise NotImplementedError

    def get_id_to_relation(self) -> dict[int, str]:
        """Return a reverse mapping from numeric ID to relation name."""
        return {idx: rel for rel, idx in self.get_relation_to_id().items()}

    def load_split(self, split: str) -> list[RelationSample]:
        """Load samples for a given split name.

        Args:
            split: One of 'train'/'training', 'val'/'dev'/'validation', or 'test'.

        Returns:
            List of canonical RelationSample objects.
        """
        normalized = split.strip().lower()
        if normalized in {"train", "training"}:
            return self.load_train()
        if normalized in {"val", "valid", "validation", "dev"}:
            return self.load_validation()
        if normalized in {"test", "testing"}:
            return self.load_test()
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of 'train', 'validation' ('val'/'dev'), or 'test'."
        )

    @property
    def num_relations(self) -> int:
        """Total number of unique relations in the dataset."""
        return len(self.get_relations())
