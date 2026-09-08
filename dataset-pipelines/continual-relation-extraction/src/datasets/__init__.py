"""Dataset abstractions and loaders for Continual Relation Extraction."""

from .base import ContinualRelationDataset
from .fewrel import FewRelDataset

__all__ = ["ContinualRelationDataset", "FewRelDataset"]
