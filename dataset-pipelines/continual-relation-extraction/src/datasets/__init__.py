"""Dataset abstractions and loaders for Continual Relation Extraction."""

from .base import ContinualRelationDataset
from .fewrel import FewRelDataset
from .tacred import TACRED_STANDARD_RELATIONS, TACREDDataset, normalize_tacred_record

__all__ = [
    "ContinualRelationDataset",
    "FewRelDataset",
    "TACREDDataset",
    "normalize_tacred_record",
    "TACRED_STANDARD_RELATIONS",
]
