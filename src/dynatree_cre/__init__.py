"""DynaTree-CRE: Dynamic Semantic Prompt Trees for Rehearsal-Free Continual Relation Extraction."""

from src.dynatree_cre.analytic import IncrementalAnalyticHead
from src.dynatree_cre.benchmark import (
    TaskStreamBundle,
    build_synthetic_task_stream,
    load_protocol_v1_task_stream,
)
from src.dynatree_cre.classifier import ExpandableCosineClassifier
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.prototype_bank import RelationPrototype, SemanticPrototypeBank
from src.dynatree_cre.query_adapter import ResidualQueryAdapter
from src.dynatree_cre.query_encoder import StableQueryEncoder
from src.dynatree_cre.router import RoutingResult, TaskAgnosticRouter
from src.dynatree_cre.trainer import DynaTreeContinualTrainer, set_deterministic_seed
from src.dynatree_cre.tree import DynamicSemanticTree, TreeNode

__all__ = [
    "DynaTreeConfig",
    "StableQueryEncoder",
    "RelationPrototype",
    "SemanticPrototypeBank",
    "TreeNode",
    "DynamicSemanticTree",
    "HierarchicalPromptBank",
    "RoutingResult",
    "TaskAgnosticRouter",
    "ExpandableCosineClassifier",
    "IncrementalAnalyticHead",
    "ResidualQueryAdapter",
    "DynaTreeCREModel",
    "DynaTreeContinualTrainer",
    "TaskStreamBundle",
    "load_protocol_v1_task_stream",
    "build_synthetic_task_stream",
    "set_deterministic_seed",
]
