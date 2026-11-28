"""Configuration dataclasses and research-invariant validation for DynaTree-CRE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MethodConfig:
    name: str = "dynatree_cre"
    variant: str = "DT-C4"


@dataclass
class PrototypeConfig:
    encoder: str = "bert-base-uncased"
    freeze_encoder: bool = True
    representation_strategy: str = "entity_pair"  # entity_pair | cls | mean_pool
    normalization: str = "l2"
    aggregation: str = "mean"
    similarity_space: str = "t1_centered_cosine"  # t1_centered_cosine | incremental_centered_cosine | raw_cosine
    retrieval_similarity_space: str | None = None  # None -> defaults to similarity_space; or raw_cosine | t1_centered_cosine | incremental_centered_cosine
    use_relation_description: bool = False
    description_weight: float = 0.25
    use_analytic_retrieval: bool = False
    analytic_projection_mode: str = "random_relu"  # identity | random_relu
    analytic_projection_dim: int = 4096
    analytic_projection_seed: int = 2021
    analytic_ridge_lambda: float = 10.0
    use_query_adapter: bool = False
    query_adapter_mode: str = "t1_frozen"  # t1_frozen | continual_anchor_preserved
    query_adapter_hidden_dim: int = 256
    query_adapter_residual_scale: float = 0.5
    query_adapter_epochs: int = 20
    query_adapter_lr: float = 1.0e-3
    query_adapter_temperature: float = 0.1
    query_adapter_drift_weight: float = 5.0
    max_seq_length: int = 128


@dataclass
class TreeConfig:
    max_depth: int = 2
    initial_clustering: str = "agglomerative"
    linkage: str = "average"
    distance_metric: str = "cosine"
    threshold_mode: str = "t1_quantile"  # t1_quantile | fixed
    t1_quantile: float = 0.70
    insertion_threshold: float = 0.75
    update_policy: str = "insert_or_create"
    cluster_weighting: str = "equal_relation"  # equal_relation | sample_count
    allow_split: bool = False
    allow_merge: bool = False
    allow_reparenting: bool = False


@dataclass
class PromptConfig:
    length: int = 8  # Matches P0-P4 token budget (L=10 configurable)
    composition: str = "residual_addition"
    init_strategy: str = "normal"  # normal | embedding_stats | zero_leaf
    use_root_prompt: bool = True
    use_cluster_prompts: bool = True
    freeze_old_root: bool = True
    freeze_old_clusters: bool = True
    freeze_old_leaves: bool = True


@dataclass
class RoutingConfig:
    strategy: str = "hierarchical_soft"  # none | flat_hard | flat_soft_topk | dense_soft | hierarchical_soft | size_corrected_hierarchical | hybrid_tree_routing | oracle_route
    train_mode: str = "inference_consistent"  # inference_consistent | dense_soft | warmup_target_assist
    warmup_epochs: int = 0
    top_k: int = 3
    cluster_temperature: float = 0.2
    relation_temperature: float = 0.2
    cluster_size_eta: float = 0.0
    hybrid_cluster_gamma: float = 0.0
    inference_mode: str = "soft_mix"  # soft_mix | candidate_rerank
    rerank_fusion_mode: str = "fusion"  # retrieval_only | prompted_only | fusion
    rerank_fusion_lambda: float = 0.5
    rerank_score_norm: str = "zscore"  # zscore | minmax | none
    rerank_forward_chunk_size: int = 128
    rerank_train_cand_k: int = 4
    use_task_id: bool = False
    use_gold_relation: bool = False


@dataclass
class ClassifierConfig:
    type: str = "expandable_cosine"
    scale: float = 20.0
    init_from_prompted_prototype: bool = True
    freeze_old_weights: bool = True
    also_evaluate_ncm: bool = True


@dataclass
class MemoryConfig:
    exemplar_budget: int = 0
    feature_replay: bool = False
    generative_replay: bool = False
    retain_instance_embeddings: bool = False


@dataclass
class LossConfig:
    classification: bool = True
    contrastive: bool = False
    routing_distillation: bool = False
    structure_preservation: bool = False
    parameter_regularization: bool = False


@dataclass
class TrainingConfig:
    epochs_per_task: int = 15
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 5.0e-3
    classifier_lr: float = 5.0e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    device: str = "auto"


@dataclass
class ExperimentConfig:
    experiment_id: str = "DT-C4"
    dataset: str = "fewrel"
    track: str = "A"
    task_distribution: str = "standard_random"  # standard_random | semantic_homogeneous | cross_task_dispersed
    base_dir: str = "data/protocol_v1/fewrel"
    split_subdir: str = "CFRLdata_10_100_10_5"
    task_order_strategy: str = "conpl_seed_shuffle"
    train_shot: int = 5
    test_shot: int = 100
    seed: int = 2021
    output_dir: str = "runs/dynatree_cre/dt_c4/fewrel_5shot/seed_2021"


@dataclass
class DynaTreeConfig:
    method: MethodConfig = field(default_factory=MethodConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def validate(self) -> None:
        """Enforce strict DynaTree-CRE scientific and MVP invariants."""
        if not self.prototype.freeze_encoder:
            raise ValueError("DynaTree-CRE requires prototype.freeze_encoder=True.")
        if self.prototype.representation_strategy not in ("entity_pair", "cls", "mean_pool"):
            raise ValueError(f"Unsupported representation_strategy: {self.prototype.representation_strategy}")
        if self.prototype.similarity_space not in (
            "t1_centered_cosine",
            "incremental_centered_cosine",
            "raw_cosine",
        ):
            raise ValueError(f"Unsupported similarity_space: {self.prototype.similarity_space}")
        if self.prototype.retrieval_similarity_space is not None and self.prototype.retrieval_similarity_space not in (
            "t1_centered_cosine",
            "incremental_centered_cosine",
            "raw_cosine",
        ):
            raise ValueError(
                f"Unsupported retrieval_similarity_space: {self.prototype.retrieval_similarity_space}"
            )

        if self.prototype.analytic_projection_mode not in ("identity", "random_relu"):
            raise ValueError(
                f"Unsupported analytic_projection_mode: {self.prototype.analytic_projection_mode}"
            )
        if self.prototype.query_adapter_mode not in ("t1_frozen", "continual_anchor_preserved"):
            raise ValueError(
                f"Unsupported query_adapter_mode: {self.prototype.query_adapter_mode}"
            )
        if not (0.0 <= float(self.prototype.description_weight) < 1.0):
            raise ValueError("prototype.description_weight must be in [0.0, 1.0).")

        if self.tree.distance_metric == "cosine" and self.tree.linkage == "ward":
            raise ValueError("Ward linkage is mathematically invalid with cosine distance.")
        if self.tree.max_depth != 2:
            raise ValueError(f"MVP requires tree.max_depth == 2, got {self.tree.max_depth}.")
        if self.tree.allow_split or self.tree.allow_merge or self.tree.allow_reparenting:
            raise ValueError("Split/merge/reparenting are out of MVP scope (M7 only).")
        if self.tree.cluster_weighting not in ("equal_relation", "sample_count"):
            raise ValueError(f"Unsupported cluster_weighting: {self.tree.cluster_weighting}")

        if self.memory.exemplar_budget != 0:
            raise ValueError("Strict rehearsal-free MVP requires memory.exemplar_budget == 0.")
        if self.memory.feature_replay or self.memory.generative_replay or self.memory.retain_instance_embeddings:
            raise ValueError("Feature replay, generative replay, and instance caching are forbidden.")

        if self.routing.use_task_id and self.method.variant != "ORACLE_DIAGNOSTIC":
            raise ValueError("Task ID at inference is forbidden outside ORACLE_DIAGNOSTIC.")
        if self.routing.use_gold_relation and self.method.variant != "ORACLE_DIAGNOSTIC":
            raise ValueError("Gold relation routing at inference is forbidden outside ORACLE_DIAGNOSTIC.")
        if self.routing.strategy not in (
            "none",
            "flat_hard",
            "flat_soft_topk",
            "dense_soft",
            "hierarchical_soft",
            "size_corrected_hierarchical",
            "hybrid_tree_routing",
            "analytic_retrieval",
            "hybrid_analytic_tree",
            "oracle_route",
        ):
            raise ValueError(f"Unsupported routing.strategy: {self.routing.strategy}")
        if self.routing.train_mode not in ("inference_consistent", "dense_soft", "warmup_target_assist"):
            raise ValueError(f"Unsupported routing.train_mode: {self.routing.train_mode}")
        if self.routing.top_k < 1:
            raise ValueError("routing.top_k must be >= 1.")
        if self.routing.inference_mode not in ("soft_mix", "candidate_rerank"):
            raise ValueError(f"Unsupported routing.inference_mode: {self.routing.inference_mode}")
        if self.routing.rerank_fusion_mode not in ("retrieval_only", "prompted_only", "fusion"):
            raise ValueError(f"Unsupported routing.rerank_fusion_mode: {self.routing.rerank_fusion_mode}")
        if not (0.0 <= float(self.routing.rerank_fusion_lambda) <= 1.0):
            raise ValueError("routing.rerank_fusion_lambda must be in [0.0, 1.0].")
        if self.routing.rerank_score_norm not in ("zscore", "minmax", "none"):
            raise ValueError(f"Unsupported routing.rerank_score_norm: {self.routing.rerank_score_norm}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DynaTreeConfig:
        cfg = cls(
            method=MethodConfig(**raw.get("method", {})),
            prototype=PrototypeConfig(**raw.get("prototype", {})),
            tree=TreeConfig(**raw.get("tree", {})),
            prompt=PromptConfig(**raw.get("prompt", {})),
            routing=RoutingConfig(**raw.get("routing", {})),
            classifier=ClassifierConfig(**raw.get("classifier", {})),
            memory=MemoryConfig(**raw.get("memory", {})),
            loss=LossConfig(**raw.get("loss", {})),
            training=TrainingConfig(**raw.get("training", {})),
            experiment=ExperimentConfig(**raw.get("experiment", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> DynaTreeConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def save_yaml(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
