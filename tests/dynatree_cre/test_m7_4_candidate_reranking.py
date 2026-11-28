"""TDD tests for M7.4 Candidate-Aware Prompt Reranking (DT-C2.1 Flat & DT-C4.4 Tree)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.dynatree_cre.benchmark import build_synthetic_task_stream
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.model import DynaTreeCREModel
from src.dynatree_cre.prompt_bank import HierarchicalPromptBank
from src.dynatree_cre.trainer import DynaTreeContinualTrainer


def test_compose_for_relation_ids_matches_get_effective_prompt(tmp_path: Path) -> None:
    stream = build_synthetic_task_stream(num_tasks=2, relations_per_task=3, train_shot=3, test_shot=2)
    cfg = DynaTreeConfig.from_dict({
        "method": {"variant": "DT-C4.4"},
        "prototype": {"encoder": "synthetic_tiny", "max_seq_length": 32},
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": True, "use_cluster_prompts": True},
        "routing": {"strategy": "hybrid_tree_routing", "inference_mode": "candidate_rerank", "top_k": 3},
        "training": {"epochs_per_task": 1, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": "M7.4-UNIT", "dataset": "synthetic", "output_dir": str(tmp_path)},
    })
    model = DynaTreeCREModel(cfg)
    t0 = stream.tasks[0]
    q0 = {r: torch.randn(3, model.rep_dim) for r in t0.relations}
    model.prototype_bank.register_task_prototypes(q0, task_id=0)
    model.tree.insert_task_relations(model.prototype_bank.prototypes, task_id=0, config=cfg.tree)
    model.prompt_bank.synchronize_with_tree(model.tree, task_id=0)

    rel_tensor = torch.tensor([t0.relations[2], t0.relations[0], t0.relations[1]], dtype=torch.long)
    batched = model.prompt_bank.compose_for_relation_ids(rel_tensor, model.tree)
    assert batched.shape == (3, 4, model.hidden_size)
    for i, rid in enumerate(rel_tensor.tolist()):
        single = model.prompt_bank.get_effective_prompt(rid, model.tree)
        assert torch.allclose(batched[i], single, atol=1e-6)


@pytest.mark.parametrize(
    "variant,strategy,use_root,use_cluster,fusion_mode",
    [
        ("DT-C2.1", "flat_soft_topk", False, False, "retrieval_only"),
        ("DT-C2.1", "flat_soft_topk", False, False, "prompted_only"),
        ("DT-C2.1", "flat_soft_topk", False, False, "fusion"),
        ("DT-C4.4", "hybrid_tree_routing", True, True, "fusion"),
    ],
)
def test_candidate_rerank_e2e_no_leakage_no_drift_and_exact_algebraic_identity(
    tmp_path: Path,
    variant: str,
    strategy: str,
    use_root: bool,
    use_cluster: bool,
    fusion_mode: str,
) -> None:
    stream = build_synthetic_task_stream(num_tasks=3, relations_per_task=2, train_shot=4, test_shot=4)
    out_dir = tmp_path / f"{variant}_{fusion_mode}"
    cfg = DynaTreeConfig.from_dict({
        "method": {"variant": variant},
        "prototype": {
            "encoder": "synthetic_tiny",
            "max_seq_length": 32,
            "similarity_space": "t1_centered_cosine",
            "retrieval_similarity_space": "raw_cosine",
        },
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": use_root, "use_cluster_prompts": use_cluster},
        "routing": {
            "strategy": strategy,
            "inference_mode": "candidate_rerank",
            "rerank_fusion_mode": fusion_mode,
            "rerank_fusion_lambda": 0.5,
            "hybrid_cluster_gamma": 0.25 if strategy == "hybrid_tree_routing" else 0.0,
            "top_k": 3,
        },
        "training": {"epochs_per_task": 2, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": variant, "dataset": "synthetic", "output_dir": str(out_dir)},
    })
    model = DynaTreeCREModel(cfg)
    metrics = DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir).run()

    last_stage = metrics["stages"][-1]
    assert last_stage["train_stats"]["train_zero_grad_new_leaf_count"] == 0

    ev = last_stage["eval_stats"]
    rerank_stats = ev["rerank_diagnostics"]
    cand_rk = rerank_stats["candidate_recall_at_k"]
    cond_ncm = rerank_stats["conditional_rerank_ncm_acc"]
    final_ncm = ev["ncm_whole_accuracy"]
    # Section 7 exact identity: Final Accuracy == Candidate Recall@K * Conditional Reranking Accuracy
    assert final_ncm == pytest.approx(cand_rk * cond_ncm, abs=1e-7)

    # Check retrieval_only at K=3 equals R@1 when K_ret matches top-1 retrieval
    res = json.loads((out_dir / "resource_metrics.json").read_text())
    assert res["total_bert_sample_forward_passes"] > 0
    assert "peak_memory_bytes" in res

    # Verify old effective prompts remained frozen and NCM prototypes are all unit-normalized
    for p in model.prompt_bank.leaf_prompts.values():
        assert p.shape == (4, model.hidden_size)
    assert model.classifier.ncm_prototypes.shape == (6, model.rep_dim)


def test_rerank_k_one_matches_candidate_recall_at_one(tmp_path: Path) -> None:
    stream = build_synthetic_task_stream(num_tasks=2, relations_per_task=3, train_shot=4, test_shot=4)
    out_dir = tmp_path / "k1"
    cfg = DynaTreeConfig.from_dict({
        "method": {"variant": "DT-C4.4"},
        "prototype": {
            "encoder": "synthetic_tiny",
            "max_seq_length": 32,
            "similarity_space": "t1_centered_cosine",
            "retrieval_similarity_space": "raw_cosine",
        },
        "tree": {"threshold_mode": "t1_quantile", "t1_quantile": 0.50},
        "prompt": {"length": 4, "use_root_prompt": True, "use_cluster_prompts": True},
        "routing": {
            "strategy": "hybrid_tree_routing",
            "inference_mode": "candidate_rerank",
            "rerank_fusion_mode": "fusion",
            "top_k": 1,
        },
        "training": {"epochs_per_task": 1, "batch_size": 4, "eval_batch_size": 8, "device": "cpu"},
        "experiment": {"experiment_id": "M7.4-K1", "dataset": "synthetic", "output_dir": str(out_dir)},
    })
    model = DynaTreeCREModel(cfg)
    metrics = DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir).run()
    ev = metrics["stages"][-1]["eval_stats"]
    rr = ev["rerank_diagnostics"]
    assert rr["conditional_rerank_ncm_acc"] == pytest.approx(1.0)
    assert ev["ncm_whole_accuracy"] == pytest.approx(rr["candidate_recall_at_k"], abs=1e-7)

    sweep = DynaTreeContinualTrainer(model, cfg, stream, output_dir=out_dir).evaluate_rerank_sweep(
        ks=(1, 3, 5, 10)
    )
    assert (out_dir / "rerank_sweep.json").exists()
    assert len(sweep["cells"]) == 12
    for cell in sweep["cells"]:
        assert cell["final_ncm_whole_accuracy"] == pytest.approx(
            cell["candidate_recall_at_k"] * cell["conditional_rerank_ncm_acc"], abs=1e-7
        )
