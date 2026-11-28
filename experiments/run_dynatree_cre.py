#!/usr/bin/env python3
"""Entrypoint runner for DynaTree-CRE experiments (DT-C0 to DT-C4)."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynatree_cre import (  # noqa: E402
    DynaTreeConfig,
    DynaTreeContinualTrainer,
    DynaTreeCREModel,
    load_protocol_v1_task_stream,
)


def get_git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        return {"commit": commit, "branch": branch}
    except Exception as exc:
        return {"error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DynaTree-CRE MVP experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, mps, cuda, auto)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (smoke runs)")
    parser.add_argument(
        "--max-test-samples-per-task",
        type=int,
        default=None,
        help="Cap test samples per task (smoke runs)",
    )
    parser.add_argument("--epochs-override", type=int, default=None, help="Override epochs_per_task")
    parser.add_argument("--similarity-space", type=str, default=None)
    parser.add_argument("--retrieval-similarity-space", type=str, default=None)
    parser.add_argument("--threshold-mode", type=str, default=None)
    parser.add_argument("--insertion-threshold", type=float, default=None)
    parser.add_argument("--prompt-length", type=int, default=None)
    parser.add_argument("--routing-strategy", type=str, default=None)
    parser.add_argument("--inference-mode", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank-fusion-mode", type=str, default=None)
    parser.add_argument("--rerank-fusion-lambda", type=float, default=None)
    parser.add_argument("--use-relation-description", action="store_true")
    parser.add_argument("--description-weight", type=float, default=None)
    parser.add_argument("--analytic-projection-mode", type=str, default=None)
    parser.add_argument("--analytic-projection-dim", type=int, default=None)
    parser.add_argument("--analytic-ridge-lambda", type=float, default=None)
    parser.add_argument("--rerank-sweep", action="store_true", help="Evaluate K in {1,3,5,10} x fusion modes at end")
    parser.add_argument("--cluster-size-eta", type=float, default=None)
    parser.add_argument("--hybrid-cluster-gamma", type=float, default=None)
    parser.add_argument("--train-mode", type=str, default=None)
    parser.add_argument("--task-distribution", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint state.pt to resume from")
    parser.add_argument("--allow-hub-download", action="store_true", help="Allow HF hub download if not cached")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    config = DynaTreeConfig.from_yaml(cfg_path)

    if args.output_dir is not None:
        config.experiment.output_dir = args.output_dir
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.experiment.seed = int(args.seed)
    if args.epochs_override is not None:
        config.training.epochs_per_task = int(args.epochs_override)
    if args.similarity_space is not None:
        config.prototype.similarity_space = args.similarity_space
    if args.retrieval_similarity_space is not None:
        config.prototype.retrieval_similarity_space = args.retrieval_similarity_space
    if args.threshold_mode is not None:
        config.tree.threshold_mode = args.threshold_mode
    if args.insertion_threshold is not None:
        config.tree.insertion_threshold = float(args.insertion_threshold)
    if args.prompt_length is not None:
        config.prompt.length = int(args.prompt_length)
    if args.routing_strategy is not None:
        config.routing.strategy = args.routing_strategy
    if args.inference_mode is not None:
        config.routing.inference_mode = args.inference_mode
    if args.top_k is not None:
        config.routing.top_k = int(args.top_k)
    if args.rerank_fusion_mode is not None:
        config.routing.rerank_fusion_mode = args.rerank_fusion_mode
    if args.rerank_fusion_lambda is not None:
        config.routing.rerank_fusion_lambda = float(args.rerank_fusion_lambda)
    if args.use_relation_description:
        config.prototype.use_relation_description = True
    if args.description_weight is not None:
        config.prototype.description_weight = float(args.description_weight)
    if args.analytic_projection_mode is not None:
        config.prototype.analytic_projection_mode = args.analytic_projection_mode
    if args.analytic_projection_dim is not None:
        config.prototype.analytic_projection_dim = int(args.analytic_projection_dim)
    if args.analytic_ridge_lambda is not None:
        config.prototype.analytic_ridge_lambda = float(args.analytic_ridge_lambda)
    if args.cluster_size_eta is not None:
        config.routing.cluster_size_eta = float(args.cluster_size_eta)
    if args.hybrid_cluster_gamma is not None:
        config.routing.hybrid_cluster_gamma = float(args.hybrid_cluster_gamma)
    if args.train_mode is not None:
        config.routing.train_mode = args.train_mode
    if args.task_distribution is not None:
        config.experiment.task_distribution = args.task_distribution
    config.validate()

    out_dir = Path(config.experiment.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    base_dir = Path(config.experiment.base_dir)
    if not base_dir.is_absolute():
        base_dir = REPO_ROOT / base_dir

    stream = load_protocol_v1_task_stream(
        base_dir=base_dir,
        split_subdir=config.experiment.split_subdir,
        task_order_strategy=config.experiment.task_order_strategy,
        task_distribution=config.experiment.task_distribution,
        seed=config.experiment.seed,
        train_shot=config.experiment.train_shot,
        test_shot=config.experiment.test_shot,
        max_seq_length=config.prototype.max_seq_length,
        max_tasks=args.max_tasks,
        max_test_samples_per_task=args.max_test_samples_per_task,
    )

    model = DynaTreeCREModel(config, local_files_only=not args.allow_hub_download)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trainer = DynaTreeContinualTrainer(
        model=model,
        config=config,
        stream=stream,
        output_dir=out_dir,
        run_id=run_id,
        git_info=get_git_info(),
    )

    resume_path = None
    if args.resume_from is not None:
        rp = Path(args.resume_from)
        resume_path = rp if rp.is_absolute() else REPO_ROOT / rp

    metrics = trainer.run(resume_from=resume_path)
    if args.rerank_sweep:
        trainer.evaluate_rerank_sweep()
    print(json.dumps({
        "variant": metrics["variant"],
        "dataset": metrics["dataset"],
        "num_tasks_executed": metrics["num_tasks_executed"],
        "final_cosine_whole_accuracy": round(metrics["final_cosine_whole_accuracy"], 6),
        "final_ncm_whole_accuracy": round(metrics["final_ncm_whole_accuracy"], 6),
        "final_oracle_cosine_whole_accuracy": (
            round(metrics["final_oracle_cosine_whole_accuracy"], 6)
            if metrics["final_oracle_cosine_whole_accuracy"] is not None
            else None
        ),
        "cosine_summary": metrics["cosine_summary"],
        "ncm_summary": metrics["ncm_summary"],
        "final_tree_metrics": metrics["final_tree_metrics"],
        "output_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
