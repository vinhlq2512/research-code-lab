#!/usr/bin/env python3
"""M7.0 & M7.1 Checkpoint Diagnostic Runner (Equality Check + Routing Decomposition)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynatree_cre.benchmark import load_protocol_v1_task_stream  # noqa: E402
from src.dynatree_cre.checkpoint import load_dynatree_checkpoint, write_json  # noqa: E402
from src.dynatree_cre.config import DynaTreeConfig  # noqa: E402
from src.dynatree_cre.diagnostic import (  # noqa: E402
    compare_checkpoints_exact,
    evaluate_checkpoint_diagnostic,
)
from src.dynatree_cre.model import DynaTreeCREModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose DynaTree-CRE checkpoint (M7.0-M7.3)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to state.pt")
    p.add_argument("--compare-checkpoint", type=str, default=None, help="Optional second state.pt for exact equality")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu, mps, auto)")
    p.add_argument("--output-json", type=str, required=True, help="Path to save diagnostic JSON")
    p.add_argument("--use-raw-retrieval", action="store_true", help="M7.2 zero-retrain raw_cosine retrieval check")
    p.add_argument("--routing-override", type=str, default=None, help="Override routing mode at diagnostic time")
    p.add_argument("--eta", type=float, default=0.0, help="Cluster-size bias eta for size_corrected_hierarchical")
    p.add_argument("--gamma", type=float, default=0.0, help="Cluster weight gamma for hybrid_tree_routing")
    p.add_argument("--allow-hub-download", action="store_true", help="Allow HF hub download if not cached")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = DynaTreeConfig.from_dict(payload["config"])
    config.training.device = args.device
    config.validate()

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
        max_tasks=int(payload["completed_task_id"]) + 1,
    )

    model = DynaTreeCREModel(config, local_files_only=not args.allow_hub_download)
    load_dynatree_checkpoint(
        ckpt_path=ckpt_path,
        config=config,
        model=model,
        dataset_fingerprint=stream.dataset_fingerprint,
        task_order_fingerprint=stream.task_order_fingerprint,
    )

    summary, records = evaluate_checkpoint_diagnostic(
        model=model,
        stream=stream,
        use_raw_retrieval=args.use_raw_retrieval,
        routing_override=args.routing_override,
        eta=args.eta,
        gamma=args.gamma,
    )

    report: dict = {
        "checkpoint": str(ckpt_path),
        "experiment_id": config.experiment.experiment_id,
        "dataset": config.experiment.dataset,
        "task_distribution": config.experiment.task_distribution,
        "diagnostic_options": {
            "use_raw_retrieval": bool(args.use_raw_retrieval),
            "routing_override": args.routing_override,
            "eta": float(args.eta),
            "gamma": float(args.gamma),
        },
        "m7_1_decomposition": summary,
    }

    if args.compare_checkpoint:
        tensor_cmp = compare_checkpoints_exact(ckpt_path, args.compare_checkpoint)
        payload_b = torch.load(Path(args.compare_checkpoint), map_location="cpu", weights_only=False)
        cfg_b = DynaTreeConfig.from_dict(payload_b["config"])
        cfg_b.training.device = args.device
        model_b = DynaTreeCREModel(cfg_b, local_files_only=not args.allow_hub_download)
        load_dynatree_checkpoint(
            ckpt_path=Path(args.compare_checkpoint),
            config=cfg_b,
            model=model_b,
            dataset_fingerprint=stream.dataset_fingerprint,
            task_order_fingerprint=stream.task_order_fingerprint,
        )
        _, records_b = evaluate_checkpoint_diagnostic(model=model_b, stream=stream)
        pred_ncm_eq = [r["pred_ncm"] for r in records] == [r["pred_ncm"] for r in records_b]
        pred_cos_eq = [r["pred_cos"] for r in records] == [r["pred_cos"] for r in records_b]
        tensor_cmp["prediction_ncm_exact_equal"] = bool(pred_ncm_eq)
        tensor_cmp["prediction_cos_exact_equal"] = bool(pred_cos_eq)
        report["m7_0_resume_equality"] = tensor_cmp

    write_json(Path(args.output_json), report)
    print(f"Saved M7 diagnostic report -> {args.output_json}")


if __name__ == "__main__":
    main()
