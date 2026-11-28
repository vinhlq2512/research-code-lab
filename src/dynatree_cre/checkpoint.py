"""Checkpoint, resume, and experiment artifact persistence for DynaTree-CRE."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dynatree_cre.config import DynaTreeConfig


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if (
        "torch_mps" in state
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["torch_mps"])


def save_dynatree_checkpoint(
    output_dir: Path,
    completed_task_id: int,
    config: DynaTreeConfig,
    model: Any,
    dataset_fingerprint: str,
    task_order_fingerprint: str,
    git_info: dict[str, Any],
    run_id: str,
    matrices_state: dict[str, Any] | None = None,
    stage_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Persist full resumable state without storing raw historical training samples."""
    ckpt_dir = output_dir / "checkpoints" / f"after_T{completed_task_id + 1}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "completed_task_id": int(completed_task_id),
        "config": config.to_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "task_order_fingerprint": task_order_fingerprint,
        "git_info": git_info,
        "rng_state": capture_rng_state(),
        "model_state": model.export_state(),
        "matrices_state": matrices_state or {},
        "stage_records": stage_records or [],
    }
    ckpt_path = ckpt_dir / "state.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_dynatree_checkpoint(
    ckpt_path: Path,
    config: DynaTreeConfig,
    model: Any,
    dataset_fingerprint: str,
    task_order_fingerprint: str,
) -> dict[str, Any]:
    """Load checkpoint and enforce config, dataset, and task-order compatibility."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    saved_cfg = payload["config"]
    cur_cfg = config.to_dict()
    defaults_compat = {
        "prototype": {
            "retrieval_similarity_space": None,
            "description_weight": 0.25,
            "use_analytic_retrieval": False,
            "analytic_projection_mode": "random_relu",
            "analytic_projection_dim": 4096,
            "analytic_projection_seed": 2021,
            "analytic_ridge_lambda": 10.0,
        },
        "routing": {
            "cluster_size_eta": 0.0,
            "hybrid_cluster_gamma": 0.0,
            "inference_mode": "soft_mix",
            "rerank_fusion_mode": "fusion",
            "rerank_fusion_lambda": 0.5,
            "rerank_score_norm": "zscore",
            "rerank_forward_chunk_size": 128,
            "rerank_train_cand_k": 4,
        },
    }
    for section in ("method", "prototype", "tree", "prompt", "routing", "classifier", "memory"):
        s_sec = dict(saved_cfg.get(section) or {})
        c_sec = dict(cur_cfg.get(section) or {})
        for k, dv in defaults_compat.get(section, {}).items():
            s_sec.setdefault(k, dv)
            c_sec.setdefault(k, dv)
        if s_sec != c_sec:
            raise ValueError(
                f"Config mismatch on '{section}' during checkpoint resume."
            )
    if payload["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError("Dataset fingerprint mismatch during checkpoint resume.")
    if payload["task_order_fingerprint"] != task_order_fingerprint:
        raise ValueError("Task-order fingerprint mismatch during checkpoint resume.")

    model.restore_state(payload["model_state"])
    restore_rng_state(payload["rng_state"])
    return payload


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
