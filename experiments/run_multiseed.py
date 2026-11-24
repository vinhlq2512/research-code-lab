#!/usr/bin/env python3
"""Multi-Seed Experiment Runner & Aggregator for TAPTA P4.

Runs TAPTA P4 over multiple seeds, aggregates results into mean +/- std,
and exports publication-grade summary tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_val = sum(values) / n
    if n < 2:
        return mean_val, 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return mean_val, math.sqrt(var)


def run_multiseed(
    seeds: list[int],
    base_config: str,
    device: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    print("==================================================================")
    print(f"TAPTA P4 MULTI-SEED EXPERIMENT RUNNER: {len(seeds)} Seeds {seeds}")
    print(f"Base Config: {base_config}")
    print(f"Execution Device: {device or 'default from config'}")
    print("==================================================================")

    results_per_seed: dict[int, dict[str, Any]] = {}

    for s_idx, seed in enumerate(seeds):
        print(f"\n>>> Running Seed {seed} ({s_idx + 1}/{len(seeds)})...")
        seed_result_path = (
            REPO_ROOT / f"runs/p4_hierarchical_tree/fewrel_5shot/seed_{seed}/summary.json"
        )

        if not dry_run:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "experiments/run_p4.py"),
                "--config",
                base_config,
            ]
            if device:
                cmd.extend(["--device", device])
            # Pass seed override if supported, or run_p4 can accept --seed
            print(f"  Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        if seed_result_path.exists():
            with seed_result_path.open("r", encoding="utf-8") as f:
                results_per_seed[seed] = json.load(f)
            print(f"  Loaded results for Seed {seed}.")
        else:
            print(f"  Warning: Result file not found for Seed {seed}: {seed_result_path}")

    # Aggregate metrics across seeds
    metric_keys = [
        "final_decision_fusion_accuracy",
        "final_decision_fusion_topk_gated_accuracy",
        "final_tree_top1_prototype_accuracy",
        "final_tree_top1_gated_prototype_accuracy",
        "average_incremental_accuracy",
        "average_forgetting",
        "final_router_top1_accuracy",
        "final_router_topk_coverage",
        "router_top1_aia",
        "router_topk_aia",
        "oracle_gap_closed_pct",
    ]

    aggregated: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        vals = [results_per_seed[s][k] for s in results_per_seed if k in results_per_seed[s]]
        if vals:
            mean_v, std_v = compute_mean_std(vals)
            if "pct" in k:
                # Already percentage (e.g. 34.51)
                formatted = f"{mean_v:.2f} ± {std_v:.2f}%"
            elif "forgetting" in k:
                # Fraction (0.1189 -> 11.89%)
                formatted = f"{mean_v * 100:.2f} ± {std_v * 100:.2f}%"
            else:
                formatted = f"{mean_v * 100:.2f} ± {std_v * 100:.2f}%"

            aggregated[k] = {
                "mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "formatted": formatted,
            }

    summary_payload = {
        "benchmark": "FewRel Track A (8x10, 5-shot, M=0)",
        "seeds_evaluated": list(results_per_seed.keys()),
        "num_seeds": len(results_per_seed),
        "aggregated_metrics": aggregated,
        "per_seed_results": results_per_seed,
    }

    out_dir = REPO_ROOT / "runs/p4_hierarchical_tree/fewrel_5shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "multiseed_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Markdown Summary
    md_lines = [
        "# Bảng Tổng Hợp Kết Quả Đa Hạt Giống (Multi-Seed Aggregation)",
        "",
        f"- **Mô hình:** TAPTA (P4) Hierarchical Prompt Tree & Decision Fusion",
        f"- **Benchmark:** FewRel Track A (8 tasks × 10 relations, strict 5-shot, $M=0$)",
        f"- **Số lượng hạt giống đã chạy:** {len(results_per_seed)} seeds ({list(results_per_seed.keys())})",
        "",
        "| Chỉ số Đánh giá | Trung bình ± Độ lệch chuẩn (Mean ± Std) | Đơn vị |",
        "|---|:---:|:---:|",
    ]
    for k, v in aggregated.items():
        clean_name = k.replace("_", " ").title()
        md_lines.append(f"| **{clean_name}** | **{v['formatted']}** | % |")

    out_md = out_dir / "multiseed_summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved multi-seed summary to: {out_json}")
    print(f"Saved multi-seed markdown to: {out_md}")
    print("\n" + "\n".join(md_lines))

    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Seed Runner for TAPTA P4")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2021],
        help="List of seeds to execute (e.g. --seeds 2021 2022 2023)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/p4_hierarchical_prompt_tree_fewrel_5shot_seed2021.yaml",
        help="Base configuration YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device ('cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only aggregate existing seed outputs without executing runs",
    )
    args = parser.parse_args()

    run_multiseed(seeds=args.seeds, base_config=args.config, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
