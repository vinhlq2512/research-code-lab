#!/usr/bin/env python3
"""Aggregate Multi-Seed Evaluation Results for Continual Relation Extraction.

Reads summary.json from multiple seed runs, computes Mean +/- Std for all
evaluation metrics, and outputs clean Markdown and LaTeX comparison tables.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def safe_stats(values: list[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def aggregate_dataset_runs(dataset_dir: Path) -> dict:
    """Aggregate all seed runs in a dataset directory."""
    seed_dirs = sorted([d for d in dataset_dir.glob("seed_*") if (d / "summary.json").is_file()])
    if not seed_dirs:
        return {}

    seeds = []
    metrics = {
        "final_p5_acc": [],
        "final_hard_top1_acc": [],
        "final_soft_fusion_acc": [],
        "aia": [],
        "forgetting": [],
        "router_top1": [],
        "router_top2": [],
    }

    for s_dir in seed_dirs:
        try:
            with (s_dir / "summary.json").open("r", encoding="utf-8") as f:
                d = json.load(f)
            seed = int(s_dir.name.replace("seed_", ""))
            seeds.append(seed)

            metrics["final_p5_acc"].append(d.get("final_average_accuracy", 0.0) * 100.0)
            metrics["final_hard_top1_acc"].append(d.get("final_hard_top1_accuracy", 0.0) * 100.0)
            metrics["final_soft_fusion_acc"].append(d.get("final_soft_fusion_accuracy", 0.0) * 100.0)
            metrics["aia"].append(d.get("average_incremental_accuracy", 0.0) * 100.0)
            metrics["forgetting"].append(d.get("average_forgetting", 0.0) * 100.0)
            metrics["router_top1"].append(d.get("final_router_top1_accuracy", 0.0) * 100.0)
            metrics["router_top2"].append(d.get("final_router_top2_coverage", 0.0) * 100.0)
        except Exception as e:
            print(f"Warning: Failed to parse {s_dir / 'summary.json'}: {e}")

    summary = {
        "dataset": dataset_dir.name,
        "seeds": sorted(seeds),
        "num_seeds": len(seeds),
        "stats": {},
    }

    for k, v in metrics.items():
        m, s = safe_stats(v)
        summary["stats"][k] = {
            "mean": round(m, 2),
            "std": round(s, 2),
            "values": [round(x, 2) for x in v],
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate Multi-Seed Results for TAPTA")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Root runs directory")
    parser.add_argument("--mode", type=str, default="p4_baseline", help="Mode name (e.g. p4_baseline, p5_b, p5_full)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    root_dir = Path(args.runs_dir)
    target_modes = [args.mode] if args.mode != "all" else [d.name.replace("p5_", "") for d in root_dir.glob("p5_*")]

    all_results = {}

    print("=" * 85)
    print(f"TAPTA MULTI-SEED AGGREGATION REPORT (Runs: {root_dir})")
    print("=" * 85)

    headers = [
        "Benchmark",
        "Mode",
        "Seeds (N)",
        "Final Acc (A_last)",
        "Hard Top-1 Acc",
        "Soft Fusion Acc",
        "AIA",
        "Forgetting (F)",
    ]
    table_rows = []

    for mode in target_modes:
        mode_dir = root_dir / f"p5_{mode}"
        if not mode_dir.exists():
            continue

        for ds_dir in sorted(mode_dir.glob("*_5shot")):
            res = aggregate_dataset_runs(ds_dir)
            if not res or res["num_seeds"] == 0:
                continue

            bench_name = ds_dir.name.replace("_5shot", "").upper()
            st = res["stats"]
            n = res["num_seeds"]

            a_last_str = f"{st['final_p5_acc']['mean']:.2f} ± {st['final_p5_acc']['std']:.2f}%"
            hard_str = f"{st['final_hard_top1_acc']['mean']:.2f} ± {st['final_hard_top1_acc']['std']:.2f}%"
            soft_str = f"{st['final_soft_fusion_acc']['mean']:.2f} ± {st['final_soft_fusion_acc']['std']:.2f}%"
            aia_str = f"{st['aia']['mean']:.2f} ± {st['aia']['std']:.2f}%"
            f_str = f"{st['forgetting']['mean']:.2f} ± {st['forgetting']['std']:.2f}%"

            table_rows.append([
                bench_name,
                mode.upper(),
                f"{n} ({res['seeds']})",
                a_last_str,
                hard_str,
                soft_str,
                aia_str,
                f_str,
            ])

            all_results[f"{bench_name}_{mode}"] = res

    # Format table
    col_widths = [max(len(str(row[i])) for row in [headers] + table_rows) for i in range(len(headers))]
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])

    print("\n" + row_fmt.format(*headers))
    print("-+-".join(["-" * w for w in col_widths]))
    for row in table_rows:
        print(row_fmt.format(*row))
    print("=" * 85)

    # LaTeX Table generator
    print("\n% ==========================================================================")
    print("% LaTeX Table Code for Master Thesis")
    print("% ==========================================================================")
    print("\\begin{table}[htbp]")
    print("  \\centering")
    print("  \\caption{Multi-Seed Performance Evaluation (Mean $\\pm$ Std) on Protocol v1 Track A ($M=0$)}")
    print("  \\begin{tabular}{lcccccc}")
    print("    \\toprule")
    print("    Benchmark & Method & Seeds & Final Acc ($A_{last}$) & Hard Top-1 & AIA & Forgetting ($F$) \\\\")
    print("    \\midrule")
    for r in table_rows:
        bench, m_name, seeds, a_last, hard, soft, aia, f = r
        print(f"    {bench} & {m_name} & {seeds.split()[0]} & {a_last} & {hard} & {aia} & {f} \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Exported JSON summary to: {out_p}")


if __name__ == "__main__":
    main()
