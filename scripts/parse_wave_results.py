#!/usr/bin/env python3
"""Parse WAVE-CRE result pickles into a protocol-v1 metrics.json.

Upstream (mrshaw01/AdaptivePromptingCRE @5273b70, methods/manager.py) writes after
each task t a pickle ``task_{t}.pickle`` containing:

    [[n_test_1, hits_1], ..., [n_test_t, hits_t], {"task": t, "results": test_total}]

* ``hits_j`` is a length-4 array returned by ``evaluate_strict_model`` for the
  SWAG classifiers on test data of task j:
    0: task-predictor classifier accuracy on relation labels (no prompt)
    1: task-ID / pool-selection accuracy
    2: prompted accuracy with PREDICTED pool  -> main, no oracle task ID
    3: prompted accuracy with ORACLE pool (pool chosen from gold label) -> oracle
* ``test_total`` alternates NON-SWAG and SWAG cumulative (size-weighted) accuracy
  arrays: [nonswag_t1, swag_t1, nonswag_t2, swag_t2, ...].

Only the SWAG per-task matrix A[t][j] is recoverable from the pickles; per-task
NON-SWAG values exist only in stdout.

Caveat: upstream wraps evaluation in a bare ``except`` that drops failed samples
from the denominator; this parser cannot detect that. Check stdout/stderr.

Usage:
    python scripts/parse_wave_results.py RESULT_DIR --out metrics.json [--meta k=v ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys

COMPONENTS = {
    0: "task_predictor_relation_acc",
    1: "task_id_acc",
    2: "predicted_pool_acc",
    3: "oracle_pool_acc",
}


def _to_list(x):
    return [float(v) for v in list(x)]


def load(result_dir: str):
    paths = glob.glob(os.path.join(result_dir, "task_*.pickle"))
    if not paths:
        sys.exit(f"no task_*.pickle in {result_dir}")
    by_t = {}
    for p in paths:
        t = int(re.search(r"task_(\d+)\.pickle$", p).group(1))
        with open(p, "rb") as f:
            by_t[t] = pickle.load(f)
    steps = sorted(by_t)
    if steps != list(range(len(steps))):
        sys.exit(f"missing task pickles: found {steps}")
    return [by_t[t] for t in steps]


def build(records):
    n_tasks = len(records)
    out = {
        "source_format": "wave-cre-upstream-pickle",
        "n_tasks_completed": n_tasks,
        "component_index": COMPONENTS,
        "test_sizes": None,
        "swag": {name: [] for name in COMPONENTS.values()},
        "cumulative": {"nonswag": {n: [] for n in COMPONENTS.values()},
                       "swag": {n: [] for n in COMPONENTS.values()}},
    }
    for t, rec in enumerate(records):
        per_task, summary = rec[:-1], rec[-1]
        assert summary["task"] == t, (summary["task"], t)
        assert len(per_task) == t + 1, (len(per_task), t)
        sizes = [int(n) for n, _ in per_task]
        if out["test_sizes"] is None or len(sizes) > len(out["test_sizes"]):
            out["test_sizes"] = sizes
        for k, name in COMPONENTS.items():
            out["swag"][name].append([float(h[k]) for _, h in per_task])
        totals = summary["results"]
        assert len(totals) == 2 * (t + 1), len(totals)
    # cumulative series from the final summary (contains all steps)
    totals = records[-1][-1]["results"]
    for t in range(n_tasks):
        ns, sw = _to_list(totals[2 * t]), _to_list(totals[2 * t + 1])
        for k, name in COMPONENTS.items():
            out["cumulative"]["nonswag"][name].append(ns[k])
            out["cumulative"]["swag"][name].append(sw[k])

    # derived metrics from SWAG predicted-pool matrix (main, no oracle)
    A = out["swag"]["predicted_pool_acc"]
    sizes = out["test_sizes"]
    acc_whole = [sum(a * n for a, n in zip(row, sizes)) / sum(sizes[: len(row)]) for row in A]
    acc_avg = [sum(row) / len(row) for row in A]
    forgetting = None
    if n_tasks > 1:
        T = n_tasks - 1
        forgetting = sum(max(A[i][j] for i in range(j, T)) - A[T][j] for j in range(T)) / T
        bwt = sum(A[T][j] - A[j][j] for j in range(T)) / T
    else:
        bwt = None
    out["derived_swag_predicted_pool"] = {
        "accuracy_whole_per_step": acc_whole,
        "accuracy_average_per_step": acc_avg,
        "final_accuracy_whole": acc_whole[-1],
        "average_incremental_accuracy_whole": sum(acc_whole) / len(acc_whole),
        "average_forgetting": forgetting,
        "bwt": bwt,
        "final_oracle_minus_predicted_gap": (
            out["cumulative"]["swag"]["oracle_pool_acc"][-1]
            - out["cumulative"]["swag"]["predicted_pool_acc"][-1]
        ),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", nargs="*", default=[], help="key=value provenance fields")
    a = ap.parse_args()
    metrics = build(load(a.result_dir))
    metrics["meta"] = dict(kv.split("=", 1) for kv in a.meta)
    metrics["result_dir"] = os.path.abspath(a.result_dir)
    with open(a.out, "w") as f:
        json.dump(metrics, f, indent=2)
    d = metrics["derived_swag_predicted_pool"]
    print(f"tasks={metrics['n_tasks_completed']} final_acc_whole={d['final_accuracy_whole']:.4f} "
          f"forgetting={d['average_forgetting']} oracle_gap={d['final_oracle_minus_predicted_gap']:.4f}")


if __name__ == "__main__":
    main()
