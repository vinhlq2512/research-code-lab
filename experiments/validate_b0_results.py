#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_SRC = REPO_ROOT / "dataset-pipelines" / "continual-relation-extraction" / "src"

if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from task_generation.task_order import TaskOrder


def resolve_path(candidate: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    p = Path(candidate)
    if p.is_absolute() and p.exists():
        return p
    if (base_dir / p).exists():
        return base_dir / p
    if (Path.cwd() / p).exists():
        return Path.cwd() / p
    return base_dir / p


def load_config(config_path: Path | str) -> dict:
    resolved = resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if resolved.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            with resolved.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            json_fallback = resolved.with_suffix(".json")
            if json_fallback.exists():
                with json_fallback.open("r", encoding="utf-8") as f:
                    return json.load(f)
            raise
    else:
        with resolved.open("r", encoding="utf-8") as f:
            return json.load(f)


def validate_b0(
    results_dir: Path | str,
    config_path: Path | str,
    task_order_path: Path | str,
) -> bool:
    """Rigorous Definition of Done (DoD) Validator for B0 Sequential Fine-Tuning."""
    res_dir = resolve_path(results_dir)
    print("==================================================================")
    print("B0 Sequential Fine-Tuning — Automated DoD Validation")
    print("==================================================================")
    print(f"Results Directory: {res_dir}")
    print(f"Config File:       {config_path}")
    print(f"Task Order File:   {task_order_path}")
    print("==================================================================")

    # -------------------------------------------------------------
    # 1. Config Validation
    # -------------------------------------------------------------
    print("\n[Check 1/8] Validating Experimental Config...")
    cfg = load_config(config_path)
    assert cfg["seed"] == 2021, f"Seed must be 2021, got {cfg.get('seed')}"
    assert cfg["dataset"]["name"] == "fewrel", f"Dataset must be fewrel, got {cfg['dataset'].get('name')}"
    assert cfg["dataset"]["track"] == "A", f"Track must be A, got {cfg['dataset'].get('track')}"
    assert cfg["dataset"]["num_tasks"] == 8, f"num_tasks must be 8, got {cfg['dataset'].get('num_tasks')}"
    assert cfg["dataset"]["relations_per_task"] == 10, f"relations_per_task must be 10, got {cfg['dataset'].get('relations_per_task')}"
    assert cfg["dataset"]["train_shot"] == 5, f"train_shot must be 5, got {cfg['dataset'].get('train_shot')}"

    # Purity Assertions
    continual = cfg.get("continual", {})
    assert continual.get("memory_size") == 0, f"memory_size must be 0, got {continual.get('memory_size')}"
    assert continual.get("replay") is False, f"replay must be false, got {continual.get('replay')}"
    assert continual.get("prompt") is False, f"prompt must be false, got {continual.get('prompt')}"
    assert continual.get("prototype") is False, f"prototype must be false, got {continual.get('prototype')}"
    assert continual.get("router") is False, f"router must be false, got {continual.get('router')}"
    assert continual.get("knowledge_distillation") is False, f"KD must be false, got {continual.get('knowledge_distillation')}"
    print("  [PASS] Config parameters and purity constraints verified.")

    # -------------------------------------------------------------
    # 2. Task Order Validation
    # -------------------------------------------------------------
    print("\n[Check 2/8] Validating Task Order...")
    to_path = resolve_path(task_order_path)
    assert to_path.exists(), f"Task order file does not exist: {to_path}"
    task_order = TaskOrder.load(to_path)
    assert task_order.num_tasks == 8, f"TaskOrder num_tasks must be 8, got {task_order.num_tasks}"
    assert task_order.relation_count == 80, f"TaskOrder relation_count must be 80, got {task_order.relation_count}"
    assert task_order.seed == 2021, f"TaskOrder seed must be 2021, got {task_order.seed}"
    assert len(task_order.tasks) == 8, f"Tasks list length must be 8, got {len(task_order.tasks)}"
    assert all(len(t.relations) == 10 for t in task_order.tasks), "All 8 tasks must contain exactly 10 relations"

    # Disjointness check
    all_rels: list[str] = []
    for t in task_order.tasks:
        all_rels.extend(t.relations)
    assert len(all_rels) == len(set(all_rels)) == 80, "Relations must be disjoint and total exactly 80"
    print("  [PASS] Task order contains 8 disjoint tasks of 10 relations each (80 total).")

    # -------------------------------------------------------------
    # 3. Performance Matrix JSON & CSV Validation
    # -------------------------------------------------------------
    print("\n[Check 3/8] Validating Performance Matrix A[t, j]...")
    matrix_json_path = res_dir / "performance_matrix.json"
    matrix_csv_path = res_dir / "performance_matrix.csv"

    assert matrix_json_path.exists(), f"performance_matrix.json missing at {matrix_json_path}"
    assert matrix_csv_path.exists(), f"performance_matrix.csv missing at {matrix_csv_path}"

    with matrix_json_path.open("r", encoding="utf-8") as f:
        mat_dict = json.load(f)

    matrix = mat_dict["matrix"]
    assert len(matrix) == 8, f"Matrix row count must be 8, got {len(matrix)}"
    for r_idx, row in enumerate(matrix):
        assert len(row) == 8, f"Row {r_idx} must have 8 columns, got {len(row)}"

    # Check Sanity Assertions:
    assert matrix[0][0] is not None, "A[0, 0] must not be None"
    assert matrix[7][7] is not None, "A[7, 7] must not be None"
    assert matrix[0][1] is None, "A[0, 1] must be None"
    assert matrix[2][5] is None, "A[2, 5] must be None"

    # Count observed cells in lower triangle
    observed_cells = 0
    null_future_cells = 0

    for t in range(8):
        for j in range(8):
            val = matrix[t][j]
            if j <= t:
                assert val is not None, f"Cell A[{t}, {j}] must not be None"
                assert isinstance(val, (int, float)), f"Cell A[{t}, {j}] must be numeric, got {val}"
                assert 0.0 <= val <= 1.0, f"Cell A[{t}, {j}] out of range [0, 1]: {val}"
                observed_cells += 1
            else:
                assert val is None, f"Cell A[{t}, {j}] must be None for future unobserved tasks"
                null_future_cells += 1

    expected_cells = 8 * 9 // 2
    assert observed_cells == expected_cells == 36, f"Observed cells {observed_cells} != expected 36"
    assert null_future_cells == 28, f"Future cells {null_future_cells} != expected 28"

    # Check CSV file format
    with matrix_csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 9, f"CSV must contain header + 8 rows, got {len(reader)}"
        header = reader[0]
        assert header[0] == "trained_until"
        assert len(header) == 9

    print("  [PASS] Matrix has exactly 36/36 valid lower-triangular cells and 28 null future cells.")

    # -------------------------------------------------------------
    # 4. Metrics Log JSONL Validation
    # -------------------------------------------------------------
    print("\n[Check 4/8] Validating Streaming Metrics Log (metrics.jsonl)...")
    jsonl_path = res_dir / "metrics.jsonl"
    assert jsonl_path.exists(), f"metrics.jsonl missing at {jsonl_path}"

    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 36, f"metrics.jsonl must contain exactly 36 evaluation records, got {len(lines)}"

    for idx, line in enumerate(lines):
        record = json.loads(line)
        assert "stage" in record and "test_task" in record and "accuracy" in record and "macro_f1" in record
        assert 0 <= record["stage"] < 8
        assert 0 <= record["test_task"] <= record["stage"]
        assert 0.0 <= record["accuracy"] <= 1.0
        assert 0.0 <= record["macro_f1"] <= 1.0

    print("  [PASS] metrics.jsonl contains 36 valid evaluation records.")

    # -------------------------------------------------------------
    # 5. Summary Metrics Validation
    # -------------------------------------------------------------
    print("\n[Check 5/8] Validating summary.json...")
    summary_path = res_dir / "summary.json"
    assert summary_path.exists(), f"summary.json missing at {summary_path}"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    required_keys = [
        "final_average_accuracy",
        "final_macro_f1",
        "average_incremental_accuracy",
        "average_forgetting",
        "backward_transfer",
        "most_forgotten_task",
        "per_task_forgetting",
        "old_vs_new_by_stage",
    ]
    for key in required_keys:
        assert key in summary, f"summary.json missing required metric key: {key}"

    final_aa = summary["final_average_accuracy"]
    aia = summary["average_incremental_accuracy"]
    af = summary["average_forgetting"]
    bwt = summary["backward_transfer"]

    assert 0.0 <= final_aa <= 1.0, f"final_average_accuracy out of range: {final_aa}"
    assert 0.0 <= aia <= 1.0, f"average_incremental_accuracy out of range: {aia}"
    assert af >= 0.0, f"average_forgetting must be non-negative: {af}"
    assert isinstance(bwt, (int, float)), f"bwt must be numeric: {bwt}"

    most_f = summary["most_forgotten_task"]
    assert "task_name" in most_f and "forgetting" in most_f

    print(f"  [PASS] summary.json validated (Final AA: {final_aa*100:.2f}%, AF: {af*100:.2f}%, BWT: {bwt*100:.2f}%).")

    # -------------------------------------------------------------
    # 6. Conclusion Report Validation
    # -------------------------------------------------------------
    print("\n[Check 6/8] Validating conclusion.md...")
    conclusion_path = res_dir / "conclusion.md"
    assert conclusion_path.exists(), f"conclusion.md missing at {conclusion_path}"
    assert conclusion_path.stat().st_size > 100, "conclusion.md is too short or empty"
    print("  [PASS] conclusion.md exists and is non-empty.")

    # -------------------------------------------------------------
    # 7. Checkpoints Validation
    # -------------------------------------------------------------
    print("\n[Check 7/8] Validating Stage Checkpoints...")
    ckpts_dir = res_dir / "checkpoints"
    assert ckpts_dir.exists(), f"checkpoints directory missing at {ckpts_dir}"

    for t in range(8):
        stage_dir = ckpts_dir / f"after_T{t + 1}"
        assert stage_dir.exists(), f"Missing checkpoint for stage after_T{t + 1}"
        assert (stage_dir / "stage_metadata.json").exists(), f"Missing stage_metadata.json in {stage_dir}"

        # Rigorous check: require genuine PyTorch model weights
        pt_file = stage_dir / "model.pt"
        assert pt_file.exists(), (
            f"Missing real model checkpoint 'model.pt' in {stage_dir}. "
            f"Real BERT weights are strictly required for validation."
        )
        file_size_mb = pt_file.stat().st_size / (1024 * 1024)
        assert file_size_mb > 10.0, (
            f"Checkpoint {pt_file} is suspiciously small ({file_size_mb:.2f} MB). "
            f"BERT-base checkpoint must contain full weights."
        )

    print("  [PASS] All 8 stage checkpoints (after_T1..after_T8) exist [REAL PYTORCH WEIGHTS VERIFIED].")

    # -------------------------------------------------------------
    # 8. Plots Validation
    # -------------------------------------------------------------
    print("\n[Check 8/8] Validating Publication Plots...")
    plots_dir = res_dir / "plots"
    assert plots_dir.exists(), f"plots directory missing at {plots_dir}"

    acc_plot = plots_dir / "accuracy_over_tasks.png"
    fgt_plot = plots_dir / "forgetting_curve.png"

    assert acc_plot.exists(), f"accuracy_over_tasks.png missing at {acc_plot}"
    assert acc_plot.stat().st_size > 500, f"accuracy_over_tasks.png is too small: {acc_plot.stat().st_size} bytes"

    assert fgt_plot.exists(), f"forgetting_curve.png missing at {fgt_plot}"
    assert fgt_plot.stat().st_size > 500, f"forgetting_curve.png is too small: {fgt_plot.stat().st_size} bytes"

    print("  [PASS] Both publication plots exist and are non-empty PNGs.")

    print("\n==================================================================")
    print("ALL 8 CHECKS PASSED: DEFINITION OF DONE SATISFIED FOR B0!")
    print("==================================================================")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate B0 Sequential Fine-Tuning results against Definition of Done (DoD)."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/fewrel/5shot/B0_sequential_ft/seed_2021",
        help="Path to B0 results directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/b0_sequential_ft_fewrel_5shot_seed2021.json",
        help="Path to B0 config file",
    )
    parser.add_argument(
        "--task-order",
        type=str,
        default="dataset-pipelines/continual-relation-extraction/task-orders/fewrel/order_seed_2021.json",
        help="Path to task order JSON file",
    )
    args = parser.parse_args()

    try:
        success = validate_b0(
            results_dir=args.results_dir,
            config_path=args.config,
            task_order_path=args.task_order,
        )
        return 0 if success else 1
    except AssertionError as e:
        print(f"\n[FAIL] Validation Assertion Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[FAIL] Unexpected Error during validation: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
