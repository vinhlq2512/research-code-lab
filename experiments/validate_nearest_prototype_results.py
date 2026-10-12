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


def validate_nearest_prototype(
    results_dir: Path | str,
    config_path: Path | str,
    task_order_path: Path | str,
) -> bool:
    """Rigorous Definition of Done (DoD) Validator for E004 Frozen BERT + Nearest Prototype."""
    res_dir = resolve_path(results_dir)
    print("==================================================================")
    print(f"Validating DoD for E004 Frozen BERT + Nearest Prototype at: {res_dir}")
    print("==================================================================")

    all_passed = True

    # -------------------------------------------------------------
    # Check 1: Config Protocol Invariants
    # -------------------------------------------------------------
    print("\n[Check 1/8] Verifying Experiment Config & Protocol Invariants...")
    try:
        cfg = load_config(config_path)
        assert cfg.get("seed") == 2021, f"Seed must be 2021, got {cfg.get('seed')}"
        assert cfg.get("dataset", {}).get("track") == "A", f"Track must be A, got {cfg.get('dataset', {}).get('track')}"
        assert cfg.get("dataset", {}).get("num_tasks") == 8, f"num_tasks must be 8, got {cfg.get('dataset', {}).get('num_tasks')}"
        assert cfg.get("dataset", {}).get("train_shot") == 5, f"train_shot must be 5, got {cfg.get('dataset', {}).get('train_shot')}"
        assert cfg.get("continual", {}).get("memory_size") == 0, "Memory size must be 0"
        assert cfg.get("continual", {}).get("replay") is False, "Replay must be False"
        assert cfg.get("continual", {}).get("prototype") is True, "Prototype must be True"
        assert cfg.get("model", {}).get("freeze_encoder") is True, "Backbone encoder must be frozen"
        assert cfg.get("model", {}).get("metric") == "cosine", f"Metric must be cosine, got {cfg.get('model', {}).get('metric')}"
        print("  PASS: Config file matches frozen prototype protocol (seed=2021, 8 tasks, 5-shot, frozen=true, metric=cosine).")
    except Exception as e:
        print(f"  FAIL: Config validation error: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 2: Task Order Coverage
    # -------------------------------------------------------------
    print("\n[Check 2/8] Verifying Task Order Disjointness & Coverage...")
    try:
        to_path = resolve_path(task_order_path)
        t_order = TaskOrder.load(to_path)
        assert t_order.num_tasks == 8, f"TaskOrder must have 8 tasks, got {t_order.num_tasks}"
        assert len(t_order.all_relations) == 80, f"Total relations must be 80, got {len(t_order.all_relations)}"
        assert len(set(t_order.all_relations)) == 80, "Relations in task order must be unique"
        print("  PASS: Task Order defines 8 disjoint tasks covering all 80 relations.")
    except Exception as e:
        print(f"  FAIL: Task Order validation error: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 3: Lower-Triangular Performance Matrix (36 Cells)
    # -------------------------------------------------------------
    print("\n[Check 3/8] Verifying Lower-Triangular Performance Matrix...")
    matrix_json_path = res_dir / "performance_matrix.json"
    matrix_csv_path = res_dir / "performance_matrix.csv"
    try:
        assert matrix_json_path.exists(), f"Missing {matrix_json_path}"
        assert matrix_csv_path.exists(), f"Missing {matrix_csv_path}"

        with matrix_json_path.open("r", encoding="utf-8") as f:
            m_data = json.load(f)
        matrix = m_data.get("matrix", [])
        assert len(matrix) == 8, f"Matrix rows must be 8, got {len(matrix)}"

        valid_cells = 0
        null_cells = 0
        for t in range(8):
            assert len(matrix[t]) == 8, f"Row {t} length must be 8, got {len(matrix[t])}"
            for j in range(8):
                val = matrix[t][j]
                if j <= t:
                    assert val is not None, f"Cell [{t}, {j}] must not be null"
                    assert isinstance(val, (int, float)), f"Cell [{t}, {j}] must be numeric, got {type(val)}"
                    assert 0.0 <= val <= 1.0, f"Cell [{t}, {j}] value {val} outside [0.0, 1.0]"
                    valid_cells += 1
                else:
                    assert val is None, f"Future cell [{t}, {j}] must be null, got {val}"
                    null_cells += 1

        assert valid_cells == 36, f"Expected 36 valid evaluated cells, got {valid_cells}"
        assert null_cells == 28, f"Expected 28 future null cells, got {null_cells}"
        print(f"  PASS: Performance matrix verified: exactly 36/36 lower-triangular cells in [0.0, 1.0], 28 future cells null.")
    except Exception as e:
        print(f"  FAIL: Performance matrix validation error: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 4: Streaming Metrics Log
    # -------------------------------------------------------------
    print("\n[Check 4/8] Verifying Streaming Metrics Log (metrics.jsonl)...")
    metrics_path = res_dir / "metrics.jsonl"
    try:
        assert metrics_path.exists(), f"Missing {metrics_path}"
        lines = [line.strip() for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 36, f"Expected exactly 36 evaluation entries in metrics.jsonl, got {len(lines)}"

        for idx, line in enumerate(lines):
            record = json.loads(line)
            assert "stage" in record and "test_task" in record and "accuracy" in record, f"Line {idx+1} malformed: {record}"
            assert 0.0 <= float(record["accuracy"]) <= 1.0, f"Accuracy out of range on line {idx+1}"

        print(f"  PASS: metrics.jsonl contains exactly 36 valid streamed evaluation records.")
    except Exception as e:
        print(f"  FAIL: metrics.jsonl validation error: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 5: Summary Metrics Invariants
    # -------------------------------------------------------------
    print("\n[Check 5/8] Verifying Summary Metrics File (summary.json)...")
    summary_path = res_dir / "summary.json"
    try:
        assert summary_path.exists(), f"Missing {summary_path}"
        with summary_path.open("r", encoding="utf-8") as f:
            s_data = json.load(f)

        required_keys = [
            "final_average_accuracy",
            "average_incremental_accuracy",
            "average_forgetting",
            "backward_transfer",
            "most_forgotten_task",
        ]
        for k in required_keys:
            assert k in s_data, f"Missing key '{k}' in summary.json"

        assert 0.0 <= s_data["final_average_accuracy"] <= 1.0, "final_average_accuracy outside [0, 1]"
        assert 0.0 <= s_data["average_incremental_accuracy"] <= 1.0, "average_incremental_accuracy outside [0, 1]"
        assert -1.0 <= s_data["backward_transfer"] <= 1.0, "backward_transfer outside [-1, 1]"

        print(f"  PASS: summary.json contains valid continual metrics:")
        print(f"        Final AA: {s_data['final_average_accuracy'] * 100:.2f}%")
        print(f"        AIA:      {s_data['average_incremental_accuracy'] * 100:.2f}%")
        print(f"        AF:       {s_data['average_forgetting'] * 100:.2f}%")
        print(f"        BWT:      {s_data['backward_transfer'] * 100:.2f}%")
    except Exception as e:
        print(f"  FAIL: summary.json validation error: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 6: Scientific Conclusion Report
    # -------------------------------------------------------------
    print("\n[Check 6/8] Verifying Scientific Conclusion Report (conclusion.md)...")
    conclusion_path = res_dir / "conclusion.md"
    try:
        assert conclusion_path.exists(), f"Missing {conclusion_path}"
        content = conclusion_path.read_text(encoding="utf-8")
        assert len(content) > 100, "conclusion.md is too short or empty"
        assert "Final Average Accuracy" in content or "Average Incremental Accuracy" in content, "Missing summary terms in conclusion"
        print("  PASS: conclusion.md exists with structured scientific summary.")
    except Exception as e:
        print(f"  FAIL: conclusion.md validation error: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 7: Checkpoints and Weight/Prototype Artifacts
    # -------------------------------------------------------------
    print("\n[Check 7/8] Verifying Checkpoint Directories & Weight/Prototype Files...")
    chk_dir = res_dir / "checkpoints"
    try:
        assert chk_dir.exists(), f"Missing checkpoints dir {chk_dir}"
        for t in range(1, 9):
            stage_dir = chk_dir / f"after_T{t}"
            assert stage_dir.exists(), f"Missing checkpoint stage {stage_dir.name}"
            meta_file = stage_dir / "stage_metadata.json"
            assert meta_file.exists(), f"Missing stage_metadata.json in {stage_dir.name}"

            model_pt = stage_dir / "model.pt"
            assert model_pt.exists(), f"Missing model.pt in {stage_dir.name}"
            size_mb = model_pt.stat().st_size / (1024 * 1024)
            assert size_mb > 10.0, f"model.pt in {stage_dir.name} is too small ({size_mb:.2f} MB)"

            proto_pt = stage_dir / "prototypes.pt"
            assert proto_pt.exists(), f"Missing prototypes.pt in {stage_dir.name}"

        print(f"  PASS: All 8 checkpoints exist with valid model.pt (> 10MB) and prototypes.pt.")
    except Exception as e:
        print(f"  FAIL: Checkpoint validation error: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 8: Visualization Plots
    # -------------------------------------------------------------
    print("\n[Check 8/8] Verifying Visualization Plots...")
    plots_dir = res_dir / "plots"
    try:
        assert plots_dir.exists(), f"Missing plots dir {plots_dir}"
        p1 = plots_dir / "accuracy_over_tasks.png"
        p2 = plots_dir / "forgetting_curve.png"
        assert p1.exists(), f"Missing {p1}"
        assert p2.exists(), f"Missing {p2}"
        assert p1.stat().st_size > 1000, f"Plot {p1.name} file size too small"
        assert p2.stat().st_size > 1000, f"Plot {p2.name} file size too small"
        print(f"  PASS: Both accuracy_over_tasks.png and forgetting_curve.png verified.")
    except Exception as e:
        print(f"  FAIL: Plots validation error: {e}")
        all_passed = False

    print("\n==================================================================")
    if all_passed:
        print("ALL 8 CHECKS PASSED: DEFINITION OF DONE SATISFIED FOR E004!")
        print("==================================================================")
        return True
    else:
        print("DEFINITION OF DONE VALIDATION FAILED! Check error messages above.")
        print("==================================================================")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Definition of Done Validator for E004.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/fewrel/5shot/Frozen_BERT_nearest_prototype/seed_2021",
        help="Path to results directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.json",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--task-order",
        type=str,
        default="dataset-pipelines/continual-relation-extraction/task-orders/fewrel/order_seed_2021.json",
        help="Path to task order JSON",
    )
    args = parser.parse_args()

    success = validate_nearest_prototype(
        results_dir=args.results_dir,
        config_path=args.config,
        task_order_path=args.task_order,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
