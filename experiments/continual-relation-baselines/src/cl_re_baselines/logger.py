from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StreamingMetricsLogger:
    """Streams evaluation records directly to JSONL and manages summary.json.

    Immediate flush policy ensures no evaluation data is lost even if training is interrupted.
    """

    def __init__(self, log_dir: Path | str, jsonl_name: str = "metrics.jsonl", append: bool = False) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / jsonl_name
        self.summary_path = self.log_dir / "summary.json"
        if not append and self.jsonl_path.exists():
            self.jsonl_path.unlink()

    def log_evaluation(
        self,
        stage: int,
        test_task: int,
        accuracy: float,
        macro_f1: float,
        sample_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a single task evaluation record to metrics.jsonl."""
        record = {
            "stage": stage,
            "stage_name": f"after_T{stage + 1}",
            "test_task": test_task,
            "test_task_name": f"T{test_task + 1}",
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "sample_count": int(sample_count),
        }
        if metadata:
            record["metadata"] = metadata

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def save_summary(self, summary_data: dict[str, Any]) -> None:
        """Write final summary metrics to summary.json."""
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_data, handle, indent=2)


def generate_conclusion_report(
    summary: dict[str, Any],
    performance_matrix: list[list[float | None]],
    output_path: Path | str,
) -> None:
    """Generate automatic empirical conclusion markdown report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_aa = summary.get("final_average_accuracy", 0.0)
    final_f1 = summary.get("final_macro_f1", 0.0)
    aia = summary.get("average_incremental_accuracy", 0.0)
    af = summary.get("average_forgetting", 0.0)
    bwt = summary.get("backward_transfer", 0.0)
    most_forgotten = summary.get("most_forgotten_task", {})

    content = f"""# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** {final_aa * 100:.2f}%
- **Final Macro-F1:** {final_f1 * 100:.2f}%
- **Average Incremental Accuracy (AIA):** {aia * 100:.2f}%
- **Average Catastrophic Forgetting:** {af * 100:.2f}%
- **Backward Transfer (BWT):** {bwt * 100:.2f}%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** {most_forgotten.get('task_name', 'N/A')}
- **Initial Performance (A_jj):** {float(most_forgotten.get('initial_score', 0.0)) * 100:.2f}%
- **Final Performance (A_8j):** {float(most_forgotten.get('final_score', 0.0)) * 100:.2f}%
- **Absolute Degradation Drop:** {float(most_forgotten.get('forgetting', 0.0)) * 100:.2f}%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
"""
    with output_path.open("w", encoding="utf-8") as f:
        f.write(content)
