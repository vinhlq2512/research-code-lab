# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 10.33%
- **Final Macro-F1:** 10.56%
- **Average Incremental Accuracy (AIA):** 29.96%
- **Average Catastrophic Forgetting:** 86.27%
- **Backward Transfer (BWT):** -86.27%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T4
- **Initial Performance (A_jj):** 92.38%
- **Final Performance (A_8j):** 0.00%
- **Absolute Degradation Drop:** 92.38%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
