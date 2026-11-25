# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 56.04%
- **Final Macro-F1:** 0.00%
- **Average Incremental Accuracy (AIA):** 67.45%
- **Average Catastrophic Forgetting:** 14.51%
- **Backward Transfer (BWT):** -14.51%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T3
- **Initial Performance (A_jj):** 70.60%
- **Final Performance (A_8j):** 45.70%
- **Absolute Degradation Drop:** 24.90%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
