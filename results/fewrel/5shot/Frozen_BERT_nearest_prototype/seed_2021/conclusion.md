# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 59.63%
- **Final Macro-F1:** 69.35%
- **Average Incremental Accuracy (AIA):** 70.63%
- **Average Catastrophic Forgetting:** 10.45%
- **Backward Transfer (BWT):** -10.45%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T1
- **Initial Performance (A_jj):** 83.45%
- **Final Performance (A_8j):** 59.49%
- **Absolute Degradation Drop:** 23.96%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
