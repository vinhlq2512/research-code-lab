# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 23.70%
- **Final Macro-F1:** 24.30%
- **Average Incremental Accuracy (AIA):** 34.43%
- **Average Catastrophic Forgetting:** 35.25%
- **Backward Transfer (BWT):** -35.25%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T1
- **Initial Performance (A_jj):** 45.16%
- **Final Performance (A_8j):** 9.91%
- **Absolute Degradation Drop:** 35.25%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
