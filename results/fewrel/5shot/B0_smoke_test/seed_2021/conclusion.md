# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 83.86%
- **Final Macro-F1:** 90.12%
- **Average Incremental Accuracy (AIA):** 85.66%
- **Average Catastrophic Forgetting:** 7.67%
- **Backward Transfer (BWT):** -7.67%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T1
- **Initial Performance (A_jj):** 87.46%
- **Final Performance (A_8j):** 79.80%
- **Absolute Degradation Drop:** 7.67%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
