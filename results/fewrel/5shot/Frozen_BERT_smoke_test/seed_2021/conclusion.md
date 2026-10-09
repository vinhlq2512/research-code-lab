# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 43.17%
- **Final Macro-F1:** 43.14%
- **Average Incremental Accuracy (AIA):** 63.02%
- **Average Catastrophic Forgetting:** 82.74%
- **Backward Transfer (BWT):** -82.74%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T1
- **Initial Performance (A_jj):** 82.87%
- **Final Performance (A_8j):** 0.13%
- **Absolute Degradation Drop:** 82.74%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
