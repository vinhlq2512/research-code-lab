# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

## 1. Executive Summary
B0 Sequential Fine-Tuning was trained sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 59.27%
- **Final Macro-F1:** 0.00%
- **Average Incremental Accuracy (AIA):** 69.71%
- **Average Catastrophic Forgetting:** 11.91%
- **Backward Transfer (BWT):** -11.91%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T3
- **Initial Performance (A_jj):** 70.00%
- **Final Performance (A_8j):** 48.40%
- **Absolute Degradation Drop:** 21.60%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
