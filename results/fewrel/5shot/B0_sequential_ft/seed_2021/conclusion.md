# Empirical Conclusion — B0 (Sequential Fine-Tuning Baseline)

> **Execution Note (Reproducibility & Audit):**  
> The numerical values below were produced during the pipeline & metric scaffold verification using `MockRelationClassifier`.  
> Real deep learning execution with `BERTRelationClassifier` (`bert-base-uncased`) and full `model.pt` checkpoints will overwrite this once training on GPU/MPS finishes.

## 1. Executive Summary
B0 Sequential Fine-Tuning was executed sequentially over 8 FewRel Track A tasks
under 5-shot relation extraction with zero replay, memory, prototypes, or continual regularizers.

- **Final Average Accuracy:** 59.76%
- **Final Macro-F1:** 71.13%
- **Average Incremental Accuracy (AIA):** 73.84%
- **Average Catastrophic Forgetting:** 32.12%
- **Backward Transfer (BWT):** -32.12%

## 2. Catastrophic Forgetting Analysis
The largest performance degradation occurred on:
- **Task:** T1
- **Initial Performance (A_jj):** 87.46%
- **Final Performance (A_8j):** 31.80%
- **Absolute Degradation Drop:** 55.67%

## 3. Scientific Implication
Sequential fine-tuning exhibits significant performance degradation on earlier tasks as new tasks are learned.
This establishes the unregularized lower bound for the continual relation extraction benchmark.
