# ADR 0002: Upper Bound Joint Training Benchmark Protocol

## Context
In Continual Relation Extraction (CRE), establishing rigorous lower and upper boundaries is essential for evaluating continual learning algorithms. Having established Baseline B0 (Sequential Fine-Tuning) as the lower bound with Final Average Accuracy ($AA_8 = 8.76\%$), we require an empirical Upper Bound benchmark where all 80 classes are learned simultaneously (Joint / Multitask Training).

## Decision
1. **Dataset Alignment**: Use the identical FewRel Track A split and class groupings from `task-orders/fewrel/order_seed_2021.json` (80 relations, 5-shot: 400 train, 400 val, 11,200 test samples).
2. **Unified Data Pooling**: Pool all 400 training samples across all 80 relations into a single shuffled training pool without temporal task shifts.
3. **Architecture Parity**: Retain the exact `BERTRelationClassifier` (`bert-base-uncased`, entity markers `[E1]...[/E1]` and `[E2]...[/E2]`, linear head of 80 classes), with all 80 classes unmasked (`seen_classes_count = 80`).
4. **Optimization Budget & Validation Selection**: Train for 15 epochs with AdamW (LR $2\times 10^{-5}$, weight decay $0.01$, linear warmup $10\%$, batch size 16). Evaluate on the 400 validation samples after each epoch and retain the checkpoint with the highest validation accuracy (`best_model.pt`).
5. **Two-Tiered Evaluation**: Evaluate the best checkpoint on all 11,200 test samples to report:
   - Overall 80-way Classification Accuracy and Macro-F1.
   - Per-task test accuracy breakdown across $T_1 \dots T_8$ for head-to-head comparison against B0's initial and final per-task performance.
6. **Provenance & Registry**: Register under `E002` with `baseline_role: upper_bound` in `experiments/registry.yaml` and `registry.json`.
