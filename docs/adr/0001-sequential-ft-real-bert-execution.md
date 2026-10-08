# ADR 0001: Execution Protocol for Sequential Fine-Tuning Baseline with Real BERT

## Context
Baseline B0 (Sequential Fine-Tuning) serves as the empirical Lower Bound for Catastrophic Forgetting in Continual Relation Extraction (FewRel Track A, 8 tasks, 5-shot, seed 2021). Previous test runs were conducted using a simulated model scaffold to validate the data pipeline and metrics. We need to execute the real baseline using genuine `bert-base-uncased` pretrained weights.

## Decision
1. **Device Acceleration**: Execute using Apple Silicon Metal Performance Shaders (`mps`) backend via PyTorch.
2. **Granular Checkpointing**: Save the full `model.pt` weights (~440 MB each) across all 8 individual task stages (`after_T1` .. `after_T8`, ~3.5 GB total) to allow post-hoc hidden representation inspection and catastrophic forgetting probing.
3. **Purity Boundary Invariant**: Freeze all hyper-parameters (seed 2021, learning rate $2\times 10^{-5}$, 5 epochs/task, batch size 16, $M=0$, no replay, no prototypes, seen-class masking with $-10^9$) without per-task tuning.
4. **Staged Execution**: Validate with a 2-task real smoke test first before launching the full 8-task run, then refresh registry `E001` and publication plots.
