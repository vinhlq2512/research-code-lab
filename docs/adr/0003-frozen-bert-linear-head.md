# ADR 0003: Frozen BERT with Linear Head (Sequential Probing) Benchmark Protocol

## Context
In Continual Relation Extraction (CRE), establishing benchmark points between the Lower Bound (Sequential Full FT, B0, $AA_8 = 8.76\%$) and Upper Bound (Joint Training, E002, $AA_8 = 52.12\%$) is vital for diagnosing the root causes of Catastrophic Forgetting. Specifically, we want to decouple **representation drift** (distortion of pretrained BERT representations) from **classifier interference** (overwriting of weights in the linear classification head). 

By freezing 100% of the BERT encoder and sequentially training only a linear classification head across tasks, we establish a clean diagnostic baseline: how much forgetting occurs solely due to classifier parameter competition on top of a static representation space.

## Decision
1. **Benchmark Alignment**: Conform 100% to FewRel Track A protocol with seed 2021 (8 tasks × 10 relations, 5-shot train/val, 140 test samples per relation, 11,200 test samples total).
2. **Backbone Freezing Boundary**: Completely freeze `bert-base-uncased` (all 12 Transformer layers, embedding matrices, and LayerNorm weights, `requires_grad = False`, set in `eval()` mode).
3. **Representation Mechanism**: Preserve the exact entity boundary markers `[E1]...[/E1]` and `[E2]...[/E2]`. Relation feature representation is the concatenated hidden states of the markers $[h_{e1}; h_{e2}] \in \mathbb{R}^{1536}$.
4. **Classifier Head Architecture & Continual Mechanics**:
   - Single shared linear classification head: $W \in \mathbb{R}^{80 \times 1536}, b \in \mathbb{R}^{80}$.
   - Seen-class masking: for stage $t \in \{0 \dots 7\}$, only the first $(t+1) \times 10$ classes are unmasked. Logits for unobserved classes $j \ge (t+1) \times 10$ are clamped to $-10^9$.
   - The head parameters are updated sequentially across stages $T_1 \to \dots \to T_8$ without resetting between stages.
5. **Optimization Hyperparameters**:
   - Tuned specifically for Linear Probing on few-shot representations: Learning rate $1.0 \times 10^{-3}$, AdamW with weight decay $0.01$, batch size 16, 15 epochs per task.
6. **Efficiency via RAM Feature Caching**:
   - Because BERT weights are completely invariant, features are extracted once into RAM/MPS cache for train, val, and test splits.
   - Sequential training and full 36-cell matrix evaluation run on cached features, reducing execution time from ~12 minutes to under 30 seconds while being mathematically 100% identical.
   - After each task, checkpoints package the frozen encoder and trained linear head into a standard `BERTRelationClassifier` saved as `model.pt`.
7. **Scientific Provenance**:
   - Register under `E003` in `experiments/registry.yaml` and `registry.json` with role `representation_probing`.
