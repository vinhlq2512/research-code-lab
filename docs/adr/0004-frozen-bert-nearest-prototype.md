# ADR 0004: Frozen BERT with Nearest Prototype (Cosine NCM) Benchmark Protocol

## Context
In Continual Relation Extraction (CRE), establishing clear benchmark diagnostic points between the Lower Bound (Sequential Full FT, B0, $AA_8 = 8.76\%$) and Upper Bound (Joint Training, E002, $AA_8 = 52.12\%$) is vital to decouple the underlying failure modes of continual learning:
1. **Representation Drift**: Deformation of the underlying representation space caused by parameter updates to the encoder backbone.
2. **Classifier Parameter Interference**: Overwriting of weights in a parametric classification head (e.g. linear softmax layer) when trained sequentially ($AF = 86.27\%$ in E003).
3. **Cross-Task Prototype Ambiguity**: Intrinsic geometric confusion in the feature space as new classes are introduced, when both the encoder and classifiers are non-degrading.

By freezing 100% of the BERT encoder and representing each relation class via a non-parametric centroid (prototype) $c_k = \frac{1}{|S_k|} \sum_{x \in S_k} f(x)$ computed once from few-shot training samples, we establish a clean diagnostic baseline: how well can static pretrained representations perform without any classifier parameter corruption, and how much performance drops purely due to candidate class expansion across tasks?

## Decision
1. **Benchmark Alignment**: Conform 100% to the FewRel Track A protocol with seed 2021 (8 tasks × 10 relations, 5-shot train/val, 140 test samples per relation, 11,200 test samples total).
2. **Backbone Freezing Boundary**: Completely freeze `bert-base-uncased` (all 12 Transformer layers, embedding matrices, and LayerNorm weights, `requires_grad = False`, set in `eval()` mode).
3. **Representation Mechanism**: Preserve the exact entity boundary markers `[E1]...[/E1]` and `[E2]...[/E2]`. Relation feature representation is the concatenated hidden states of the markers $[h_{e1}; h_{e2}] \in \mathbb{R}^{1536}$.
4. **Prototype Formation & Non-Parametric Inference**:
   - Each relation class $k$ is represented by a centroid vector $c_k \in \mathbb{R}^{1536}$ calculated as the arithmetic mean of its 5-shot training sample representations:
     $$c_k = \frac{1}{|S_k|} \sum_{x \in S_k} f(x)$$
   - Class prototypes are computed once when the class is first introduced in Task $t$, and are strictly static (frozen forever, zero parameter updates).
   - Inference utilizes Cosine Similarity:
     $$\hat{y} = \arg\max_{k \in \text{Seen}_t} \frac{f(x) \cdot c_k}{\|f(x)\|_2 \|c_k\|_2}$$
   - Seen-class masking: for stage $t \in \{0 \dots 7\}$, only the first $(t+1) \times 10$ classes are evaluated. Similarities for unobserved classes are clamped to $-10^9$.
5. **Efficiency via Dual-Mode Feature Caching**:
   - Features are cached in RAM / persisted to disk cache (`data/cache/features_fewrel_seed2021_frozen_bert.pt`) to enable instant execution of re-runs and smoke tests while being mathematically identical to direct BERT evaluation.
6. **Artifact Parity & Checkpoint Packaging**:
   - Save full PyTorch models `after_T{t}/model.pt` (418 MB) wrapping the frozen BERT encoder and the accumulated prototype tensor $C \in \mathbb{R}^{80 \times 1536}$ conforming to the `RelationPredictor` interface.
   - Save companion lightweight `after_T{t}/prototypes.pt` (~500 KB) for rapid direct prototype inspection.
7. **Scientific Provenance**:
   - Register under `E004` in `experiments/registry.yaml` and `registry.json` with role `non_parametric_probing`.
