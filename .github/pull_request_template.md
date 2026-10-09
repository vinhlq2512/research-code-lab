<!--
  Pull Request Template for AI/ML Research & Engineering
  Repository: research-code-lab
  Guidelines:
  - Fill out all applicable sections.
  - Keep prose concise and focused on What and Why.
  - HTML comments like this one will not appear in the rendered GitHub PR description.
-->

## 1. Metadata & Classification

### 1.1 Type of Change
<!-- Select all that apply by putting an 'x' in the brackets: [x] -->
- [ ] 🧪 **New Benchmark / Baseline Execution** (e.g. B0, Frozen BERT, Upper Bound)
- [ ] 🧠 **Model Architecture / Training Algorithm** (e.g. TAPTA, Prompt, Replay, Distillation)
- [ ] 📦 **Dataset Pipeline & Task Order** (e.g. FewRel, TACRED, loader, sampler)
- [ ] 📐 **Evaluation & Metrics Extension** (e.g. Evaluator, BWT, AIA, Matrix export)
- [ ] 🏛️ **Architecture Decision Record (ADR) / PRD**
- [ ] 🛠️ **Refactoring / Performance Optimization** (e.g. Feature Caching, GPU speedup)
- [ ] 🐛 **Bug Fix / Scientific Correctness Patch**
- [ ] 📚 **Documentation / Memory Notes**

### 1.2 Tracking References
- **Experiment ID:** <!-- e.g., E001, E002, E003 or N/A -->
- **Related Issue:** <!-- Closes #123, Fixes #456, or N/A -->
- **Related ADR / PRD:** <!-- e.g., docs/adr/0003-frozen-bert-linear-head.md -->
- **Experiment Registry:** <!-- Updated experiments/registry.yaml? [Yes/No/NA] -->

---

## 2. Motivation & Scientific Context (Why)

<!--
  Explain why this change is necessary:
  - What research problem or technical debt does this solve?
  - What was the core scientific hypothesis or engineering requirement?
-->
### 2.1 Problem Statement
<!-- Describe the problem or baseline gap addressed by this PR. -->

### 2.2 Core Hypothesis & Objective
<!-- State the scientific question or objective. E.g.: Decoupling representation drift from classifier interference. -->

---

## 3. Summary of Changes (What)

<!--
  Summarize the key changes across system layers.
  Be specific about files and modules modified or created.
-->
### 3.1 Architectural / Code Breakdown
- **Data & Invariants:** <!-- Changes to dataset loaders, task order, or entity markers -->
- **Model & Backbone:** <!-- Freezing scope, layer additions, classifier heads -->
- **Trainer & Runner:** <!-- Optimization loop, hyperparameters, caching mechanics -->
- **Evaluation & Persistence:** <!-- Matrices, loggers, DoD gates, plot generation -->

### 3.2 Visual Structure / Architecture Diff Sketch
<!--
  Provide a lightweight visual representation (file tree, call flow, or Mermaid diagram)
  as recommended by the @pr skill. Example:
-->
```text
[Input RelationSample: Tokens + [E1]...[/E1], [E2]...[/E2]]
                            ↓
       [Frozen BERT Backbone: requires_grad=False]
                            ↓
      [Concatenated Entity States: h_rep in R^1536]
                            ↓
     [Linear Classification Head: 1536 -> 80 classes]
                            ↓
   [Softmax with Seen-Class Masking: logits[:, seen:] = -1e9]
```

---

## 4. Empirical Evidence & Benchmark Results

<!--
  REQUIRED for benchmark and model changes.
  Provide concrete metrics comparing against existing lower and upper bounds.
-->

### 4.1 Head-to-Head Benchmark Comparison
<!-- Update the numbers below with evaluated values. Fill N/A if not applicable. -->
| Scientific Metric | This PR (e.g. E003) | Lower Bound (B0 / E001) | Upper Bound (Joint / E002) | Delta vs B0 |
| :--- | :---: | :---: | :---: | :---: |
| **Backbone Mode** | *[Frozen / Trainable]* | Full Fine-Tuning | Full Joint Fine-Tuning | — |
| **Final Average Accuracy ($AA_N$)** | **0.00%** | 8.76% | 52.12% | *+0.00%* |
| **Final Macro-F1** | **0.00%** | 9.69% | 48.62% | *+0.00%* |
| **Average Incremental Acc ($AIA$)** | **0.00%** | 18.98% | N/A | *+0.00%* |
| **Average Forgetting ($AF$)** | **0.00%** | 33.62% | N/A | *+0.00%* |
| **Backward Transfer ($BWT$)** | **0.00%** | -33.62% | N/A | *+0.00%* |
| **Most Forgotten Task Drop** | *Task (0.00%)* | T1 (-41.45%) | N/A | — |

### 4.2 Visual Artifacts & Plots
<!-- Attach or link generated plots. Screenshots or SVG/PNG embeds are highly encouraged. -->
- **Accuracy Evolution Curve:** `results/.../plots/accuracy_over_tasks.png`
- **Catastrophic Forgetting Curve:** `results/.../plots/forgetting_curve.png`
- **Interactive HTML Dashboard:** `results/.../dashboard.html`

### 4.3 Exported Artifacts & Provenance Pointers
- [ ] `performance_matrix.csv` & `performance_matrix.json` (Valid lower-triangular matrix)
- [ ] `metrics.jsonl` (Complete streaming evaluation log)
- [ ] `summary.json` (Structured summary metrics)
- [ ] `conclusion.md` (Self-contained empirical conclusion report)
- [ ] `checkpoints/` (Binary `model.pt` weights for all stages verified on disk)

---

## 5. Verification & Definition of Done (DoD)

### 5.1 Automated Quality Gates
<!-- Check the boxes that have been verified locally -->
- [ ] **Unit Tests Passed:** `.venv/bin/python3 -m pytest` passes with 0 regressions.
- [ ] **Automated DoD Validator:** `experiments/validate_*_results.py` passed with **ALL 8/8 CHECKS PASSED**.
- [ ] **Smoke Test Verified:** 2-task isolated smoke test completed cleanly before full run.
- [ ] **Real Model Weights:** Verified checkpoints contain real PyTorch weights (> 10MB each), zero silent mock fallbacks.

### 5.2 Scientific Protocol Invariants (Strict Guarantees)
- [ ] **Zero-Leakage Invariant:** $M=0$ memory buffer, zero access to past raw training samples.
- [ ] **Label Space Invariant:** 80 relations assigned global IDs $0 \dots 79$ without local re-indexing.
- [ ] **Task Order Invariant:** Strictly uses official order from `task-orders/fewrel/order_seed_2021.json`.
- [ ] **Seen-Class Masking Active:** Future classes $j \ge (t+1) \times 10$ masked with $-10^9$ during inference.

---

## 6. Merge Risk & Blast Radius

<!--
  Answer the two key risk questions from the @pr skill:
  1. Door Type: Is this easily reversible (Two-way door) or high-consequence / irreversible (One-way door)?
  2. Blast Radius: What components could potentially be broken by this change?
-->
### 6.1 Door Type
- **Door:** `[Two-way door | One-way door]`
- **Rationale:** <!-- Why is this reversible or irreversible? E.g., Adding an isolated experiment runner is a two-way door. Modifying core dataset schemas or git history is a one-way door. -->

### 6.2 Blast Radius
- **Blast Radius:** `[Isolated | Component-level | Repository-wide]`
- **Potential Ramifications:** <!-- Could this affect other baselines, future models, or CI runtime? -->

### 6.3 Breaking Changes & Deprecations
- [ ] **No breaking changes** (backward-compatible with existing baseline runners and configs).
- [ ] **Breaking change:** <!-- Describe any config schema changes, file renames, or API modifications -->

---

## 7. Reviewer Focus Guide

<!--
  Help the reviewer by highlighting the most critical or subtle lines of code:
  - Where should they look first?
  - Which invariant checks or math formulas require extra scrutiny?
-->
**Please pay close attention to:**
1. `file_path_1:line_range`: <!-- e.g., Masking logic or loss computation -->
2. `file_path_2:line_range`: <!-- e.g., Feature tensor extraction and gradient freezing -->
3. `configs/...`: <!-- e.g., Hyperparameters and seed alignment -->
