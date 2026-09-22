# B0 — Sequential Fine-Tuning Baseline Implementation Plan

> **Task Identifier:** `b0-sequential-ft`  
> **Target Baseline:** B0 (Sequential Fine-Tuning, Pure Lower Bound)  
> **Benchmark Protocol:** FewRel Track A, 8 tasks × 10 relations, 5-shot, seed = 2021  
> **Model Backbone:** `bert-base-uncased` (or project standard)  
> **Status:** APPROVED FOR IMPLEMENTATION (Planning Phase Complete)  
> **Date:** 2026-09-21  

---

## 0. Executive Summary & Core Research Objective

Baseline **B0 (Sequential Fine-Tuning)** provides the fundamental empirical lower bound for Continual Relation Extraction (CRE). 

### Core Question
> *Nếu chỉ fine-tune tuần tự mô hình qua từng task mà không có bất kỳ cơ chế chống quên nào, mô hình sẽ quên các quan hệ đã học ở các task trước đến mức nào?*

### Strict Purity Guarantees (No-Leakage / No-Method Rule)
To ensure a completely uncontaminated lower bound, B0 **MUST NOT** include:
- Replay buffers / Exemplar memory ($M = 0$)
- Previous-task raw training samples
- Prototypes / Centroids / Contrastive clustering
- Prompts / Prefix-tuning / Soft prompts
- Routers / Task classifiers / Mixture of Experts
- Knowledge Distillation (KD) / Feature distillation
- EMA Teachers / Weight averaging
- Parameter regularizers (EWC, SI, MAS)
- Any TAPTA-specific components (Tree, Routing, Multi-granularity Prototypes)

---

## 1. Agent Assignment & Workflow Sequence

In accordance with the AG Kit protocol:

| Agent | Responsibilities | Key Deliverables |
| :--- | :--- | :--- |
| `project-planner` | Protocol audit, config freezing, task-order verification, label mapping invariant check | `docs/b0-sequential-ft.md`, `task_order_seed_2021.json`, protocol checks |
| `test-engineer` | Evaluator gate, $A_{t,j}$ matrix verification, metric extensions (BWT, AIA, Macro-F1), unit tests | `metrics.py` additions, `test_b0_evaluator.py`, `validate_b0_results.py` |
| `backend-specialist` | Sequential trainer implementation, model head, checkpoint manager, streaming metric logger | `sequential_ft_trainer.py`, `run_sequential_ft.py`, plotting scripts |

### Execution Order & Gate Dependencies

```text
[project-planner: Audit & Protocol Freeze]
                    ↓
          [Evaluator Gate: PASS]
                    ↓
      [backend-specialist: Trainer & Head]
                    ↓
        [Phase 9: 2-Task Smoke Test]
                    ↓
   [Smoke Verification: A11, A21, A22 Valid]
                    ↓
       [Phase 10: Full T1→T8 Execution]
                    ↓
   [Phase 11: Artifact Validation & E001]
```

> 🛑 **Evaluator Gate Policy:** If evaluator tests fail or fail to produce deterministic $A_{t,j}$ evaluation cells, training MUST NOT start. The evaluator must be fixed and re-tested first.

---

## 2. Phase 0 — Codebase Audit (`B0_PRECHECK.md`)

An audit of the current workspace (`research-code-lab`) was completed:

| Component | Repository Status | Location / Details | Action for B0 |
| :--- | :--- | :--- | :--- |
| **FewRel Raw Data** | **EXISTS** | `dataset-pipelines/.../data/raw/fewrel/` (`train_wiki.json`, `val_wiki.json`, `pid2name.json`) | Reuse as-is |
| **FewRel Loader** | **EXISTS** | `src/datasets/fewrel.py` (`FewRelDataset`) | Reuse canonical loader |
| **Relation Mapping** | **EXISTS** | `data/processed/fewrel/relation_mapping.json` (80 relations $\to$ 0..79) | Reuse fixed mapping |
| **Task Order Schema** | **EXISTS** | `src/task_generation/task_order.py` | Reuse schema & validation |
| **Task Builder** | **EXISTS** | `src/task_generation/task_builder.py` (`ContinualTaskBuilder`) | Adapt for 5-shot sampling |
| **Evaluator** | **PARTIALLY USABLE** | `src/evaluation/evaluator.py` (Accuracy, Macro-F1, `RelationPredictor` protocol) | Pass gate; wrap PyTorch model |
| **Performance Matrix** | **EXISTS** | `src/evaluation/performance_matrix.py` ($A[t,j]$, CSV/JSON export) | Reuse directly |
| **CL Metrics** | **PARTIALLY USABLE** | `src/evaluation/metrics.py` (has $ACC_t$, $F_j$, Avg $F$) | **Missing:** BWT, AIA, Macro-F1 matrix, Old/New breakdown |
| **Neural Backbone / PyTorch** | **MISSING** | Host environment has Python 3.11; PyTorch/Transformers not installed in `.venv` | Install PyTorch/Transformers; build model wrapper |
| **Sequential FT Trainer** | **MISSING** | Empty `experiments/continual-relation-baselines` | Implement dedicated B0 trainer |
| **Streaming Result Logger** | **MISSING** | Need streaming `metrics.jsonl` and plotting | Implement in Phase 6 & 7 |

---

## 3. Phase 1 — Experimental Protocol & Configuration Freeze

All hyperparameters and protocol settings are frozen in `configs/b0_sequential_ft_fewrel_5shot_seed2021.yaml`.

```yaml
experiment:
  id: "B0"
  name: "sequential_ft"
  registry_id: "E001"
  description: "Sequential Fine-Tuning baseline without replay or regularization"

dataset:
  name: "fewrel"
  track: "A"
  num_tasks: 8
  relations_per_task: 10
  total_relations: 80
  train_shot: 5
  val_shot: 5
  max_test_per_relation: null  # Use full test split per relation

seed: 2021

model:
  backbone: "bert-base-uncased"
  max_seq_length: 128
  num_classes: 80
  classifier_dropout: 0.1
  classifier_type: "global_head"  # 80 classes fixed global head
  evaluation_scope: "seen_classes" # Mask predictions to currently seen classes: 0..(t+1)*10 - 1

continual:
  method: "sequential_ft"
  memory_size: 0
  replay: false
  prompt: false
  prototype: false
  router: false
  knowledge_distillation: false
  regularization: false

training:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  batch_size: 16
  epochs_per_task: 5
  warmup_ratio: 0.1
  optimizer: "AdamW"
  lr_scheduler: "linear"
  gradient_clip_norm: 1.0
  device: "auto" # mps, cuda, or cpu

evaluation:
  evaluate_seen_tasks: true
  save_per_task_metrics: true
  metric: "accuracy"
  additional_metrics: ["macro_f1", "forgetting", "bwt", "aia"]

paths:
  task_order: "task-orders/fewrel/order_seed_2021_8tasks.json"
  results_dir: "results/fewrel/5shot/B0_sequential_ft/seed_2021"
```

---

## 4. Phase 2 — Task Order & Label Space Validation

### Task Order Generation for Seed 2021 (8 Tasks × 10 Relations)
1. Alphabetically sort all 80 FewRel relation names ($P1001, P101, \dots, P991$).
2. Shuffle using isolated `random.Random(2021)`.
3. Partition cleanly into 8 tasks of exactly 10 relations:
   - $T_1: r_1 \dots r_{10}$
   - $T_2: r_{11} \dots r_{20}$
   - $\dots$
   - $T_8: r_{71} \dots r_{80}$
4. Save to `task-orders/fewrel/order_seed_2021_8tasks.json`.

### Global Label Invariant
- **Rule:** A global relation mapping (`relation_name -> global_id` from $0$ to $79$) is constructed once from `data/processed/fewrel/relation_mapping.json`.
- Relation $X$ assigned ID $k$ retains ID $k$ permanently across all stages $T_1 \dots T_8$.
- No stage-specific local re-indexing during training or evaluation.

---

## 5. Phase 3 — Evaluator Gate & Continual Metric Extensions

### 5.1 Metrics Implementation (`src/evaluation/metrics.py`)
Extend existing metrics to include:

1. **Final Average Accuracy & Final Macro-F1:**
   $$FinalAA = \frac{1}{8} \sum_{j=1}^{8} A_{8,j}, \quad FinalF1 = \frac{1}{8} \sum_{j=1}^{8} F1_{8,j}$$
2. **Average Incremental Accuracy ($AIA$):**
   $$AA_t = \frac{1}{t} \sum_{j=1}^{t} A_{t,j}, \quad AIA = \frac{1}{8} \sum_{t=1}^{8} AA_t$$
3. **Average Forgetting ($AF$):**
   $$F_j = \max_{l \in \{j,\dots,7\}} A_{l,j} - A_{8,j} \quad (j = 1 \dots 7), \quad AF = \frac{1}{7} \sum_{j=1}^{7} F_j$$
4. **Backward Transfer ($BWT$):**
   $$BWT = \frac{1}{7} \sum_{j=1}^{7} (A_{8,j} - A_{j,j})$$
5. **Old-Task vs New-Task Tracking:**
   $$Old_t = \frac{1}{t-1} \sum_{j=1}^{t-1} A_{t,j}, \quad New_t = A_{t,t} \quad (\text{for } t > 1)$$
6. **Most Forgotten Task Identification:**
   $$\text{argmax}_{j \in \{1 \dots 7\}} F_j$$

### 5.2 Evaluator Unit Test Gate
Create `tests/test_b0_evaluator_gate.py` with known test fixtures:
- Synthetic 3-task and 8-task matrices.
- Verify $ACC_t$, $F_j$, $AF$, $BWT$, $AIA$, and Old/New calculations match manual calculations.
- Verify predictor with seen-class masking does not leak future classes.

---

## 6. Phase 4 — Sequential Fine-Tuning Trainer Architecture

Implement `experiments/continual-relation-baselines/src/cl_re_baselines/` (or dedicated runner `trainers/sequential_ft_trainer.py`):

```text
[Input RelationSample: tokens + head/tail entity span markers]
                             ↓
[Tokenizer: bert-base-uncased with [E1]...[/E1], [E2]...[/E2] entity markers]
                             ↓
                 [BERT Encoder Backbone]
                             ↓
[Entity Span / [CLS] Pooling: [CLS] or concat(h_e1, h_e2)]
                             ↓
                 [Dropout (p=0.1)]
                             ↓
         [Linear Classification Head: 768 → 80]
                             ↓
[Softmax with Seen-Class Masking: mask unobserved classes j > (t * 10 + 9)]
```

### Continual Training Mechanics
- **Model Continuance:**
  - Initialize BERT once before $T_1$.
  - Train on $T_1$ dataset $\to$ save checkpoint $M_1$.
  - Load $M_1$ weights $\to$ continue fine-tuning directly on $T_2$ dataset $\to$ save checkpoint $M_2$.
  - $\dots$ continue directly without reinitializing weights.
- **Data Isolation:**
  - $T_t$ `train_loader` contains strictly samples belonging to the 10 relations of $T_t$.
  - Zero memory buffer; zero samples from $T_1 \dots T_{t-1}$.

---

## 7. Phase 5 — Checkpoint & Recovery Management

Organized checkpoint storage:
```text
results/fewrel/5shot/B0_sequential_ft/seed_2021/checkpoints/
├── stage_1_after_T1/
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── stage_2_after_T2/
...
└── stage_8_after_T8/
```
Each checkpoint includes:
- Model state dict
- Optimizer state dict (optional for evaluation recovery)
- Stage metadata & random generator state

---

## 8. Phase 6 — Streaming Metrics & Performance Matrix Persistence

### Immediate Persistence Protocol
To prevent data loss if a long run crashes mid-way:
1. After training $T_t$, evaluate each seen task $j \in \{1 \dots t\}$ immediately.
2. Append each evaluation result directly to `metrics.jsonl`:
   ```json
   {"stage": 1, "test_task": 1, "accuracy": 0.84, "macro_f1": 0.83, "sample_count": 1400}
   {"stage": 2, "test_task": 1, "accuracy": 0.48, "macro_f1": 0.45, "sample_count": 1400}
   {"stage": 2, "test_task": 2, "accuracy": 0.81, "macro_f1": 0.80, "sample_count": 1400}
   ```
3. Update `performance_matrix.json` and `performance_matrix.csv`.
4. After $T_8$, generate `summary.json`.

---

## 9. Phase 7 — Visualization & Plotting

Create plotting utility producing:
1. `plots/accuracy_over_tasks.png`:
   - X-axis: Training Stage $T_1 \dots T_8$
   - Y-axis: Average Accuracy ($AA_t$), Old-task Accuracy, New-task Accuracy
2. `plots/forgetting_curve.png`:
   - Trajectory curves of individual task performances ($A_{t,1}, A_{t,2}, \dots$) showing performance decline as new tasks are learned.

---

## 10. Phase 8 — Automated Verification Script (`validate_b0_results.py`)

A rigorous validation script verifying all Definition of Done (DoD) criteria:
- [ ] `config.yaml` exists and matches protocol ($M=0$, replay=false, seed=2021).
- [ ] `task_order.json` exists with 8 tasks, 10 relations each, disjoint.
- [ ] `metrics.jsonl` exists with all 36 evaluation entries.
- [ ] Exactly $\frac{8 \times 9}{2} = 36$ valid, non-null cells in $A_{t,j}$.
- [ ] All upper-triangular cells ($j > t$) are strictly `null` / empty.
- [ ] `summary.json` contains valid numeric values for:
  - `final_average_accuracy`
  - `final_macro_f1`
  - `average_incremental_accuracy`
  - `average_forgetting`
  - `bwt`
  - `most_forgotten_task`
- [ ] Both plots (`accuracy_over_tasks.png`, `forgetting_curve.png`) exist and are valid PNG images.

---

## 11. Phase 9 — 2-Task Smoke Test

Before launching the full 8-task training:
- Execute smoke test:
  - 2 tasks ($T_1, T_2$), 1 epoch, 5-shot, seed 2021.
- Expected outcome:
  - $A_{1,1}$ evaluated and logged.
  - $T_2$ continues from $M_1$.
  - $A_{2,1}$ and $A_{2,2}$ evaluated and logged.
  - Performance matrix has exactly 3 cells.
  - Streaming logger and recovery tested.
- **Gate:** Only proceed to Phase 10 if smoke test passes completely.

---

## 12. Phase 10 — Full T1→T8 Baseline Reproduction Run

Execute full baseline:
```bash
python experiments/run_sequential_ft.py \
  --config configs/b0_sequential_ft_fewrel_5shot_seed2021.yaml
```

---

## 13. Phase 11 — E001 Registration & Conclusion Generation

### 13.1 Experiment Registry (`experiments/registry.yaml`)
Register the experiment:
```yaml
experiments:
  E001:
    baseline: "B0"
    method: "sequential_ft"
    dataset: "fewrel"
    track: "A"
    num_tasks: 8
    relations_per_task: 10
    shot: 5
    seed: 2021
    status: "completed"
    artifacts:
      config: "results/fewrel/5shot/B0_sequential_ft/seed_2021/config.yaml"
      summary: "results/fewrel/5shot/B0_sequential_ft/seed_2021/summary.json"
      matrix_csv: "results/fewrel/5shot/B0_sequential_ft/seed_2021/performance_matrix.csv"
      matrix_json: "results/fewrel/5shot/B0_sequential_ft/seed_2021/performance_matrix.json"
      metrics_log: "results/fewrel/5shot/B0_sequential_ft/seed_2021/metrics.jsonl"
      plots: "results/fewrel/5shot/B0_sequential_ft/seed_2021/plots/"
```

### 13.2 Automated Conclusion (`results/.../conclusion.md`)
Auto-generate final empirical conclusion based strictly on real evaluated numbers (no hardcoded prose).

---

## 14. Phase X: Verification Checklist

- [ ] Evaluator Gate passes unit tests.
- [ ] Task order for seed 2021 has 8 tasks x 10 relations (80 total).
- [ ] 2-Task smoke test passes with $A_{1,1}, A_{2,1}, A_{2,2}$.
- [ ] Full T1→T8 run completes with same model continuously fine-tuned.
- [ ] 36/36 $A_{t,j}$ cells are valid floats in $[0.0, 1.0]$.
- [ ] Continual metrics calculated: Final AA, Final F1, AIA, AF, BWT.
- [ ] `validate_b0_results.py` returns exit code 0.
- [ ] E001 registered in experiment catalog.
