# Experiment Registry & Tracking Protocol (EXPERIMENT_REGISTRY.md)

This registry provides a human-readable index and protocol for all benchmark runs, baseline checkpoints, and proposed model evaluations. The source of truth for machine execution is [`experiments/registry.yaml`](file:///Users/vinhlq2512/code/research-code-lab/experiments/registry.yaml).

---

## 1. Registered Benchmarks & Experiments

| ID | Title / Method | Dataset & Task Split | Backbone / Device | Status | Key Results ($A_T$ / $F_T$) | Config / Logs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`E001`** | **Baseline B0: Sequential Fine-Tuning** (Lower Bound) | FewRel Track A, 8 Tasks, 10 rels/task, 5-shot, seed 2021 | `bert-base-uncased`, Apple Silicon MPS | 🟢 Completed (Real Run) | $A_T = 8.76\%$<br>$F_T = 78.4\%$ (Catastrophic Forgetting) | [`configs/b0_sequential_ft_fewrel_5shot_seed2021.yaml`](file:///Users/vinhlq2512/code/research-code-lab/configs/b0_sequential_ft_fewrel_5shot_seed2021.yaml) |
| **`E002`** | **Upper Bound: Multitask Joint Training** | FewRel Track A, All 80 relations simultaneously, 5-shot, seed 2021 | `bert-base-uncased`, Apple Silicon MPS | 🟢 Completed (Real Run) | $A_T = 84.12\%$<br>$F_T = 0.00\%$ (No forgetting, joint ceiling) | [`configs/upper_bound_joint_fewrel_5shot_seed2021.yaml`](file:///Users/vinhlq2512/code/research-code-lab/configs/upper_bound_joint_fewrel_5shot_seed2021.yaml) |
| **`E003`** | **TAPA-CRE Full Model** (Proposed) | FewRel Track A, 8 Tasks, 5-shot, seed 2021 | Frozen BERT + LoRA + Prompt Tree + Proto KD | ⚪ Planned (Pending Stage R7) | *Pending execution* | `configs/tapa_cre_fewrel_5shot_seed2021.yaml` (to be created) |
| **`E004`** | **Ablation: TAPA w/o Hierarchical Prompt Tree** | FewRel Track A, 8 Tasks, 5-shot, seed 2021 | Flat Prompt Routing + Proto KD | ⚪ Planned | *Pending execution* | `configs/ablations/tapa_no_tree.yaml` |
| **`E005`** | **Ablation: TAPA w/o EMA Teacher KD** | FewRel Track A, 8 Tasks, 5-shot, seed 2021 | Prompt Tree + Prototype Routing (No KD) | ⚪ Planned | *Pending execution* | `configs/ablations/tapa_no_ema_kd.yaml` |

---

## 2. Scientific Integrity & Execution Rules

1. **Anti-Fabrication Rule**: Under no circumstances should unexecuted benchmark runs be filled with plausible or estimated numbers. If an experiment has not finished executing on genuine hardware, status must be `Planned` or `In Progress`, and metric fields must remain `null` or `Pending`.
2. **Deterministic Reproducibility**:
   - Seed schedule: Baseline and proposed models must be evaluated on identical seeds (`2021`, `2022`, `2023`).
   - Task order: Tasks must use canonical deterministic relation orderings specified in `dataset-pipelines/continual-relation-extraction/`.
3. **Purity Boundary Declaration**:
   - Any baseline run claiming to be B0 (Sequential FT) must strictly enforce `memory_budget = 0`, `replay_buffer = false`, and `prototypes_enabled = false`.
   - The proposed model must clearly separate parameters: frozen PLM backbone vs trainable prompt/adapter weights.
4. **Compute Gate**:
   - Training runs requiring longer than 2 minutes of wall clock time or utilizing discrete GPU clusters must have explicit user authorization recorded prior to running.
