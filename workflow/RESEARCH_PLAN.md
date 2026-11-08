# Continual Relation Extraction: Task-Aware Prompt Adaptation (TAPA-CRE)

## Executive Research Plan

### 1. Research Context & Problem Statement
Continual Relation Extraction (CRE) aims to learn novel relation classes sequentially across arriving tasks without catastrophically forgetting previously mastered relations. Standard Sequential Fine-Tuning exhibits catastrophic forgetting, with accuracy dropping precipitously on prior tasks (as evidenced by baseline benchmark E001 where average accuracy degraded to 8.76%).

This research investigates **Task-Aware Prompt Adaptation (TAPA-CRE)** on **FewRel** and **TACRED** benchmarks under the following architectural formulation:
1. **Frozen Pre-trained Language Model (PLM)**: BERT / RoBERTa backbone with parameters frozen to eliminate parameter drift.
2. **PEFT Adaptations (LoRA / QLoRA)**: Low-Rank Adapters for task-specific adaptation under minimal parameter budgets.
3. **Hierarchical Prompt Tree**: Multi-level prompt hierarchy structuring relation semantics from coarse domain clusters down to fine-grained relations.
4. **Dynamic Soft Routing with Prototypes**: Input-guided routing network evaluating instance embeddings against relation prototypes to soft-select relevant prompt combinations.
5. **Prototype Memory**: Memory-bounded buffer maintaining centroids of relation representations in feature space without raw replay vulnerabilities.
6. **Knowledge Distillation & EMA Teacher**: Exponential Moving Average (EMA) teacher model providing soft-label distillation constraints to penalize catastrophic feature drift on historical relations.
7. **Evaluation Protocol**: Formal measurement of **Average Accuracy ($A_T$)** and **Average Forgetting ($F_T$)** across identical seeds and standard task orders.

---

## 2. Nine-Stage Research Pipeline (R1–R9)

| Stage | Focus | Primary Agent | Key Skills | Primary Output | Human Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Literature Review | AGY | `0-autoresearch-skill` | `research/literature/` synthesis | Gate 1 (Scope) |
| **R2** | Gap Identification | AGY | `rigor-reviewer`, `brainstorming-research-ideas` | Problem boundary doc | None |
| **R3** | Hypothesis Generation | AGY | `brainstorming-research-ideas`, `creative-thinking-for-research` | `research/hypotheses/` | Gate 2 (Hypothesis) |
| **R4** | Methodology Design | AGY | `peft`, `knowledge-distillation`, `architecture` | `research/methodology/` design | Gate 3 (Methodology) |
| **R5** | Baseline Implementation | Codex | `clean-code`, `accelerate`, `verify-changes` | Baseline runner & tests | None |
| **R6** | Proposed Method Implementation | Codex | `peft`, `bitsandbytes`, `ml-training-recipes` | Model code & unit tests | Gate 4 (Code Review) |
| **R7** | Experiment Execution | Codex | `accelerate`, `tensorboard`, `weights-and-biases` | Raw runs & checkpoints | Gate 5 (GPU/Compute) |
| **R8** | Evaluation & Ablation | Codex + AGY | `lm-evaluation-harness`, `academic-plotting` | Metric tables & plots | None |
| **R9** | Scientific Writing | AGY | `ml-paper-writing`, `academic-plotting` | Paper draft in `research/papers/` | Gate 6 (Submission) |

---

## 3. Milestones & Deliverables

- **Milestone 1 (Infrastructure & Baseline Verification)**:
  - Verify baseline E001 (Sequential FT) results on FewRel Track A (5-shot, seed 2021).
  - Verify Upper Bound Joint Training benchmark E002.
- **Milestone 2 (Methodology & Architecture Formulation)**:
  - Complete mathematical specification of Hierarchical Prompt Tree and Dynamic Soft Routing.
  - Implement prototype memory update rules and EMA distillation loss formulation.
- **Milestone 3 (Core Implementation & Unit Validation)**:
  - Codex implements modular prompt adapter layer, prototype router, and EMA loss wrapper.
  - Run fast sanity tests on dummy tensors before full training.
- **Milestone 4 (Empirical Benchmarking & Ablation Study)**:
  - Execute 8-task FewRel Track A and TACRED benchmark across 3 standard random seeds (2021, 2022, 2023).
  - Execute ablations: (w/o Prompt Tree, w/o Dynamic Routing, w/o KD EMA, w/o Prototype Memory).
- **Milestone 5 (Publication Artifact Compilation)**:
  - Generate publication-quality SVG/PDF performance curves and confusion matrices.
  - Compile ACL/NeurIPS formatted paper draft in `research/papers/`.
