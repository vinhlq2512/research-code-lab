# Agent Instructions & Collaborative Research Protocol

Repository instructions and conventions for AI coding agents (**Google Antigravity / AGY** and **OpenAI Codex**).

---

## 1. Research Objective & Constraints

### Research Topic
- **Master's Thesis Project**: **Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA / TAPA-CRE)** (overarching thesis project).
- **Active Proposed Method within TAPTA (since 2026-11-27)**: **DynaTree-CRE — Dynamic Semantic Prompt Trees for Rehearsal-Free Continual Relation Extraction with Task-Agnostic Routing** (next-generation semantic-tree method for the TAPTA Master's Thesis, advancing beyond earlier task-level iterations `P0`–`P5` / `E007`–`E018`). See [`docs/research/README.md`](file:///Users/vinhlq2512/code/research-code-lab/docs/research/README.md), [`docs/research/methods/dynatree-cre/METHOD_SPEC.md`](file:///Users/vinhlq2512/code/research-code-lab/docs/research/methods/dynatree-cre/METHOD_SPEC.md), and [`docs/research/migrations/RESEARCH_DIRECTION_MIGRATION.md`](file:///Users/vinhlq2512/code/research-code-lab/docs/research/migrations/RESEARCH_DIRECTION_MIGRATION.md).

### Core Direction
- **Benchmarks**: FewRel (80 relations, Track A 8×10 5-shot) and TACRED (40 positive relations, Track A 8×5 5-shot) under Protocol v1 (`M=0` strict rehearsal-free, zero Task ID at inference).
- **Backbone Architecture**: 100% Frozen Pre-trained Language Model (`bert-base-uncased`) in deterministic `eval()` mode for both stationary query extraction (`T1-centered cosine space`) and prompted relation representation.
- **Adaptation & Routing**: Dynamic 2-tier Semantic Prompt Tree (`Root -> Semantic Cluster -> Relation`, residual additive prompt budget `L=8`), incremental prototype-based clustering (average linkage, no Ward), and Task-Agnostic Hierarchical Soft Top-K Routing (`DT-C0..DT-C4`).
- **Evaluation Metrics**: Rigorous calculation of `Final AA`, `AIA`, `Average Forgetting`, `BWT`, routing diagnostics (`Recall@K`, `Cluster Acc`, `Entropy`), tree evolution metrics, and persistent memory/latency (`ExpandableCosineClassifier` + `PromptedNCMClassifier` reported side-by-side).

### Operational Constraints
1. **Zero-Cost Policy**: Do not activate paid cloud APIs or continuous autonomous loops without human authorization.
2. **Colab-Only Experiment Execution & Compute Gating**: **100% of real-data experiments** (smoke runs, baselines, ablations, single-seed debug runs, stress tests, and multi-seed runs) **MUST execute exclusively on Google Colab GPU (`device: cuda`)**, NEVER on the local MacBook (`mps`/`cpu`). Local execution is strictly restricted to fast offline `pytest` (`synthetic_tiny`), static validation, and post-Colab result analysis/plotting. Running Colab notebooks or cloud GPU jobs requires explicit human sign-off (Human Gate 5).
3. **Data & Baseline Preservation**: Never delete, overwrite, or mutate existing baseline results, raw datasets, or run configurations.

---

## 2. Scientific Integrity Rules

1. **No Data Fabrication**: Never record unexecuted experiments as completed or invent plausible metrics. If an experiment was not physically run on authentic hardware, its status must be `Planned` or `In Progress`.
2. **Separation of Concerns**: Hypotheses, methodologies, and raw empirical findings must reside in dedicated directories (`research/hypotheses/`, `research/methodology/`, `research/results/`).
3. **Citation Authenticity**: All cited literature in `research/literature/` or paper drafts must be verified against genuine DOIs, arXiv IDs, or PubMed/Semantic Scholar records. Never invent author lists or publication venues.
4. **Purity Boundaries**: Baselines must adhere to their formal specification. For instance, Baseline B0 Sequential Fine-Tuning strictly enforces zero replay memory and zero prototype assistance.

---

## 3. Division of Responsibilities (AGY & Codex)

| Domain | Google Antigravity (AGY) | OpenAI Codex |
| :--- | :--- | :--- |
| **Primary Focus** | Research Planning, Literature Review, Hypotheses, Orchestration, Rigor Review, Paper Writing | Model Implementation, Unit Testing, Baseline Validation, Colab CUDA Notebook Packaging & Post-Run Verification |
| **Lifecycle Stages** | R1 (Literature), R2 (Gaps), R3 (Hypotheses), R4 (Methodology Design), R9 (Writing) | R5 (Baselines), R6 (Implementation), R7 (Execution), R8 (Evaluation & Ablations) |
| **Working Mode** | Architectural & strategic synthesis | High-precision engineering & verification |
| **Isolation Boundary** | Branch `research/[slug]` or local workspace | Branch `feature/[slug]` or isolated worktree |

---

## 4. Skill Discovery & Activation

Project-local AI/ML research skills are installed at `.agents/skills/` and symlinked to `.codex/skills/`. Only skills **directly relevant to the TAPA-CRE project** are retained, eliminating unrelated domain skills (wet-lab, clinical, chemistry) to keep agent context tight and highly focused.

### Curated Project Research Skill Domains (35 Core Skills)
1. **Model Backbone & Adaptation (PLM & PEFT)**:
   - `@[transformers]`: HuggingFace model loading, tokenization, BERT/RoBERTa representations.
   - `@[peft]`: LoRA, QLoRA, Prompt Tuning, Prefix Tuning with frozen PLMs.
   - `@[bitsandbytes]`: 4-bit / 8-bit quantization for memory-efficient adapters.
   - `@[pytorch-lightning]`: Scalable PyTorch training structure.
   - `@[accelerate]`: Distributed and multi-device training abstraction (MPS/CUDA).
   - `@[optimize-for-gpu]`: GPU and Apple Silicon MPS memory optimization and profiling.
   - `@[ml-training-recipes]`: PyTorch training loops, warmup schedulers, reproducibility recipes.

2. **Continual Learning, Routing & Prototypes**:
   - `@[knowledge-distillation]`: Teacher-student distillation, EMA teacher anti-forgetting constraints.
   - `@[scikit-learn]`: Prototype clustering, centroid computation, cosine distance routing.
   - `@[umap-learn]`: 2D/3D visualization of relation prototype representations and drift.
   - `@[shap]`: Model interpretability and feature attribution for prompt tokens.

3. **Benchmarking, Data & MLOps**:
   - `@[lm-evaluation-harness]`: Standard NLP benchmarks and few-shot evaluation metrics.
   - `@[polars]`: High-performance DataFrame ETL for relation matrices $R_{i,j}$ and logs.
   - `@[tensorboard]`: Local, zero-cost, offline metrics tracking.
   - `@[weights-and-biases]`: Centralized hyperparameter and run logging.

4. **Scientific Ideation, Methodology & Review**:
   - `@[0-autoresearch-skill]`: Full research lifecycle orchestration across stages R1–R9.
   - `@[literature-review]`: Systematic literature surveys across arXiv and Semantic Scholar.
   - `@[citation-management]`: BibTeX generation, DOI verification, citation management.
   - `@[hypothesis-generation]`: Formulating testable hypotheses for prompt hierarchy and routing.
   - `@[scientific-brainstorming]`: Evidence-aware scientific ideation.
   - `@[scientific-critical-thinking]`: Rigorous evaluation of methodology and claims.
   - `@[experimental-design]`: Task scheduling, few-shot split controls, sample size justification.
   - `@[peer-review]`: Pre-submission paper review and critique.
   - `@[rigor-reviewer]`: Epistemic rigor checks and ablation validity verification.

5. **Statistical Analysis & Publication**:
   - `@[statistical-analysis]`: Formal hypothesis testing (t-tests, ANOVA) across seeds 2021, 2022, 2023.
   - `@[statistical-power]`: Statistical power and effect size calculation for $A_T$ and $F_T$.
   - `@[statsmodels]`: Statistical diagnostics and linear modeling.
   - `@[scientific-writing]`: Publication-grade manuscript drafting.
   - `@[scientific-visualization]`: Truthful, accessible figure creation.
   - `@[matplotlib]`: Fine-grained plotting of catastrophic forgetting curves.
   - `@[seaborn]`: Heatmaps and confusion matrices for sequential relation learning.
   - `@[venue-templates]`: Conference-specific LaTeX formatting (ACL, EMNLP, NeurIPS, ICML).
   - `@[scientific-slides]`: Research talk presentations and defense slide decks.
   - `@[latex-posters]`: Academic conference poster layouts.
   - `@[markdown-mermaid-writing]`: Architectural and pipeline flowcharting.

### Discovery Paths
- **AGY**: Discovers skills automatically from `.agents/skills/` and workspace rules.
- **Codex**: Discovers skills via `.codex/skills/` (relative symlink to `../.agents/skills/`) and instructions in this file.
- **Activation Rule**: Announce active skill usage prior to execution (e.g. `📚 Using skill: @[transformers] + @[peft]...`).

---

## 5. Experiment Reproducibility Requirements

1. **Deterministic Seeds**: All benchmarks must evaluate across identical seeds (`2021`, `2022`, `2023`).
2. **Task Ordering**: The relation assignment per continual task must match the deterministic schedule in `dataset-pipelines/continual-relation-extraction/`.
3. **Traceability**: Every completed run must be logged in [`experiments/registry.yaml`](file:///Users/vinhlq2512/code/research-code-lab/experiments/registry.yaml) and cross-referenced in [`workflow/EXPERIMENT_REGISTRY.md`](file:///Users/vinhlq2512/code/research-code-lab/workflow/EXPERIMENT_REGISTRY.md) with exact git commit hashes, hyperparameters, and environment specs.

---

## 6. File Ownership & Agent Handoff Procedure

1. **Handoff Medium**: File-based synchronization via [`workflow/HANDOFF.md`](file:///Users/vinhlq2512/code/research-code-lab/workflow/HANDOFF.md).
2. **Session Resumption**: An incoming agent (AGY or Codex) must read `AGENTS.md`, `workflow/STATUS.md`, and `workflow/HANDOFF.md` before starting work.
3. **Single-Writer Constraint**: To prevent write conflicts, only the assigned agent for the active task should modify source code files. Parallel work must use Git branches or separate Git worktrees.

---

## 7. Human Approval Gates

The research lifecycle incorporates 6 strict human approval gates:

1. **Gate 1 (Scope Approval)**: Literature review and problem framing validation.
2. **Gate 2 (Hypothesis Approval)**: Formal hypotheses and mathematical assumptions review.
3. **Gate 3 (Methodology Approval)**: Architectural review of prompt tree and routing mechanisms.
4. **Gate 4 (Code Review Gate)**: Source code and unit tests verified before launching runs.
5. **Gate 5 (Compute & GPU Authorization)**: Mandatory human sign-off before training on discrete GPUs or running benchmarks > 2 minutes.
6. **Gate 6 (Pre-Submission Gate)**: Verification of paper draft, figures, and claims before publication.

---

## 8. General Repository Conventions

### Issue tracker
Local markdown in `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels
Five canonical roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs
Single-context layout (`GLOSSARY.md` + `docs/adr/`). See `docs/agents/domain.md`.

### Git Commit Policy
- **Strict Prohibition**: Never execute `git commit` automatically or proactively.
- **Explicit Command Only**: Only commit when the user explicitly gives a command or request to commit.
