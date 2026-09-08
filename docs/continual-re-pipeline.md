# Continual Relation Extraction Dataset Pipeline & Evaluator — W5–W6 Implementation Plan

## 1. Overview

This plan defines the end-to-end implementation of a reusable, model-agnostic dataset pipeline and continual-learning evaluator for Continual Relation Extraction (CRE) in `dataset-pipelines/continual-relation-extraction/`. 

The objective is to establish:
- Standardized canonical relation sample schema (`RelationSample`) with half-open span semantics `[start, end)`.
- Reusable dataset interface (`ContinualRelationDataset`) decoupling dataset logic from downstream neural models.
- Deterministic raw loader for FewRel with stable relation-to-ID mapping.
- Reproducible task-order schema and seed-based generator.
- Disjoint class-incremental task builder partitioning train, validation, and test splits without class leakage.
- Model-agnostic evaluator supporting Accuracy and Macro-F1.
- Continual evaluation performance matrix $A[t,j]$ (performance on task $j$ after training through task $t$) with CSV and JSON exporters.
- Standard continual-learning metrics: Average Accuracy ($ACC_t$) and per-task / average Forgetting ($F_j$).
- Diagnostic inspection scripts (`inspect_fewrel.py`, `inspect_tacred.py`, `generate_task_orders.py`, `validate_pipeline.py`).
- TACRED access, preprocessing feasibility check, and documentation.

---

## 2. Project Type & Scope

- **Project Type**: BACKEND (Data Engineering, Evaluation Infrastructure, Scientific Research Lab)
- **Primary Agents**: `backend-specialist` (Core logic & scripts), `test-engineer` (Verification & metrics), `project-planner` (Architecture & planning)
- **Strict Boundary**: NO neural network training loops, NO BERT/encoder dependencies, NO replay memory or prompt routing algorithms in this package.

---

## 3. Success Criteria & Definition of Done

1. **FewRel Loading & Normalization**: Raw FewRel JSON data parses reliably into `RelationSample` with validated token spans `0 <= start < end <= len(tokens)`.
2. **Stable Mapping**: Deterministic assignment of `relation -> relation_id` (alphabetically sorted), reproducible across runs.
3. **Task Order Reproducibility**: Given seed $S$ and $K$ tasks, relations are partitioned disjointly across tasks; running with identical seed produces identical JSON outputs.
4. **Task Isolation**: For all tasks $t$, train/validation/test samples contain only relations assigned to task $t$; no cross-task relation contamination.
5. **Evaluator Contract**: Minimal `RelationPredictor` protocol allowing any predictor to be evaluated with Accuracy and Macro-F1.
6. **Performance Matrix $A[t, j]$**: Strict semantics where $A[t, j]$ represents performance on task $j$ after learning task $t$. Missing cells represented as `None`/`null`.
7. **Continual Metrics**: Exact calculation of Average Accuracy $ACC_t = \frac{1}{t+1}\sum_{j=0}^t A_{t,j}$ and Forgetting $F_j = \max_{l \in \{j,\dots,T-1\}} A_{l,j} - A_{T,j}$, verified with hand-calculated test fixtures.
8. **Pipeline Validation**: `python scripts/validate_pipeline.py` executes end-to-end with a dummy predictor, exercises steps 1–15, exports CSV/JSON matrices and metrics, and returns exit code 0.
9. **TACRED Feasibility**: Comprehensive audit report in `docs/datasets/tacred-preprocessing-check.md` evaluating token spans, `no_relation` handling, and compatibility with `RelationSample`.

---

## 4. Tech Stack & Environment Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Runtime | Python >=3.10 | Native union syntax `int \| None`, modern dataclasses |
| Schemas | Standard `dataclasses(frozen=True)` | Model-agnostic, zero runtime overhead, immutability guarantees |
| Serialization | Built-in `json` & `csv` | Standardized, human-readable, zero third-party dependency |
| Randomness | Local `random.Random(seed)` | Thread-safe, avoids mutating global RNG state |
| Testing | `pytest` / `unittest` | Comprehensive unit testing with hand-crafted matrix assertions |
| Linting/Type | Built-in type hints (`typing.Protocol`, `dataclass`) | Strict contracts between components |

---

## 5. Target File Structure

```text
dataset-pipelines/
└── continual-relation-extraction/
    ├── data/
    │   ├── raw/
    │   │   ├── fewrel/
    │   │   └── tacred/
    │   └── processed/
    │       ├── fewrel/
    │       └── tacred/
    │
    ├── src/
    │   ├── datasets/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── fewrel.py
    │   │   └── tacred.py
    │   │
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   └── relation_sample.py
    │   │
    │   ├── task_generation/
    │   │   ├── __init__.py
    │   │   ├── task_order.py
    │   │   └── task_builder.py
    │   │
    │   ├── evaluation/
    │   │   ├── __init__.py
    │   │   ├── evaluator.py
    │   │   ├── performance_matrix.py
    │   │   └── metrics.py
    │   │
    │   └── utils/
    │       ├── __init__.py
    │       ├── io.py
    │       └── seed.py
    │
    ├── scripts/
    │   ├── inspect_fewrel.py
    │   ├── inspect_tacred.py
    │   ├── generate_task_orders.py
    │   └── validate_pipeline.py
    │
    ├── task-orders/
    │   ├── fewrel/
    │   └── tacred/
    │
    ├── tests/
    │   ├── __init__.py
    │   ├── test_fewrel_loader.py
    │   ├── test_task_order.py
    │   ├── test_task_builder.py
    │   ├── test_evaluator.py
    │   ├── test_performance_matrix.py
    │   └── test_metrics.py
    │
    └── README.md

docs/
└── datasets/
    └── tacred-preprocessing-check.md
```

---

## 6. Phase-by-Phase Task Breakdown

### Phase 0 — Inspect Existing Repository
- **Agent**: `project-planner`
- **Skills**: `architecture`
- **Dependencies**: None
- **INPUT**: Current workspace files, `.gitignore`, existing `cl_re_pipeline`.
- **OUTPUT**: Implementation note detailing conventions, existing FewRel raw files (`pid2name.json`, `train_wiki.json`, `val_wiki.json`), and architecture alignment.
- **VERIFY**: Check directories and ensure raw data paths are covered by `.gitignore`.

### Phase 1 — Create Dataset Pipeline Directory Structure
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 0
- **INPUT**: Clean workspace target tree.
- **OUTPUT**: Directory skeleton (`src/datasets`, `src/schemas`, `src/task_generation`, `src/evaluation`, `src/utils`, `scripts`, `task-orders/fewrel`, `tests`).
- **VERIFY**: `tree` or `ls -R` verifies directory structure.

### Phase 2 — Define Canonical Relation Sample (`relation_sample.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 1
- **INPUT**: Schema requirements (`tokens`, `head_text`, `head_start`, `head_end`, `tail_text`, `tail_start`, `tail_end`, `relation`, `relation_id`, `metadata`).
- **OUTPUT**: `src/schemas/relation_sample.py` with validation:
  - $0 \le \text{head\_start} < \text{head\_end} \le \text{len}(\text{tokens})$
  - $0 \le \text{tail\_start} < \text{tail\_end} \le \text{len}(\text{tokens})$
  - Span convention: $[start, end)$ half-open interval.
- **VERIFY**: Unit test with valid and invalid spans raises `ValueError`.

### Phase 3 — Define Base Dataset Interface (`base.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`, `api-patterns`
- **Dependencies**: Phase 2
- **INPUT**: Abstract base class requirements.
- **OUTPUT**: `src/datasets/base.py` defining `ContinualRelationDataset` with `load_train()`, `load_validation()`, `load_test()`, `get_relations()`, `get_relation_to_id()`.
- **VERIFY**: Abstract class cannot be instantiated without implementations.

### Phase 4 — Implement FewRel Raw Data Loader (`fewrel.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 2, Phase 3
- **INPUT**: `data/raw/fewrel/` raw JSON structures (`train_wiki.json`, `val_wiki.json`, `pid2name.json`).
- **OUTPUT**: `src/datasets/fewrel.py` implementing `FewRelDataset`:
  - Parses head/tail tokens and converts 0-indexed positions into $[start, end)$ spans.
  - Generates deterministic sample IDs: `fewrel_{split}_{relation}_{idx}`.
  - Validates head text matches `tokens[head_start:head_end]`.
- **VERIFY**: Load sample records and assert valid span tokens.

### Phase 5 — Stable Relation Mapping
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 4
- **INPUT**: Unique relations extracted across FewRel splits.
- **OUTPUT**: Deterministic alphabetical sorting `relation -> id` and persistence to `data/processed/fewrel/relation_mapping.json`.
- **VERIFY**: Re-running produces identical mapping dictionary and JSON hash.

### Phase 6 — FewRel Inspection Script (`inspect_fewrel.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 4, Phase 5
- **INPUT**: CLI flags (`--data-dir`).
- **OUTPUT**: `scripts/inspect_fewrel.py` printing split statistics, relation count, sample inspection, span validation errors, duplicate checks.
- **VERIFY**: Execute `python scripts/inspect_fewrel.py` and inspect console output.

### Phase 7 — Define Task Order Schema (`task_order.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 1
- **INPUT**: Schema specification (`dataset`, `seed`, `num_tasks`, `relation_count`, `tasks`).
- **OUTPUT**: `src/task_generation/task_order.py` with validation logic:
  - Tasks count match `num_tasks`.
  - Relations disjoint and cover all expected relations.
  - No unknown or duplicate relations.
- **VERIFY**: Unit tests check valid and malformed task order schemas.

### Phase 8 — Task Order Generator (`generate_task_orders.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 7
- **INPUT**: Dataset relation list, `--num-tasks`, `--seed`.
- **OUTPUT**: `scripts/generate_task_orders.py` generating `task-orders/fewrel/order_seed_{seed}.json` using isolated `random.Random(seed)`.
- **VERIFY**: Seed 42 run 1 == run 2; seed 42 != seed 43. Generate seeds 42, 43, 44.

### Phase 9 — Continual Task Builder (`task_builder.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 3, Phase 7
- **INPUT**: Dataset instance + `TaskOrder`.
- **OUTPUT**: `src/task_generation/task_builder.py` creating `ContinualTask` objects:
  - Partitions train/val/test by task relations.
  - Zero cross-task relation contamination.
- **VERIFY**: Assert `sample.relation in task.relations` across all tasks and splits.

### Phase 10 — Task Builder Inspection Output
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 9
- **INPUT**: Built tasks for FewRel.
- **OUTPUT**: Summary table / CLI output reporting per-task sample counts and relation disjunction check.
- **VERIFY**: Total unique relations across tasks equals dataset total; pairwise intersections are empty.

### Phase 11 — Generic Evaluator Interface (`evaluator.py`)
- **Agent**: `test-engineer`
- **Skills**: `clean-code`
- **Dependencies**: Phase 2
- **INPUT**: `RelationPredictor` protocol (`predict(samples) -> list[int]`).
- **OUTPUT**: `src/evaluation/evaluator.py` computing Accuracy and Macro-F1 over `EvaluationResult`.
- **VERIFY**: Test with known ground truth and predictions; check error handling on length mismatches.

### Phase 12 — Implement Performance Matrix $A[t,j]$ (`performance_matrix.py`)
- **Agent**: `test-engineer`
- **Skills**: `clean-code`
- **Dependencies**: Phase 11
- **INPUT**: Num tasks $T$.
- **OUTPUT**: `src/evaluation/performance_matrix.py`:
  - Stores performance on task $j$ after learning through task $t$.
  - Unseen tasks initialized to `None`.
  - Boundary validation: bounds checking on $t, j$.
- **VERIFY**: Set and get scores; accessing $j > t$ or out-of-bounds behaves according to contract.

### Phase 13 — Export Performance Matrix (CSV & JSON)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 12
- **INPUT**: Populated `PerformanceMatrix`.
- **OUTPUT**: Export methods to CSV (`accuracy_matrix.csv`) and JSON (`accuracy_matrix.json`) with metadata.
- **VERIFY**: Parse exported CSV and JSON; verify `None` translates to empty/`null` without turning into 0.0.

### Phase 14 — Implement Continual Learning Metrics (`metrics.py`)
- **Agent**: `test-engineer`
- **Skills**: `clean-code`
- **Dependencies**: Phase 12
- **INPUT**: `PerformanceMatrix`.
- **OUTPUT**: `src/evaluation/metrics.py`:
  - Average Accuracy: $ACC_t = \frac{1}{t+1}\sum_{j=0}^t A_{t,j}$.
  - Task Forgetting: $F_j = \max_{l \in \{j,\dots,T-1\}} A_{l,j} - A_{T,j}$.
  - Mean Forgetting: average over observed tasks.
- **VERIFY**: Unit tests using hand-calculated matrix fixture ($0.90 \to 0.70$ gives $F=0.20$).

### Phase 15 — Build Dummy Predictor
- **Agent**: `test-engineer`
- **Skills**: `clean-code`
- **Dependencies**: Phase 11
- **INPUT**: `RelationSample` lists.
- **OUTPUT**: `PerfectPredictor` (returns exact `sample.relation_id`) and `ImperfectPredictor` in test helpers.
- **VERIFY**: Perfect predictor achieves exactly 1.0 Accuracy and 1.0 Macro-F1.

### Phase 16 — End-to-End Pipeline Validation Script (`validate_pipeline.py`)
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`, `lint-and-validate`
- **Dependencies**: Phases 1–15
- **INPUT**: CLI `--dataset fewrel --task-order task-orders/fewrel/order_seed_42.json`.
- **OUTPUT**: `scripts/validate_pipeline.py` executing 8-stage validation suite:
  1. Load dataset & validate samples.
  2. Validate relation mapping.
  3. Load & validate task order.
  4. Build continual tasks.
  5. Validate relation isolation.
  6. Run evaluator with dummy predictor.
  7. Populate performance matrix $A[t,j]$.
  8. Export CSV, JSON & print metrics.
- **VERIFY**: Execute script and confirm exit code 0 and generated report artifacts.

### Phase 17 — Unit Test Suite
- **Agent**: `test-engineer`
- **Skills**: `clean-code`
- **Dependencies**: Phases 2–15
- **INPUT**: Test specifications.
- **OUTPUT**: Comprehensive unit tests:
  - `tests/test_fewrel_loader.py`
  - `tests/test_task_order.py`
  - `tests/test_task_builder.py`
  - `tests/test_evaluator.py`
  - `tests/test_performance_matrix.py`
  - `tests/test_metrics.py`
- **VERIFY**: Run test suite; 100% tests pass.

### Phase 18 — TACRED Access Check
- **Agent**: `backend-specialist`
- **Skills**: `architecture`
- **Dependencies**: Phase 0
- **INPUT**: TACRED license, source availability, file format inspection.
- **OUTPUT**: Audit of TACRED distribution, format, subject/object annotations, and presence in local workspace.
- **VERIFY**: Script detects if raw TACRED files are present or documents manual acquisition requirement.

### Phase 19 — TACRED Canonical Mapping Prototype
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: Phase 2, Phase 18
- **INPUT**: TACRED sample structures (e.g. `subj_start`, `subj_end`, `obj_start`, `obj_end`, `relation`).
- **OUTPUT**: Prototype normalization in `src/datasets/tacred.py` and `scripts/inspect_tacred.py` demonstrating mapping to `RelationSample` and treatment of `no_relation`.
- **VERIFY**: Validate synthetic or available sample converts to valid `RelationSample`.

### Phase 20 — TACRED Preprocessing Report
- **Agent**: `project-planner`
- **Skills**: `plan-writing`, `documentation-templates`
- **Dependencies**: Phases 18, 19
- **INPUT**: TACRED findings.
- **OUTPUT**: `docs/datasets/tacred-preprocessing-check.md` following standard 8-section template.
- **VERIFY**: Markdown file is complete, accurate, and structured.

### Phase 21 — README Documentation
- **Agent**: `project-planner`
- **Skills**: `documentation-templates`
- **Dependencies**: Phases 1–20
- **INPUT**: Full pipeline architecture and usage.
- **OUTPUT**: Comprehensive `dataset-pipelines/continual-relation-extraction/README.md`.
- **VERIFY**: Verify all sections (Architecture, Quick Start, $A[t,j]$ semantics, Metrics formulas) are complete.

### Phase 22 — Git Hygiene & Review
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Dependencies**: All previous phases
- **INPUT**: Workspace status.
- **OUTPUT**: Verified `.gitignore` ensuring raw data is ignored while task orders and code are tracked; no secrets or cache files staged.
- **VERIFY**: `git status` shows clean tracking state.

### Phase 23 — Final Integration & Verification
- **Agent**: `orchestrator` / `test-engineer`
- **Skills**: `lint-and-validate`
- **Dependencies**: All phases
- **INPUT**: Complete codebase.
- **OUTPUT**: End-to-end execution of `validate_pipeline.py` and test suite.
- **VERIFY**: `PIPELINE VALIDATION PASSED` output and clean test run.

---

## 7. Phase X: Final Verification Checklist

- [ ] All unit tests pass cleanly (`tests/test_*.py`).
- [ ] `scripts/inspect_fewrel.py` runs and prints valid sample & split statistics.
- [ ] `scripts/generate_task_orders.py` reproducibly outputs task order JSONs for seeds 42, 43, 44.
- [ ] `scripts/validate_pipeline.py` executes all 8 checkpoints and exits 0.
- [ ] CSV and JSON performance matrices accurately distinguish missing cells (`None`) from 0.0.
- [ ] TACRED preprocessing audit report written to `docs/datasets/tacred-preprocessing-check.md`.
- [ ] `.gitignore` prevents committing raw data.
- [ ] README.md clearly documents $A[t,j]$ semantics and usage.
