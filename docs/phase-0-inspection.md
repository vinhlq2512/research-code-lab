# Phase 0 Inspection Report — Continual Relation Extraction Repository

**Date:** 2026-09-08  
**Scope:** `research-code-lab/` — inspection of `dataset-pipelines/continual-relation-extraction/`, `experiments/`, `shared/`, `README.md`, `.gitignore`.

---

## 1. Executive Summary

The repository currently hosts an initial prototype in `dataset-pipelines/continual-relation-extraction/src/cl_re_pipeline/`. Raw data for FewRel (`train_wiki.json`, `val_wiki.json`, `pid2name.json`) is already downloaded and present in `data/raw/fewrel/`. However, no generic continual-learning evaluator, performance matrix $A[t, j]$, or standardized `RelationSample` schema with $[start, end)$ span semantics exists yet.

---

## 2. Detailed Findings

### 2.1 Python Package & Layout Conventions
- **Layout:** `src/` layout is used in `dataset-pipelines/continual-relation-extraction/` with `pyproject.toml` (setuptools backend).
- **Python Version:** Host environment has Python 3.11 (`/opt/homebrew/bin/python3.11`), matching `pyproject.toml` (`requires-python = ">=3.10"`).
- **Schema Preference:** Standard library `@dataclass(frozen=True)` is used across the codebase (no Pydantic dependency, zero runtime overhead).
- **Dependencies:** Minimal third-party dependencies (`certifi` for downloading, `pytest` for testing). Core pipeline relies purely on Python standard library (`json`, `csv`, `pathlib`, `dataclasses`, `random`, `hashlib`).

### 2.2 Reusable Utilities & Codebase Inspection
- `cl_re_pipeline.io_utils`: Contains clean helpers for `read_json`, `write_json`, `read_jsonl`, `write_jsonl`, `sha256_file`, `resolve_path`. These can be adapted into `src/utils/io.py`.
- `cl_re_pipeline.fetch_fewrel`: Downloads and parses FewRel data. Currently maps positions to inclusive `[start, end]`. In Phase 2 & 4, this must be converted to half-open $[start, end)$ intervals to adhere to canonical standard.
- `cl_re_pipeline.prepare_tacred`: Checks for required TACRED splits (`train.json`, `dev.json`, `test.json`) and handles `no_relation`. Currently TACRED files are not committed (as expected for licensed datasets).

### 2.3 Evaluator & Experiment Status
- **Continual Evaluator:** No evaluator exists yet. There is no predictor protocol, no $A[t,j]$ performance matrix implementation, and no $ACC_t$ / Forgetting calculation.
- **Experiments Folder:** `experiments/continual-relation-baselines/` and `shared/` are currently empty directories. There is zero duplication or conflicting baseline code.
- **Model Decoupling:** No neural network code (PyTorch, Transformers, WAVE, ConPL) is present in the pipeline, maintaining clean separation of concerns.

### 2.4 Git Hygiene & Data Storage
- `.gitignore` already ignores:
  - `**/data/raw/`
  - `**/data/processed/`
  - `**/data/tasks/`
  - `**/outputs/`, `**/runs/`, `**/wandb/`
  - `*.pyc`, `__pycache__/`, `.venv/`
- Task order files (`task-orders/fewrel/*.json`) are not ignored, which correctly preserves experiment reproducibility in git.

---

## 3. Key Decisions for W5–W6 Implementation

1. **Adopt Canonical `RelationSample`**: Implement `src/schemas/relation_sample.py` enforcing half-open token spans $[start, end)$ with strict bounds checks:
   $$0 \le start < end \le \text{len}(tokens)$$
2. **Modular Architecture**: Create clean subpackages under `src/`:
   - `schemas/`: canonical dataclasses.
   - `datasets/`: `base.py` ABC and `fewrel.py` / `tacred.py` loaders.
   - `task_generation/`: `task_order.py` schema/validation and `task_builder.py` disjoint task partitioner.
   - `evaluation/`: `evaluator.py`, `performance_matrix.py` ($A[t,j]$), and `metrics.py`.
   - `utils/`: I/O and isolated seed handling.
3. **Deterministic Randomness**: Use isolated `random.Random(seed)` instances to generate task orders without mutating global RNG state.
4. **Model-Agnostic Evaluator**: Define a clean `RelationPredictor` typing protocol (`predict(samples) -> list[int]`) so evaluation does not depend on any deep learning framework.
5. **Phase Output Location**: All documentation and markdown reports will strictly reside in `docs/` according to workspace rules.

---

## 4. Phase 0 Checklist Status

- [x] Existing project structure inspected.
- [x] Existing Python conventions identified (`dataclasses(frozen=True)`, standard library I/O, `src/` layout).
- [x] Existing reusable utilities identified (`io_utils.py` patterns).
- [x] Existing evaluator/dataset code checked for duplication (No evaluator exists; `experiments/` is empty).
- [x] `.gitignore` reviewed before adding data directories.
