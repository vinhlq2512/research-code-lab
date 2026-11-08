# Research Environment & Discovery Report

This document records the baseline environment, tooling availability, configuration directories, and existing codebase conventions for the **Continual Relation Extraction with Task-Aware Prompt Adaptation** project.

---

## 1. System & Runtime Environment

| Component | Detected Version / Path | Status |
| :--- | :--- | :--- |
| **Operating System** | macOS 15.5 (Darwin Kernel 24.5.0, `arm64` Apple Silicon) | Operational |
| **Node.js** | `v22.22.0` | Ready |
| **npm / npx** | `10.9.4` | Ready |
| **Python** | `Python 3.14.0` (with local `.venv/` virtual environment) | Ready |
| **Git** | `2.39.5 (Apple Git-154)` | Clean working tree |
| **Google Antigravity CLI (`agy`)** | `/Users/vinhlq2512/.local/bin/agy` | Available & functional |
| **OpenAI Codex CLI (`codex`)** | `/Users/vinhlq2512/.local/bin/codex` (version `0.166.0`) | Available & functional |

---

## 2. Agent Ecosystem & Directory Discovery

### 2.1 Google Antigravity (AGY) & AG Kit Setup
- **Config / State Path**: `.agents/`
- **Agent Definitions**: 20 specialist agents in `.agents/agent/` (including `orchestrator`, `project-planner`, `backend-specialist`, `debugger`, etc.).
- **Installed Skills**: 52 general software engineering and meta-cognitive skills in `.agents/skills/` (e.g., `clean-code`, `architecture`, `plan-writing`, `verify-changes`, `systematic-debugging`).
- **Memory Store**: `.agents/memory/MEMORY.md` tracking user preferences and project conventions.

### 2.2 OpenAI Codex Setup
- **Global Path**: `~/.codex/`
- **CLI Behavior**: `codex debug prompt-input` confirms Codex automatically loads `AGENTS.md` from the project root into developer context.
- **Project Discovery**: Codex has restricted read access to `.agents/` and `.codex/`. Local project skills are discovered via `.codex/skills/` or referenced directly in `AGENTS.md`.

### 2.3 Shared Root Instructions
- **File**: `AGENTS.md`
- **Current Role**: Provides issue tracker (`.scratch/`), canonical triage labels, and domain documentation pointers (`GLOSSARY.md`, `docs/adr/`).
- **Integration Target**: Expand `AGENTS.md` to define dual-agent roles (AGY for planning/orchestration, Codex for execution/verification), scientific integrity rules, skill discovery, reproducibility standards, and handoff protocols.

---

## 3. Existing Research Codebase & Assets

The workspace is pre-configured for empirical research on **Continual Relation Extraction (CRE)**:

- **Benchmark Datasets**: FewRel (80 relations, Track A) and TACRED (continual benchmarks in `dataset-pipelines/continual-relation-extraction/`).
- **Baseline Implementations**:
  - `experiments/run_sequential_ft.py`: Baseline B0 Sequential Fine-Tuning (Lower Bound, catastrophic forgetting measurement).
  - `experiments/run_joint_training.py`: Upper Bound Multitask Joint Training.
  - `experiments/validate_b0_results.py`: Rigorous statistical validation of baseline outputs.
  - `experiments/registry.yaml`: Experiment registry recording run `E001` (FewRel Track A 5-shot seed 2021 on MPS).
- **Paper Reproductions**: `paper-reproductions/ConPL/`, `paper-reproductions/continual-relation-baselines/`, `paper-reproductions/CPL/`.
- **Notebooks**: Reproducible Colab notebooks in `notebooks/` (`b0_sequential_ft.ipynb`, `upper_bound_joint.ipynb`).

---

## 4. Project Conventions & User Preferences

Based on `.agents/memory/MEMORY.md`:
1. **Phase Execution Transparency**: After each phase is executed, record a structured note detailing **WHAT** was done and the exact scientific/engineering rationale **WHY** it was done.
2. **Git Commit Authorization**: Never auto-commit git changes proactively; only commit upon explicit user command.
3. **Safety & Zero-Cost Policy**: No autonomous infinite loops, no paid external APIs, and no unapproved GPU-intensive training runs.
