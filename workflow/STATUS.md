# Research Project Status (STATUS.md)

**Project Title**: Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPA-CRE)  
**Last Updated**: 2026-11-08  
**Current Phase**: Infrastructure Setup & Pipeline Integration  
**Active Lead Agent**: AGY (Google Antigravity)  
**Partner Agent**: Codex (OpenAI Codex)  

---

## 1. Executive Summary

- Curated project-relevant research skills ecosystem (35 core skills covering PLM/PEFT, Continual Learning, EMA Distillation, Statistical Testing, MLOps, and Paper Writing) installed locally in `.agents/skills` and symlinked to `.codex/skills`. All 150+ unrelated domain skills (wet-lab, genomics, chemistry, astronomy) pruned to keep agent context focused.
- Environment discovery verified on macOS 15.5 (`arm64`), Python 3.14, Node v22.22.0.
- Existing baselines preserved: Baseline B0 (Sequential FT, Run E001) and Upper Bound Joint Training.
- Shared workspace structure (`research/`, `workflow/`, `docs/`) fully established.
- Formal file-based handoff mechanism active between AGY and Codex.

---

## 2. Research Pipeline Progress

| Stage | Stage Name | Owner | Status | Gate Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | Literature Review | AGY | 🟡 In Progress | Gate 1: Human scope confirmation |
| **R2** | Gap Identification | AGY | ⚪ Pending R1 | Automatic verification |
| **R3** | Hypothesis Generation | AGY | ⚪ Pending R2 | Gate 2: Hypothesis approval |
| **R4** | Methodology Design | AGY | ⚪ Pending R3 | Gate 3: Math & architecture approval |
| **R5** | Baseline Implementation | Codex | 🟢 Completed | Validated (`experiments/run_sequential_ft.py`) |
| **R6** | Proposed Method Code | Codex | ⚪ Pending R4 | Gate 4: Code review approval |
| **R7** | Experiment Execution | Codex | ⚪ Pending R6 | Gate 5: Compute & GPU execution approval |
| **R8** | Evaluation & Ablation | Codex + AGY | ⚪ Pending R7 | Automatic metric verification |
| **R9** | Scientific Writing | AGY | ⚪ Pending R8 | Gate 6: Pre-submission human review |

---

## 3. Active Handoff

- **Active Handoff ID**: `HO-001`
- **From**: AGY (Research Planning & Setup)
- **To**: Codex (Smoke Test & Engineering Validation)
- **Status**: 🟢 Completed (Logged in `reports/smoke-test-codex.log`)
- **Next Transition**: AGY initiates Stage R1 Literature Review (`workflow/TASKS.md` Task `T-003`)
- **Artifact**: `workflow/HANDOFF.md`

---

## 4. Current Blockers & Risks

- **Compute Gating**: Any full multi-task GPU run (e.g. 8-task FewRel Track A with full BERT backbone) must receive explicit human confirmation prior to invocation.
- **Dependency Drift**: PyTorch and HuggingFace Transformers versions must be pinned in `.venv` to prevent subtle changes in random seeding across tasks.
