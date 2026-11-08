# Research Task Backlog (TASKS.md)

This backlog organizes ongoing and future research tasks between **AGY** (Planning, Literature, Hypotheses, Orchestration) and **Codex** (Implementation, Testing, Experiments, Engineering).

---

## Active & Upcoming Tasks

### Task T-001: Project-Relevant Skills Setup & Pruning
- **Stage**: Infrastructure
- **Assignee**: AGY
- **Priority**: P0 (Blocker)
- **Status**: 🟢 Completed
- **Description**: Pruned 152 unrelated domain skills (wet-lab, genomics, chemistry, astronomy, quantum) and consolidated a tightly curated set of 35 research skills strictly relevant to TAPA-CRE (PLM, PEFT, Distillation, Prototype Routing, Statistics, Experiment Tracking, Paper Writing) into `.agents/skills` and `.codex/skills`.
- **Acceptance Criteria**:
  - Exactly 35 dedicated project research skills active.
  - Zero unrelated domain clutter.
  - Baseline test suites continue to pass cleanly.

### Task T-002: Dual-Agent Smoke Test & Handoff Validation
- **Stage**: Validation
- **Assignee**: Codex
- **Priority**: P0
- **Status**: 🟢 Completed
- **Description**: Codex reads `workflow/HANDOFF.md`, verifies skill access via `.codex/skills`, runs lightweight experiment registry validation test (`experiments/tests/`), and confirms operational readiness.
- **Acceptance Criteria**:
  - `experiments/tests/` passes without errors (4/4 passed).
  - Codex updates `workflow/HANDOFF.md` with completion log (`reports/smoke-test-codex.log`).

### Task T-003: Systematic Literature Review on CRE Prompting
- **Stage**: R1 (Literature Review)
- **Assignee**: AGY
- **Priority**: P1
- **Status**: ⚪ Pending
- **Description**: Synthesize recent papers on prompt-based continual learning (ConPL, CPL, Prompt-Tuning for CRE) into `research/literature/cre-prompt-survey.md`.
- **Acceptance Criteria**:
  - Compares parameter efficiency, replay memory requirements, and forgetting metrics across key baselines.
  - Citations validated against authentic sources.

### Task T-004: Formal Hypothesis Specification (TAPA-CRE)
- **Stage**: R3 (Hypotheses)
- **Assignee**: AGY
- **Priority**: P1
- **Status**: ⚪ Pending
- **Description**: Document formal mathematical hypotheses for Hierarchical Prompt Tree routing and Prototype Memory distillation.
- **Acceptance Criteria**:
  - Formulates null and alternative hypotheses on Average Accuracy and Forgetting.
  - Human Gate 2 approval recorded in `workflow/DECISIONS.md`.

### Task T-005: Prototype-Guided Soft Routing Architecture Implementation
- **Stage**: R6 (Implementation)
- **Assignee**: Codex
- **Priority**: P1
- **Status**: ⚪ Pending
- **Description**: Implement PyTorch modules for hierarchical prompt trees, soft routing via prototype cosine similarity, and EMA teacher distillation loss.
- **Acceptance Criteria**:
  - Unit tests covering forward pass, gradient flow to prompt tokens only, and frozen PLM verification.
  - Passes linting and type checking.

### Task T-006: 8-Task Continual Evaluation on FewRel Track A
- **Stage**: R7 & R8 (Experiments & Evaluation)
- **Assignee**: Codex
- **Priority**: P2
- **Status**: ⚪ Pending (Requires Human Gate 5 Approval)
- **Description**: Run 8-task sequential training with 5-shot FewRel Track A under seeds 2021, 2022, 2023. Record metrics in `experiments/registry.yaml`.
- **Acceptance Criteria**:
  - Full matrix $R_{i,j}$ logged.
  - $A_T$ and $F_T$ computed with standard formulas.
