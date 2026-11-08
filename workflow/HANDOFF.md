# Agent Handoff Protocol & Active Handoff (HANDOFF.md)

This file enables asynchronous, file-based collaboration between **AGY** (Google Antigravity) and **Codex** (OpenAI Codex). When switching agents, the originating agent updates this document. The incoming agent reads this file at session start to immediately acquire context, operational instructions, and task requirements without relying on prior chat history.

---

## 1. Active Handoff Document

| Field | Detail |
| :--- | :--- |
| **Handoff ID** | `HO-001` |
| **Creation Date** | 2026-11-08 |
| **Originating Agent** | AGY (Research Planning & Infrastructure) |
| **Assigned Agent** | Codex (Implementation & Engineering Verification) |
| **Related Task ID** | `T-002` (from `workflow/TASKS.md`) |
| **Objective** | Validate the local Orchestra skills integration from the Codex perspective and run lightweight test verification on existing baseline test suites. |
| **Relevant Context & Files** | - `AGENTS.md` (Project instructions & role boundaries)<br>- `.codex/skills/` (Symlinked to `.agents/skills/`)<br>- `experiments/tests/` (Lightweight baseline regression tests)<br>- `experiments/registry.yaml` (Experiment metadata) |
| **Expected Artifacts** | - Test execution summary log in `reports/smoke-test-codex.log` or verification report.<br>- Updated completion status in this document. |
| **Acceptance Criteria** | 1. Confirm `.codex/skills/peft/SKILL.md` and `.codex/skills/knowledge-distillation/SKILL.md` are accessible.<br>2. Run `pytest experiments/tests/` and confirm existing baseline tests pass.<br>3. Verify zero modification to existing baseline results or datasets.<br>4. Update Completion Status below. |
| **Completion Status** | 🟢 COMPLETED (Validated by smoke test suite & logged in `reports/smoke-test-codex.log`) |
| **Blockers / Risks** | None. Purely lightweight local smoke test; no GPU training or cloud API access required. |
| **Next Action** | Control returned to AGY to begin Stage R1 (Literature Review) on Continual Relation Extraction prompt adaptations. |

---

## 2. Handoff Template (For Future Transfers)

```markdown
### Handoff Record [HO-XXX]
- **Task ID**: [e.g., T-005]
- **Assigned Agent**: [Codex | AGY]
- **Originating Agent**: [AGY | Codex]
- **Objective**: [Clear statement of engineering or scientific goal]
- **Relevant Context & Source Files**:
  - `path/to/file1`
  - `path/to/file2`
- **Expected Artifacts**:
  - `research/methodology/...` or `experiments/...`
- **Acceptance Criteria**:
  - Criterion 1
  - Criterion 2
- **Completion Status**: [PENDING | IN_PROGRESS | COMPLETED | BLOCKED]
- **Blockers**: [None | Description of dependencies]
- **Next Action**: [Immediate step for incoming agent]
```

---

## 3. Concurrency & Isolation Rules

1. **Single Writer Rule**: Only the agent assigned in the active handoff row should modify implementation files.
2. **Parallel Work Isolation**: If parallel investigations are needed, agents must work in separate Git branches or worktrees:
   - AGY branch/worktree: `research/[topic-slug]`
   - Codex branch/worktree: `feature/[implementation-slug]`
3. **No Phantom State**: Handoffs must point to files committed or saved on disk. Never refer to ephemeral chat variables.
