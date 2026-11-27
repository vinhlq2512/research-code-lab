# Research Architecture Decisions (DECISIONS.md)

> **Cập nhật 2026-11-27 (Research Direction Migration):** Kể từ ngày 2026-11-27, các quyết định kiến trúc cho hướng nghiên cứu active **DynaTree-CRE** được ghi nhận theo chuẩn Research Decision Records tại [`docs/research/decisions/RD-001`..`RD-012`](../docs/research/decisions/RD-001-migrate-to-dynatree-cre.md). Các ADR lịch sử của TAPTA dưới đây được giữ nguyên để truy vết.

This log records major technical and methodological decisions for the Continual Relation Extraction (CRE) project, documenting the rationale, trade-offs, and consensus between AGY and Codex.

---

## Decision ADR-001: Adoption of Frozen PLM Backbone with Modular PEFT Adapters

- **Date**: 2026-11-08
- **Status**: Accepted
- **Author**: AGY & Codex
- **Context**: In continual learning, fine-tuning the entire backbone across non-stationary task distributions causes representation distortion (semantic drift) across earlier relation representations.
- **Decision**: Keep the base language model (`bert-base-uncased` / `roberta-base`) strictly frozen. All task-specific learning is channeled through parameter-efficient modules (LoRA adapters and prompt tokens).
- **Consequences**:
  - Positive: Guarantees zero catastrophic forgetting inside the foundational feature representation layer; drastically reduces trainable parameters (< 1.5% of backbone).
  - Negative: Requires careful dynamic routing to select appropriate adapter/prompt combinations during inference when task identity is unknown.

---

## Decision ADR-002: Hierarchical Prompt Tree with Prototype Soft Routing

- **Date**: 2026-11-08
- **Status**: Accepted
- **Author**: AGY
- **Context**: Relation extraction tasks exhibit natural ontological hierarchies (e.g., entity categories -> coarse domain -> fine-grained predicate). Flat prompt tuning treats all relations independently, failing to share coarse domain representations.
- **Decision**: Structure prompt tokens into a two-level tree:
  1. *Coarse Domain Prompts*: Shared across semantically related relation groups.
  2. *Fine-Grained Relation Prompts*: Specific to individual predicates.
  Routing is performed via cosine similarity between instance [CLS] representations and clustering prototypes stored in Prototype Memory.
- **Consequences**:
  - Positive: Enables forward transfer for new relations in familiar domains; minimizes parameter expansion.
  - Verification: Requires ablation study comparing Hierarchical Tree against Flat Prompt Pools (Task `T-006`).

---

## Decision ADR-003: Dual-Agent Division of Labor (AGY Orchestration, Codex Implementation)

- **Date**: 2026-11-08
- **Status**: Accepted
- **Author**: Research Infrastructure Architect
- **Context**: Autonomous AI research requires both high-level scientific reasoning (literature synthesis, hypothesis formulation, rigor review, paper writing) and rigorous software engineering (PyTorch modules, testing, experiment execution, profiling).
- **Decision**:
  - **AGY** is primary lead for stages R1 (Literature), R2 (Gaps), R3 (Hypotheses), R4 (Methodology Design), and R9 (Paper Writing).
  - **Codex** is primary lead for stages R5 (Baselines), R6 (Implementation), R7 (Execution), and R8 (Evaluation metrics & validation).
- **Handoff Mechanism**: File-based synchronization through `workflow/HANDOFF.md`, `workflow/STATUS.md`, and `workflow/TASKS.md`.
