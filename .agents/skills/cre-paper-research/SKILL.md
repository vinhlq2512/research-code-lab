---
name: cre-paper-research
description: Research and manage papers for continual relation extraction and task-aware prompt adaptation in the user's Knowledge Hub vault. Use for paper discovery, metadata verification, deep paper reading, literature synthesis, gap analysis, or converting verified methods into experiment specifications.
---

# CRE Paper Research

## Canonical locations and scope

- Vault root: `/Users/vinhlq2512/Obisidian/Knowledge Hub`
- Paper notes: `20 - Research/Papers`
- PDFs: `00 - Sources/PDFs/Papers`
- Reading logs: `20 - Research/Paper Reading`
- Literature notes: `20 - Research/Literature Notes`
- Concepts: `04 - Concepts`
- Research questions: `20 - Research/Research Questions`
- Experiment summaries: `20 - Research/Experiments`
- Weekly reports: `20 - Research/Weekly Reports`
- Dashboard and workflow: `20 - Research/CRE Research Dashboard.md`, `20 - Research/CRE AI Research Workflow.md`

Read `AGENTS.md` and `20 - Research/CRE AI Research Workflow.md` first. Prefer direct Markdown filesystem access. Never edit `.obsidian`, install plugins, or overwrite/relabel user-authored content. Use the Obsidian CLI only if it responds; filesystem is the supported fallback.

## Paper discovery and deduplication

1. Search vault notes, PDFs, Zotero exports/references, and project reading queue before creating a paper note.
2. Search Semantic Scholar/arXiv via `papers-skill`; use `hugging-face-papers` where appropriate. Validate metadata against primary venue pages and DOI/Crossref/arXiv records.
3. Deduplicate in order: normalized DOI; arXiv base ID with versions treated as one work; normalized title plus author/year; citekey. Link versions to one canonical note unless they materially differ.
4. When metadata only is verified, create only metadata and source links from `[[CRE AI Paper Note]]`. Never invent abstract, method, result, contribution, limitations, or gap.

## Deep reading

1. Resolve DOI/URL/title to the exact paper; inspect the local PDF first. Record physical page count and extraction quality.
2. Ground technical claims in PDF page/section references. Inspect rendered pages for equations, figures, and tables when extraction loses structure.
3. Explain symbols, assumptions, objective terms, algorithm steps, task protocol, and evidence limits. For CRE capture task-ID access, task order, class/task-incremental setting, shots, memory, model, seed/repeats, metric/averaging, parameter growth, and inference cost.
4. Label statements `reported`, `observed`, `inferred`, or `reproduced`. Only use `reproduced` for a real run with traceable config/artifacts.
5. Preserve `reading_status`, personal notes, reflections, ratings, and reading logs unless the user explicitly supplied a change.

## Literature synthesis and research gaps

- Read canonical paper notes plus source evidence; do not synthesize from abstracts alone.
- Compare protocol-compatible evidence only. Keep numbers from unlike settings separate.
- Reconcile with existing canonical CRE literature notes and question files before creating new notes.
- Describe gaps as evidence-backed open questions or hypotheses, not established facts. State the source set, search boundary, conflicting results, and what experiment could falsify each proposed gap.

## Paper-to-experiment handoff

Create a specification in the Git research repository, not the vault, covering source claims, protocol, model/data flow, equations, config, assumptions, implementation tasks, acceptance criteria, tests, metrics, and reproducibility artifacts. Link the resulting small experiment summary from `20 - Research/Experiments`. Keep code, checkpoints, logs, and large outputs in Git/experiment storage.

## Vault write rules

- Reuse existing folders and canonical notes; use Obsidian `[[WikiLinks]]`.
- Keep property names lowercase `snake_case` and explanations Vietnamese.
- Before changes over five files, state a concrete file plan.
- Create new files only within the listed authorized research folders or template/skill folders.
- Preserve existing text; do not delete, rename, or replace user-authored sections.
