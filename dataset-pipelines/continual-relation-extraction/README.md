# Continual Relation Extraction Dataset Pipeline

Subproject này chuẩn bị FewRel/TACRED task streams có thể tái lập cho continual relation extraction papers như ConPL, CPL, WAVE-CRE, WAVE++, và các baseline liên quan.

## Scope

- Normalize FewRel and TACRED into one JSONL schema.
- Generate deterministic task streams with fixed relation order and k-shot sampling.
- Save manifests with config, input hash, output hashes, relation order, and per-task counts.
- Keep raw data out of Git and avoid redistributing licensed TACRED files.

## Data Sources

- FewRel: public dataset from THUNLP. The script downloads `train_wiki.json`, `val_wiki.json`, and `pid2name.json`.
- TACRED: released through LDC by Stanford NLP. Put your licensed `train.json`, `dev.json`, and `test.json` under `data/raw/tacred/`.

## Quickstart Từ Lab Root

```bash
cd /Users/vinhlq2512/code/research-code-lab
python3 -m venv .venv
.venv/bin/python -m pip install -e dataset-pipelines/continual-relation-extraction
cd dataset-pipelines/continual-relation-extraction
```

Prepare FewRel:

```bash
../../.venv/bin/python -m cl_re_pipeline.fetch_fewrel --raw-dir data/raw/fewrel --out data/processed/fewrel.jsonl
../../.venv/bin/python -m cl_re_pipeline.build_tasks --config configs/fewrel_10task_5shot.json
```

Prepare TACRED after placing licensed files in `data/raw/tacred/`:

```bash
../../.venv/bin/python -m cl_re_pipeline.prepare_tacred --raw-dir data/raw/tacred --out data/processed/tacred.jsonl
../../.venv/bin/python -m cl_re_pipeline.build_tasks --config configs/tacred_10task_5shot.json
```

Run tests:

```bash
../../.venv/bin/python -m pytest
```

## Output Layout

Each generated task directory contains:

- `train.jsonl`: k-shot training examples for current task relations.
- `val.jsonl`: validation examples for current task relations.
- `test.jsonl`: test examples for current task relations only.
- `cumulative_test.jsonl`: test examples from all seen relations up to this task.

The root output directory contains:

- `manifest.json`: reproducibility metadata and SHA256 hashes.
- `relation_order.json`: exact relation order used for the continual stream.

## Notes For Paper Reproduction

- FewRel papers commonly use balanced relation classes, so `train_shots` and `val_shots` are enforced per relation by default.
- TACRED has `no_relation`; the default config excludes it from task identities because continual relation extraction papers usually model emerging positive relation types. You can set `include_no_relation: true` if a paper requires it.
- If a dataset has too few examples for a k-shot setting, generation fails unless `allow_undersampled` is enabled in the config.
