#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cre-prepare-tacred --raw-dir data/raw/tacred --out data/processed/tacred.jsonl
cre-build-tasks --config configs/tacred_10task_5shot.json
