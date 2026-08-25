#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cre-fetch-fewrel --raw-dir data/raw/fewrel --out data/processed/fewrel.jsonl
cre-build-tasks --config configs/fewrel_10task_5shot.json
