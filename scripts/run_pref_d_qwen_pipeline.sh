#!/usr/bin/env bash
# Qwen D ペア BT → GPM を同一デタッチジョブ内で順に実行する。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

GPM_HEAD_DIM="${GPM_HEAD_DIM:-16}"

make --no-print-directory pref-d-bt-qwen3-8b-fg "$@"
make --no-print-directory pref-d-gpm-qwen3-8b-fg GPM_HEAD_DIM="$GPM_HEAD_DIM" "$@"
