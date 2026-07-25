#!/usr/bin/env bash
# 既存 pref + machine_neg をマージし、別 split で pref-bt 変種を学習する。
# 既存の data/pref_split/ と outputs/pref-bt/ は触らない。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

BASE_PREF="${BASE_PREF:-$ROOT/data/pref_dataset.jsonl}"
MACHINE_NEG="${MACHINE_NEG:-$ROOT/data/pref_dataset_machine_neg.jsonl}"
MERGED="${MERGED:-$ROOT/data/pref_dataset_merged_machine_neg.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-$ROOT/data/pref_split_machine_neg}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/pref-bt-machine-neg}"

EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
TRUNCATE_DIM="${TRUNCATE_DIM:-0}"
TEXT_PREFIX="${TEXT_PREFIX:-文章: }"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"

test -s "$MACHINE_NEG" || {
  echo "missing $MACHINE_NEG — run: make machine-neg-pref" >&2
  exit 1
}

if [[ "${SMOKE:-0}" == 1 ]]; then
  echo "[smoke] machine_neg only (no base pref merge)"
  cp -f "$MACHINE_NEG" "$MERGED"
else
  test -s "$BASE_PREF" || {
    echo "missing $BASE_PREF — run make train or section-pref-data first" >&2
    exit 1
  }
  echo "[1/3] merge base + machine_neg -> $MERGED"
  "$PYTHON" - <<PY "$BASE_PREF" "$MACHINE_NEG" "$MERGED" "$ROOT"
import json, sys
from pathlib import Path

root = Path(sys.argv[4])
sys.path.insert(0, str(root / "scripts"))
from build_machine_neg_pref import merge_pref

base_path, neg_path, out_path = map(Path, sys.argv[1:4])
n, tagged = merge_pref([base_path, neg_path], out_path)
print(f"merged rows: {n} (machine_neg-tagged: {tagged})")
PY
fi

echo "[2/3] split -> $SPLIT_DIR (group-by base_id, force-train machine_neg)"
SPLIT_ARGS=(--input "$MERGED" --out-dir "$SPLIT_DIR" --group-by base_id)
if [[ "${SMOKE:-0}" == 1 ]]; then
  SPLIT_ARGS+=(--train-ratio 0.65 --valid-ratio 0.30)
else
  SPLIT_ARGS+=(--force-train-label machine_neg)
fi
"$PYTHON" "$ROOT/scripts/split_pref_dataset.py" "${SPLIT_ARGS[@]}"

test -s "$SPLIT_DIR/train.jsonl" || {
  echo "empty train split: $SPLIT_DIR/train.jsonl" >&2
  exit 1
}
test -s "$SPLIT_DIR/valid.jsonl" || {
  echo "empty valid split: $SPLIT_DIR/valid.jsonl" >&2
  exit 1
}

echo "[3/3] train pref-bt -> $OUTPUT_DIR"
"$PYTHON" "$ROOT/scripts/train_pref_bt.py" \
  --model "$EMBED_MODEL" \
  --train-file "$SPLIT_DIR/train.jsonl" \
  --eval-file "$SPLIT_DIR/valid.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --truncate-dim "$TRUNCATE_DIM" \
  --text-prefix "$TEXT_PREFIX" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --batch-size "$BATCH_SIZE"
