#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python3"
PREF="${ROOT}/data/pref_dataset.jsonl"
OUT="${ROOT}/outputs/eval_xproject"
mkdir -p "$OUT"

run() {
  local name="$1" model="$2" trunc="$3" prefix="$4" batch="$5" maxlen="$6"
  local safe
  safe=$(echo "$name" | tr '/:' '__')
  echo "======== START $name (max_seq=$maxlen) ========"
  "$PY" scripts/eval_pref_xproject.py \
    --input "$PREF" \
    --model "$model" \
    --truncate-dim "$trunc" \
    --text-prefix "$prefix" \
    --batch-size "$batch" \
    --max-seq-length "$maxlen" \
    --report "$OUT/${safe}.json"
  echo "======== DONE $name ========"
}

# GLuCoSE-v2 は現行 transformers で tokenizer 非互換のため除外
run "multilingual-e5-base" "intfloat/multilingual-e5-base" 0 "passage: " 32 512
run "ruri-v3-70m" "cl-nagoya/ruri-v3-70m" 0 "文章: " 32 512
run "ruri-v3-130m" "cl-nagoya/ruri-v3-130m" 0 "文章: " 16 512

"$PY" scripts/summarize_embed_compare.py
