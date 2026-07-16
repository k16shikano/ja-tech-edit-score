#!/bin/bash
# DOK: held-out 推敲生成（adapter / base_norms など）→ アーティファクトへ
set -euo pipefail
cd /app

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ADAPTER="${ADAPTER:-/app/adapter}"
HELDOUT="${HELDOUT:-data/edit_sft/heldout.jsonl}"
LIMIT="${LIMIT:-64}"
# 空白区切り: adapter base_norms など
EVAL_MODES="${EVAL_MODES:-adapter base_norms}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
MAX_CHARS="${MAX_CHARS:-0}"
MIN_CHARS="${MIN_CHARS:-1}"
DEVICE="${DEVICE:-cuda}"
# V100 32GB では規範前置で fp16 全文読みが OOM しやすい
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"

EXTRA=()
if [[ "${TRUST_REMOTE_CODE:-}" == "1" ]]; then
  EXTRA+=(--trust-remote-code)
fi
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  EXTRA+=(--load-in-4bit)
fi

mkdir -p outputs/edit-sft-eval

for mode in ${EVAL_MODES}; do
  echo "=== generate mode=${mode} ==="
  python scripts/generate_edit_sft.py \
    --heldout "${HELDOUT}" \
    --base-model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --mode "${mode}" \
    --norms-file data/tech-writing-norms.md \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --min-chars "${MIN_CHARS}" \
    --max-chars "${MAX_CHARS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --out "outputs/edit-sft-eval/${mode}.jsonl" \
    "${EXTRA[@]}"
done

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"
cp -a outputs/edit-sft-eval/. "${art}/"
echo "artifacts under ${art}:"
find "${art}" -type f -ls
