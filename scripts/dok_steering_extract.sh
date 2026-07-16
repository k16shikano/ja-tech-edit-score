#!/bin/bash
# DOK タスク用: 層活性抽出 → SAKURA_ARTIFACT_DIR へ成果物を置く
set -euo pipefail
cd /app

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
# 空白区切りで none / reading / norms を並べる
PROMPT_MODES="${PROMPT_MODES:-reading norms}"

EXTRA=()
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi
if [[ "${TRUST_REMOTE_CODE:-}" == "1" ]]; then
  EXTRA+=(--trust-remote-code)
fi

for mode in ${PROMPT_MODES}; do
  echo "=== prompt-mode: ${mode} ==="
  python scripts/extract_revision_activations.py \
    --pairs data/revision_pairs.jsonl \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --prompt-mode "${mode}" \
    "${EXTRA[@]}"
done

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"
cp -a outputs/steering/. "${art}/"
echo "artifacts under ${art}:"
find "${art}" -type f -ls
