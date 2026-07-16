#!/bin/bash
# DOK タスク用: 編集モデル QLoRA SFT → SAKURA_ARTIFACT_DIR へ成果物を置く
set -euo pipefail
cd /app

MODEL="${MODEL:-Qwen/Qwen3-8B}"
LIMIT="${LIMIT:-0}"
EPOCHS="${EPOCHS:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"

EXTRA=()
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi
if [[ "${TRUST_REMOTE_CODE:-}" == "1" ]]; then
  EXTRA+=(--trust-remote-code)
fi

python scripts/train_edit_sft.py \
  --train data/edit_sft/train.jsonl \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --learning-rate "${LEARNING_RATE}" \
  "${EXTRA[@]}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"
# チェックポイント中間は重いので最終 adapter と meta だけ
cp -a outputs/edit-sft/. "${art}/"
# 巨大な checkpoints はアーティファクトから除く（容量制限対策）
find "${art}" -type d -name checkpoints -exec rm -rf {} + 2>/dev/null || true
echo "artifacts under ${art}:"
find "${art}" -type f -ls
