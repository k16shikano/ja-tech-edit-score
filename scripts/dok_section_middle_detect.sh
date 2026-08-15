#!/bin/bash
# DOK: 節ペアの三つ組みで人間検出評価器を学ぶ
# 既存の outputs/pref-sentseq-keep* と outputs/pref-nce-section は触らない
set -euo pipefail
cd /app

EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
DEVICE="${DEVICE:-cuda}"
TRAIN_FILE="${TRAIN_FILE:-data/section_middle/pref_train.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/section_middle/pref_valid.jsonl}"
OUT="${OUT:-outputs/pref-detect-section}"
ART_NAME="${ART_NAME:-pref-detect-section}"
COMPOSER_OVER_DRAFT="${COMPOSER_OVER_DRAFT:-0}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
MAX_SENTS="${MAX_SENTS:-128}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-3e-4}"
SEED="${SEED:-0}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"

test -s "${TRAIN_FILE}" || {
  echo "missing ${TRAIN_FILE}" >&2
  exit 1
}
test -s "${EVAL_FILE}" || {
  echo "missing ${EVAL_FILE}" >&2
  exit 1
}

echo "=== train pref-detect from section triples device=${DEVICE} composer_over_draft=${COMPOSER_OVER_DRAFT} ==="
echo "train=${TRAIN_FILE} eval=${EVAL_FILE} out=${OUT}"
extra=()
if [[ "${COMPOSER_OVER_DRAFT}" == "1" ]]; then
  extra+=(--composer-over-draft)
fi
python scripts/train_pref_detect.py \
  --model "${EMBED_MODEL}" \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --output-dir "${OUT}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --max-sents "${MAX_SENTS}" \
  --d-model "${D_MODEL}" \
  --num-layers "${NUM_LAYERS}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --text-prefix "文章: " \
  --device "${DEVICE}" \
  "${extra[@]}"

mkdir -p "${art}/${ART_NAME}"
cp -a "${OUT}/." "${art}/${ART_NAME}/"
if [[ "${COMPOSER_OVER_DRAFT}" == "1" ]]; then
  loss_name=bce_human_one_plus_composer_over_draft
else
  loss_name=bce_human_one
fi
printf '%s\n' "unit=section" "teacher=section_triple_v1" "loss=${loss_name}" "composer_over_draft=${COMPOSER_OVER_DRAFT}" > "${art}/${ART_NAME}/DATA_UNIT.txt"

echo "artifacts under ${art}:"
find "${art}" -type f -ls
