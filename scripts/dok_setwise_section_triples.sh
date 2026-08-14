#!/bin/bash
# DOK: 節三つ組みで setwise 文列型を学ぶ
set -euo pipefail
cd /app

EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
DEVICE="${DEVICE:-cuda}"
TRAIN_FILE="${TRAIN_FILE:-data/section_middle/pref_train.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/section_middle/pref_valid.jsonl}"
OUT="${OUT:-outputs/pref-setwise-section-triples}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
MAX_SENTS="${MAX_SENTS:-128}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-3e-4}"

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

echo "=== train pref-setwise from section triples device=${DEVICE} ==="
echo "train=${TRAIN_FILE} eval=${EVAL_FILE} out=${OUT}"
python scripts/train_pref_setwise.py \
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
  --text-prefix "文章: " \
  --device "${DEVICE}"

mkdir -p "${art}/pref-setwise-section-triples"
cp -a "${OUT}/." "${art}/pref-setwise-section-triples/"
printf '%s\n' "unit=section" "teacher=section_triple_v1" "kind=pref-setwise-section-triples" > "${art}/pref-setwise-section-triples/DATA_UNIT.txt"

echo "artifacts under ${art}:"
find "${art}" -type f -ls
