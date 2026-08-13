#!/bin/bash
# DOK: 生成文人手選好で section sentseq を学習する（PLAN 工程 8c）
#   手元 CPU スモークは make generated-pref-sentseq-smoke
#   本学習は本スクリプト（cuda）
set -euo pipefail
cd /app

EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-data/generated_pref_experiment}"
ANCHOR_FILE="${ANCHOR_FILE:-data/pref_keep_split_section/train.jsonl}"
ANCHOR_BATCH_FRACTION="${ANCHOR_BATCH_FRACTION:-0.25}"
FOLDS="${FOLDS:-0 1 2 3 4}"
LENGTH_FEATURES="${LENGTH_FEATURES:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
MAX_SENTS="${MAX_SENTS:-128}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"

len_flag=()
len_suffix=""
if [ "${LENGTH_FEATURES}" = "1" ]; then
  len_flag=(--length-features)
  len_suffix="-lenon"
fi

for fold in ${FOLDS}; do
  train_file="${DATA_DIR}/folds_section/fold_${fold}/train.jsonl"
  valid_file="${DATA_DIR}/folds_section/fold_${fold}/valid.jsonl"
  out="outputs/generated-pref-sentseq-fold${fold}${len_suffix}"
  test -s "${train_file}" || {
    echo "missing ${train_file}" >&2
    exit 1
  }
  test -s "${valid_file}" || {
    echo "missing ${valid_file}" >&2
    exit 1
  }
  echo "=== generated-pref-sentseq fold=${fold} device=${DEVICE} length_features=${LENGTH_FEATURES} ==="
  python scripts/train_generated_pref_sentseq.py \
    --train-file "${train_file}" \
    --valid-file "${valid_file}" \
    --output-dir "${out}" \
    --anchor-file "${ANCHOR_FILE}" \
    --anchor-batch-fraction "${ANCHOR_BATCH_FRACTION}" \
    --model "${EMBED_MODEL}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --max-sents "${MAX_SENTS}" \
    --d-model "${D_MODEL}" \
    --num-layers "${NUM_LAYERS}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --text-prefix "文章: " \
    --device "${DEVICE}" \
    "${len_flag[@]}"
  dest="${art}/generated-pref-sentseq-fold${fold}${len_suffix}"
  mkdir -p "${dest}"
  cp -a "${out}/." "${dest}/"
done

echo "artifacts under ${art}:"
find "${art}" -type f -ls
