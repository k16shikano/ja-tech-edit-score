#!/bin/bash
# DOK: A1 段落内ペアと B 節三つ組みで GPM 評価器を学ぶ
set -euo pipefail
cd /app

EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
DEVICE="${DEVICE:-cuda}"
TRAIN_HUNK_FILE="${TRAIN_HUNK_FILE:-data/pref_keep_split_hunk/train.jsonl}"
TRAIN_TRIPLE_FILE="${TRAIN_TRIPLE_FILE:-data/section_middle/pref_train.jsonl}"
VALID_HUNK_FILE="${VALID_HUNK_FILE:-data/pref_keep_split_hunk/valid.jsonl}"
VALID_TRIPLE_FILE="${VALID_TRIPLE_FILE:-data/section_middle/pref_valid.jsonl}"
OUT="${OUT:-outputs/pref-gpm-a1b}"
HEAD_DIM="${HEAD_DIM:-4}"
HEAD_HIDDEN="${HEAD_HIDDEN:-0}"
TRUNCATE_DIM="${TRUNCATE_DIM:-0}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
MAX_SENTS="${MAX_SENTS:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
SEED="${SEED:-0}"
DROP_GEN_OVER_DRAFT="${DROP_GEN_OVER_DRAFT:-0}"
NORMALIZE_EMBEDDING="${NORMALIZE_EMBEDDING:-0}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"

for f in "${TRAIN_HUNK_FILE}" "${TRAIN_TRIPLE_FILE}" "${VALID_HUNK_FILE}" "${VALID_TRIPLE_FILE}"; do
  test -s "${f}" || {
    echo "missing ${f}" >&2
    exit 1
  }
done

echo "=== train pref-gpm from A1 hunk pairs and B section triples device=${DEVICE} ==="
echo "train_hunk=${TRAIN_HUNK_FILE} train_triple=${TRAIN_TRIPLE_FILE}"
echo "valid_hunk=${VALID_HUNK_FILE} valid_triple=${VALID_TRIPLE_FILE} out=${OUT}"
extra=()
if [[ "${DROP_GEN_OVER_DRAFT}" == "1" ]]; then
  extra+=(--drop-gen-over-draft)
fi
if [[ "${NORMALIZE_EMBEDDING}" == "1" ]]; then
  extra+=(--normalize-embedding)
fi
python scripts/train_pref_gpm.py \
  --model "${EMBED_MODEL}" \
  --train-hunk-file "${TRAIN_HUNK_FILE}" \
  --train-triple-file "${TRAIN_TRIPLE_FILE}" \
  --valid-hunk-file "${VALID_HUNK_FILE}" \
  --valid-triple-file "${VALID_TRIPLE_FILE}" \
  --output-dir "${OUT}" \
  --head-dim "${HEAD_DIM}" \
  --head-hidden "${HEAD_HIDDEN}" \
  --truncate-dim "${TRUNCATE_DIM}" \
  --text-prefix "文章: " \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE}" \
  --d-model "${D_MODEL}" \
  --num-layers "${NUM_LAYERS}" \
  --max-sents "${MAX_SENTS}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${extra[@]}"

mkdir -p "${art}/pref-gpm-a1b"
cp -a "${OUT}/." "${art}/pref-gpm-a1b/"
printf '%s\n' \
  "unit=a1+b" \
  "teacher_a1=hunk_pair" \
  "teacher_b=section_triple_v1" \
  "loss=gpm" \
  > "${art}/pref-gpm-a1b/DATA_UNIT.txt"

echo "artifacts under ${art}:"
find "${art}" -type f -ls
