#!/bin/bash
# DOK: レビュー済み keep 分割で評価器を学習する
#   BT      = 空行なし hunk のみ → outputs/pref-bt-keep
#   sentseq = 節（空行あり）のみ → outputs/pref-sentseq-keep
#   MODE=bt | sentseq | both（既定）
set -euo pipefail
cd /app

MODE="${MODE:-both}"
EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
DEVICE="${DEVICE:-cuda}"

BT_TRAIN_FILE="${BT_TRAIN_FILE:-data/pref_keep_split_hunk/train.jsonl}"
BT_EVAL_FILE="${BT_EVAL_FILE:-data/pref_keep_split_hunk/valid.jsonl}"
SENTSEQ_TRAIN_FILE="${SENTSEQ_TRAIN_FILE:-data/pref_keep_split_section/train.jsonl}"
SENTSEQ_EVAL_FILE="${SENTSEQ_EVAL_FILE:-data/pref_keep_split_section/valid.jsonl}"

# BT
BT_MAX_SEQ_LENGTH="${BT_MAX_SEQ_LENGTH:-512}"
BT_BATCH_SIZE="${BT_BATCH_SIZE:-64}"
BT_EPOCHS="${BT_EPOCHS:-80}"
BT_LR="${BT_LR:-1e-2}"

# sentseq（採用構成に合わせる: 40ep / lr 3e-4）
SENTSEQ_MAX_SEQ_LENGTH="${SENTSEQ_MAX_SEQ_LENGTH:-256}"
SENTSEQ_MAX_SENTS="${SENTSEQ_MAX_SENTS:-128}"
SENTSEQ_D_MODEL="${SENTSEQ_D_MODEL:-256}"
SENTSEQ_NUM_LAYERS="${SENTSEQ_NUM_LAYERS:-2}"
SENTSEQ_BATCH_SIZE="${SENTSEQ_BATCH_SIZE:-64}"
SENTSEQ_EPOCHS="${SENTSEQ_EPOCHS:-40}"
SENTSEQ_LR="${SENTSEQ_LR:-3e-4}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"

run_bt() {
  echo "=== train pref-bt-keep from hunk split (device=${DEVICE}) ==="
  echo "train=${BT_TRAIN_FILE} eval=${BT_EVAL_FILE}"
  python scripts/train_pref_bt.py \
    --model "${EMBED_MODEL}" \
    --train-file "${BT_TRAIN_FILE}" \
    --eval-file "${BT_EVAL_FILE}" \
    --output-dir outputs/pref-bt-keep \
    --max-seq-length "${BT_MAX_SEQ_LENGTH}" \
    --batch-size "${BT_BATCH_SIZE}" \
    --epochs "${BT_EPOCHS}" \
    --lr "${BT_LR}" \
    --text-prefix "文章: " \
    --device "${DEVICE}"
  mkdir -p "${art}/pref-bt-keep"
  cp -a outputs/pref-bt-keep/. "${art}/pref-bt-keep/"
  printf '%s\n' "unit=hunk" > "${art}/pref-bt-keep/DATA_UNIT.txt"
}

run_sentseq() {
  echo "=== train pref-sentseq-keep from section split (device=${DEVICE}) ==="
  echo "train=${SENTSEQ_TRAIN_FILE} eval=${SENTSEQ_EVAL_FILE}"
  python scripts/train_pref_sentseq.py \
    --model "${EMBED_MODEL}" \
    --train-file "${SENTSEQ_TRAIN_FILE}" \
    --eval-file "${SENTSEQ_EVAL_FILE}" \
    --output-dir outputs/pref-sentseq-keep \
    --max-seq-length "${SENTSEQ_MAX_SEQ_LENGTH}" \
    --max-sents "${SENTSEQ_MAX_SENTS}" \
    --d-model "${SENTSEQ_D_MODEL}" \
    --num-layers "${SENTSEQ_NUM_LAYERS}" \
    --batch-size "${SENTSEQ_BATCH_SIZE}" \
    --epochs "${SENTSEQ_EPOCHS}" \
    --lr "${SENTSEQ_LR}" \
    --text-prefix "文章: " \
    --device "${DEVICE}"
  mkdir -p "${art}/pref-sentseq-keep"
  cp -a outputs/pref-sentseq-keep/. "${art}/pref-sentseq-keep/"
  printf '%s\n' "unit=section" > "${art}/pref-sentseq-keep/DATA_UNIT.txt"
}

case "${MODE}" in
  bt) run_bt ;;
  sentseq) run_sentseq ;;
  both)
    run_bt
    run_sentseq
    ;;
  *)
    echo "unknown MODE: ${MODE} (bt|sentseq|both)" >&2
    exit 1
    ;;
esac

echo "artifacts under ${art}:"
find "${art}" -type f -ls
