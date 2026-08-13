#!/bin/bash
# DOK タスク用: 文列 Transformer 報酬モデル（pref-sentseq）
#   MODE=xproject : LOPO 評価（fold ごとに学習し直す）→ report を成果物へ
#   MODE=train    : 全 train/valid で1本学習 → モデルと metrics を成果物へ
set -euo pipefail
cd /app

MODE="${MODE:-xproject}"
EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
MAX_SENTS="${MAX_SENTS:-128}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
# MODE=train 用: 学習/検証データの差し替え（例: アンカー付き分割）
TRAIN_FILE="${TRAIN_FILE:-data/pref_split/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/pref_split/valid.jsonl}"
# MODE=train 用: 学習済み model.pt からの追学習（二段階学習の第2段）
INIT_FROM="${INIT_FROM:-}"
# スモーク用: カンマ区切りの project_id。空なら全 fold
ONLY_PROJECTS="${ONLY_PROJECTS:-}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"

COMMON=(
  --model "${EMBED_MODEL}"
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --max-sents "${MAX_SENTS}"
  --d-model "${D_MODEL}"
  --num-layers "${NUM_LAYERS}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --device cuda
)

case "${MODE}" in
  xproject)
    EXTRA=()
    if [[ -n "${ONLY_PROJECTS}" ]]; then
      EXTRA+=(--only-projects "${ONLY_PROJECTS}")
    fi
    python scripts/eval_pref_sentseq_xproject.py \
      --input data/pref_dataset.jsonl \
      --report "${art}/eval_sentseq_xproject.json" \
      "${COMMON[@]}" "${EXTRA[@]}"
    ;;
  train)
    EXTRA=()
    if [[ -n "${INIT_FROM}" ]]; then
      EXTRA+=(--init-from "${INIT_FROM}")
    fi
    python scripts/train_pref_sentseq.py \
      --train-file "${TRAIN_FILE}" \
      --eval-file "${EVAL_FILE}" \
      --output-dir outputs/pref-sentseq \
      "${COMMON[@]}" "${EXTRA[@]}"
    cp -a outputs/pref-sentseq/. "${art}/pref-sentseq/"
    ;;
  *)
    echo "unknown MODE: ${MODE} (xproject|train)" >&2
    exit 1
    ;;
esac

echo "artifacts under ${art}:"
find "${art}" -type f -ls
