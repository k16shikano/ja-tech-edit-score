#!/bin/bash
# DOK タスク用: cross-encoder 報酬モデル（段階2b）
#   MODE=xproject : LOPO 評価（fold ごとに学習し直す）→ report を成果物へ
#   MODE=train    : 全 train/valid で1本学習 → モデルと metrics を成果物へ
set -euo pipefail
cd /app

MODE="${MODE:-xproject}"
BASE_MODEL="${BASE_MODEL:-sbintuitions/modernbert-ja-130m}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-3e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PRECISION="${PRECISION:-auto}"
# スモーク用: カンマ区切りの project_id。空なら全 fold
ONLY_PROJECTS="${ONLY_PROJECTS:-}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"

COMMON=(
  --base-model "${BASE_MODEL}"
  --max-length "${MAX_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --grad-accum "${GRAD_ACCUM}"
  --precision "${PRECISION}"
)
if [[ "${GRADIENT_CHECKPOINTING:-}" == "1" ]]; then
  COMMON+=(--gradient-checkpointing)
fi

case "${MODE}" in
  xproject)
    EXTRA=()
    if [[ -n "${ONLY_PROJECTS}" ]]; then
      EXTRA+=(--only-projects "${ONLY_PROJECTS}")
    fi
    python scripts/eval_pref_ce_xproject.py \
      --input data/pref_dataset.jsonl \
      --report "${art}/eval_ce_xproject.json" \
      "${COMMON[@]}" "${EXTRA[@]}"
    ;;
  train)
    python scripts/train_pref_ce.py \
      --train-file data/pref_split/train.jsonl \
      --eval-file data/pref_split/valid.jsonl \
      --output-dir outputs/pref-ce \
      "${COMMON[@]}"
    cp -a outputs/pref-ce/. "${art}/pref-ce/"
    ;;
  *)
    echo "unknown MODE: ${MODE} (xproject|train)" >&2
    exit 1
    ;;
esac

echo "artifacts under ${art}:"
find "${art}" -type f -ls
