#!/bin/bash
# DOK: 段階 1 から 7 を書いた順に学び、検証 50 件を採点する
set -euo pipefail
cd /app

EMBED_MODEL="${EMBED_MODEL:-cl-nagoya/ruri-v3-30m}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-40}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-10}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
STAGES="${STAGES:-1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-0,1,2}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_VALID="${MAX_VALID:-0}"
GATE_MODEL="${GATE_MODEL:-outputs/pref-bt-keep}"
REPORT_DIR="${REPORT_DIR:-outputs/pref-multigranular-report}"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"

echo "=== pref-multigranular stages=${STAGES} seeds=${SEEDS} device=${DEVICE} ==="
python scripts/run_pref_multigranular_stages.py \
  --stages "${STAGES}" \
  --model "${EMBED_MODEL}" \
  --text-prefix "文章: " \
  --max-seq-length 256 \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --phase1-epochs "${PHASE1_EPOCHS}" \
  --phase2-epochs "${PHASE2_EPOCHS}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --seeds "${SEEDS}" \
  --max-train "${MAX_TRAIN}" \
  --max-valid "${MAX_VALID}" \
  --gate-model "${GATE_MODEL}" \
  --report-dir "${REPORT_DIR}"

mkdir -p "${art}/pref-multigranular-report"
cp -a "${REPORT_DIR}/." "${art}/pref-multigranular-report/"
for d in \
  outputs/pref-sentseq-keep-pairsplit \
  outputs/pref-pair-draft \
  outputs/pref-pair-humantop \
  outputs/pref-pair-humantop-hunk-then-section \
  outputs/pref-pair-humantop-hunk-replay \
  outputs/pref-joint-humantop-hunk-replay \
  outputs/pref-pair-draft-seed* \
  outputs/pref-pair-humantop-seed* \
  outputs/pref-pair-humantop-hunk-then-section-seed* \
  outputs/pref-pair-humantop-hunk-replay-seed* \
  outputs/pref-joint-humantop-hunk-replay-seed*
do
  if [ -d "${d}" ]; then
    mkdir -p "${art}/$(basename "${d}")"
    cp -a "${d}/." "${art}/$(basename "${d}")/"
  fi
done

echo "artifacts under ${art}:"
find "${art}" -type f -ls
