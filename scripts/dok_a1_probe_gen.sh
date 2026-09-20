#!/bin/bash
# DOK: A1 学習側の全件について、三群を温度付き 1 本ずつ出す。
# base / base_norms（japanese-tech-writing）/ adapter（規範なし）
# スモークは LIMIT>0（同じ件数を三群とも同じ下書きから出す）。
set -euo pipefail
cd /app

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ADAPTER="${ADAPTER:-/app/adapter}"
HELDOUT="${HELDOUT:-data/edit_sft_hunk_nopara/train.jsonl}"
IDS_FILE="${IDS_FILE:-}"
NORMS_FILE="${NORMS_FILE:-data/a1_probe/japanese-tech-writing.md}"
OUT_DIR="${OUT_DIR:-outputs/a1-probe}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
DEVICE="${DEVICE:-cuda}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-0}"

test -s "${HELDOUT}" || { echo "missing ${HELDOUT}" >&2; exit 1; }

EXTRA=()
if [[ "${TRUST_REMOTE_CODE:-}" == "1" ]]; then
  EXTRA+=(--trust-remote-code)
fi
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  EXTRA+=(--load-in-4bit)
fi
if [[ -n "${IDS_FILE}" ]]; then
  test -s "${IDS_FILE}" || { echo "missing ${IDS_FILE}" >&2; exit 1; }
  EXTRA+=(--ids-file "${IDS_FILE}")
fi

mkdir -p "${OUT_DIR}"

run_one() {
  local mode="$1"
  local extra_norms=()
  if [[ "${mode}" == "base_norms" ]]; then
    test -s "${NORMS_FILE}" || { echo "missing ${NORMS_FILE}" >&2; exit 1; }
    extra_norms+=(--norms-file "${NORMS_FILE}")
  fi
  echo "=== generate mode=${mode} samples=1 limit=${LIMIT} ==="
  python scripts/generate_edit_sft.py \
    --heldout "${HELDOUT}" \
    --base-model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --mode "${mode}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --num-samples 1 \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --out "${OUT_DIR}/${mode}_samples.jsonl" \
    "${extra_norms[@]}" \
    "${EXTRA[@]}"
}

run_one base
run_one base_norms
run_one adapter

python scripts/generation_integrity.py \
  "${OUT_DIR}/base_samples.jsonl" \
  "${OUT_DIR}/base_norms_samples.jsonl" \
  "${OUT_DIR}/adapter_samples.jsonl" \
  --out "${OUT_DIR}/integrity_summary.json"

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"
cp -a "${OUT_DIR}/." "${art}/"
echo "artifacts under ${art}:"
find "${art}" -type f -ls
