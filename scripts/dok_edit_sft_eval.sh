#!/bin/bash
# DOK: 検証ペアの推敲生成（base / adapter、貪欲 + 任意でサンプリング）
set -euo pipefail
cd /app

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ADAPTER="${ADAPTER:-/app/adapter}"
HELDOUT="${HELDOUT:-data/edit_sft_all/heldout.jsonl}"
# 0 = heldout 全件（貪欲）。サンプリング対象は IDS_FILE で絞る
LIMIT="${LIMIT:-0}"
# 空白区切り。本計画の既定は base adapter（規範前置は使わない）
EVAL_MODES="${EVAL_MODES:-base adapter}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
MAX_CHARS="${MAX_CHARS:-0}"
MIN_CHARS="${MIN_CHARS:-1}"
DEVICE="${DEVICE:-cuda}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
OUT_DIR="${OUT_DIR:-outputs/edit-sft-eval-v3}"
# サンプリング: 0 なら貪欲のみ。>0 なら IDS_FILE 対象に N 本
# base のサンプルはどの比較ペアにも使わないので、既定は adapter のみ
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SAMPLE_MODES="${SAMPLE_MODES:-adapter}"
IDS_FILE="${IDS_FILE:-}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-0}"

EXTRA=()
if [[ "${TRUST_REMOTE_CODE:-}" == "1" ]]; then
  EXTRA+=(--trust-remote-code)
fi
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  EXTRA+=(--load-in-4bit)
fi

mkdir -p "${OUT_DIR}"

# チェックポイント一括評価モード:
# イメージに /app/checkpoints（checkpoint-*/）が入っているとき、
# CKPT_EVAL_ONLY=1 なら各チェックポイントで判定対象を少数貪欲生成して
# 機械計測だけ出して終わる（使用エポックの選定用）。
if [[ "${CKPT_EVAL_ONLY:-0}" == "1" ]]; then
  CKPT_EVAL_LIMIT="${CKPT_EVAL_LIMIT:-30}"
  CKPT_IDS="${IDS_FILE:-data/blind_eval/items.jsonl}"
  mkdir -p "${OUT_DIR}/ckpt_eval"
  gen_files=()
  for ckpt in /app/checkpoints/checkpoint-*; do
    [[ -d "${ckpt}" ]] || continue
    name="$(basename "${ckpt}")"
    out_file="${OUT_DIR}/ckpt_eval/${name}_greedy.jsonl"
    echo "=== ckpt_eval ${name} ==="
    python scripts/generate_edit_sft.py \
      --heldout "${HELDOUT}" \
      --base-model "${MODEL}" \
      --adapter "${ckpt}" \
      --mode adapter \
      --device "${DEVICE}" \
      --limit "${CKPT_EVAL_LIMIT}" \
      --ids-file "${CKPT_IDS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-input-tokens "${MAX_INPUT_TOKENS}" \
      --out "${out_file}" \
      "${EXTRA[@]}"
    gen_files+=("${out_file}")
  done
  if [[ "${#gen_files[@]}" -eq 0 ]]; then
    echo "checkpoints が無い（ビルド時に CKPTS_SRC を指定したか）" >&2
    exit 1
  fi
  python scripts/generation_integrity.py "${gen_files[@]}" \
    --out "${OUT_DIR}/ckpt_eval/summary.json"
  art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
  mkdir -p "${art}"
  cp -a "${OUT_DIR}/." "${art}/"
  echo "artifacts under ${art}:"
  find "${art}" -type f -ls
  exit 0
fi

run_one() {
  local mode="$1"
  local num_samples="$2"
  local out_name="$3"
  local extra_ids=()
  if [[ -n "${IDS_FILE}" ]]; then
    extra_ids+=(--ids-file "${IDS_FILE}")
  fi
  echo "=== generate mode=${mode} samples=${num_samples} out=${out_name} ==="
  python scripts/generate_edit_sft.py \
    --heldout "${HELDOUT}" \
    --base-model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --mode "${mode}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --min-chars "${MIN_CHARS}" \
    --max-chars "${MAX_CHARS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --num-samples "${num_samples}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    --out "${OUT_DIR}/${out_name}" \
    "${extra_ids[@]}" \
    "${EXTRA[@]}"
}

# 貪欲: heldout 全件（IDS_FILE 無し）。LIMIT で絞れる
IDS_SAVE="${IDS_FILE}"
IDS_FILE=""
for mode in ${EVAL_MODES}; do
  run_one "${mode}" 0 "${mode}_greedy.jsonl"
done
IDS_FILE="${IDS_SAVE}"

# サンプリング: IDS_FILE があるときだけ
if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
  if [[ -z "${IDS_FILE}" ]]; then
    echo "NUM_SAMPLES>0 には IDS_FILE が必要" >&2
    exit 1
  fi
  LIMIT=0
  for mode in ${SAMPLE_MODES}; do
    run_one "${mode}" "${NUM_SAMPLES}" "${mode}_samples.jsonl"
  done
fi

# 機械ゲート: 人手判定に進む前に、比較実験として成立する生成かを数値で確認する
python scripts/generation_integrity.py "${OUT_DIR}"/*.jsonl \
  --out "${OUT_DIR}/integrity_summary.json" || true

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"
cp -a "${OUT_DIR}/." "${art}/"
echo "artifacts under ${art}:"
find "${art}" -type f -ls
