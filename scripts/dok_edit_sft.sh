#!/bin/bash
# DOK タスク用: 編集モデル QLoRA SFT → SAKURA_ARTIFACT_DIR へ成果物を置く
set -euo pipefail
cd /app

MODEL="${MODEL:-Qwen/Qwen3-8B}"
LIMIT="${LIMIT:-0}"
# 1 回目の学習（2 エポック）は過小編集（無編集率 23%、sim_median 0.98＝ほぼ下書きコピー）。
# エポックを増やし、エポック毎チェックポイントを機械計測（無編集・構成保持・文中断）で比べて選ぶ。
EPOCHS="${EPOCHS:-6}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
OUT_DIR="${OUT_DIR:-outputs/Qwen__Qwen3-8B-pairsplit-v2}"

EXTRA=()
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi
if [[ "${TRUST_REMOTE_CODE:-}" == "1" ]]; then
  EXTRA+=(--trust-remote-code)
fi

python scripts/train_edit_sft.py \
  --train data/edit_sft_all/train.jsonl \
  --model "${MODEL}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --learning-rate "${LEARNING_RATE}" \
  "${EXTRA[@]}"

# --- チェックポイント別の少数生成と機械計測（エポック選定用） ---
# 判定 60 件のうち CKPT_EVAL_LIMIT 件を貪欲生成し、無編集率・保持率・文中断を比べる。
CKPT_EVAL="${CKPT_EVAL:-1}"
CKPT_EVAL_LIMIT="${CKPT_EVAL_LIMIT:-30}"
CKPT_EVAL_IDS="${CKPT_EVAL_IDS:-data/blind_eval/items.jsonl}"
if [[ "${CKPT_EVAL}" == "1" && -s data/edit_sft_all/heldout.jsonl ]]; then
  mkdir -p "${OUT_DIR}/ckpt_eval"
  gen_files=()
  for ckpt in "${OUT_DIR}/checkpoints"/checkpoint-*; do
    [[ -d "${ckpt}" ]] || continue
    name="$(basename "${ckpt}")"
    out_file="${OUT_DIR}/ckpt_eval/${name}_greedy.jsonl"
    echo "=== ckpt_eval ${name} ==="
    python scripts/generate_edit_sft.py \
      --heldout data/edit_sft_all/heldout.jsonl \
      --base-model "${MODEL}" \
      --adapter "${ckpt}" \
      --mode adapter \
      --device cuda \
      --limit "${CKPT_EVAL_LIMIT}" \
      --ids-file "${CKPT_EVAL_IDS}" \
      --max-new-tokens 4096 \
      --max-input-tokens 3072 \
      --load-in-4bit \
      --out "${out_file}" \
      "${EXTRA[@]}"
    gen_files+=("${out_file}")
  done
  if [[ "${#gen_files[@]}" -gt 0 ]]; then
    python scripts/generation_integrity.py "${gen_files[@]}" \
      --out "${OUT_DIR}/ckpt_eval/summary.json"
  fi
fi

art="${SAKURA_ARTIFACT_DIR:-/opt/artifact}"
mkdir -p "${art}"
cp -a "${OUT_DIR}/." "${art}/"
# チェックポイントは adapter 重みと学習ログだけ残す（optimizer 等は落とす）
find "${art}" -type f \( -name 'optimizer.pt' -o -name 'scheduler.pt' \
  -o -name 'rng_state*.pth' -o -name 'training_args.bin' \) -delete 2>/dev/null || true
echo "artifacts under ${art}:"
find "${art}" -type f -ls
