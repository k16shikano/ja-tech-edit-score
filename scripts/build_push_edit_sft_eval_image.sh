#!/usr/bin/env bash
# 使い方:
#   export REGISTRY=（コンテナレジストリ名）.sakuracr.jp
#   # 省略時は取り直し adapter
#   ADAPTER_SRC=outputs/Qwen__Qwen3-8B-re/adapter ./scripts/build_push_edit_sft_eval_image.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ADAPTER_SRC="${ADAPTER_SRC:-outputs/Qwen__Qwen3-8B-pairsplit-v2/adapter}"
# 任意: CKPTS_SRC=<checkpoints ディレクトリ> でチェックポイント群を同梱
# （DOK 側 CKPT_EVAL_ONLY=1 のエポック選定に使う）
CKPTS_SRC="${CKPTS_SRC:-}"
STAGE="outputs/edit-sft/_dok_adapter"
STAGE_CKPTS="outputs/edit-sft/_dok_checkpoints"

test -s data/edit_sft_all/heldout.jsonl || {
  echo "data/edit_sft_all/heldout.jsonl が無い。先に make pairsplit-data" >&2
  exit 1
}
test -s data/blind_eval/items.jsonl || {
  echo "data/blind_eval/items.jsonl が無い。先に make select-blind-items" >&2
  exit 1
}
test -d "${ADAPTER_SRC}" || {
  echo "ADAPTER_SRC=${ADAPTER_SRC} が無い" >&2
  exit 1
}
test -s "${ADAPTER_SRC}/adapter_config.json" || {
  echo "${ADAPTER_SRC}/adapter_config.json が無い" >&2
  exit 1
}
test -s data/tech-writing-norms.md || {
  echo "data/tech-writing-norms.md が無い" >&2
  exit 1
}
test -n "${REGISTRY:-}" || {
  echo "REGISTRY=（コンテナレジストリ名）.sakuracr.jp を export してから実行" >&2
  exit 1
}

TAG="${TAG:-edit-sft-eval:latest}"
IMAGE="${REGISTRY}/${TAG}"

rm -rf "${STAGE}" "${STAGE_CKPTS}"
mkdir -p "${STAGE}" "${STAGE_CKPTS}"
cp -a "${ADAPTER_SRC}/." "${STAGE}/"
echo "staged adapter: ${ADAPTER_SRC} -> ${STAGE}"
if [[ -n "${CKPTS_SRC}" ]]; then
  test -d "${CKPTS_SRC}" || { echo "CKPTS_SRC=${CKPTS_SRC} が無い" >&2; exit 1; }
  cp -a "${CKPTS_SRC}/." "${STAGE_CKPTS}/"
  echo "staged checkpoints: ${CKPTS_SRC} -> ${STAGE_CKPTS}"
fi

# 既定の .dockerignore は Fly.io 用で data/scripts を除外するため、評価用に差し替える
IGNORE_BACKUP=""
cleanup() {
  rm -rf "${STAGE}" "${STAGE_CKPTS}"
  if [[ -n "${IGNORE_BACKUP}" && -f "${IGNORE_BACKUP}" ]]; then
    mv -f "${IGNORE_BACKUP}" .dockerignore
  fi
}
trap cleanup EXIT
if [[ -f .dockerignore ]]; then
  IGNORE_BACKUP="$(mktemp)"
  cp -a .dockerignore "${IGNORE_BACKUP}"
fi
cp -f .dockerignore.edit-sft-eval .dockerignore

docker buildx build --platform linux/amd64 \
  -f Dockerfile.edit-sft-eval \
  -t "${IMAGE}" \
  --push \
  .

cleanup
trap - EXIT

echo "pushed: ${IMAGE}"
echo "ADAPTER_SRC was: ${ADAPTER_SRC}"
echo "DOK: EVAL_MODES='base adapter' LIMIT=0 NUM_SAMPLES=8 IDS_FILE=items.jsonl（イメージ既定）"
echo "スモーク: LIMIT=8 NUM_SAMPLES=0"
