#!/usr/bin/env bash
# 使い方:
#   export REGISTRY=（コンテナレジストリ名）.sakuracr.jp
#   # 省略時は取り直し adapter
#   ADAPTER_SRC=outputs/Qwen__Qwen3-8B-re/adapter ./scripts/build_push_edit_sft_eval_image.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ADAPTER_SRC="${ADAPTER_SRC:-outputs/Qwen__Qwen3-8B-revised-2000/adapter}"
STAGE="outputs/edit-sft/_dok_adapter"

test -s data/edit_sft_all/heldout.jsonl || {
  echo "data/edit_sft_all/heldout.jsonl が無い。先に make edit-sft-export-keeps" >&2
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

rm -rf "${STAGE}"
mkdir -p "${STAGE}"
cp -a "${ADAPTER_SRC}/." "${STAGE}/"
echo "staged adapter: ${ADAPTER_SRC} -> ${STAGE}"

# 既定の .dockerignore は Fly.io 用で data/scripts を除外するため、評価用に差し替える
IGNORE_BACKUP=""
cleanup() {
  rm -rf "${STAGE}"
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
echo "DOK バッチ: イメージ=${IMAGE} / LIMIT=64 でスモーク → LIMIT=0"
echo "DOK 対話: docs/DOK-EDIT-SFT-CHAT.md（SSH ON・エントリーポイント上書き）"
