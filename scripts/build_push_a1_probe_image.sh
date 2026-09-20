#!/usr/bin/env bash
# A1 三群生成イメージを非公開レジストリへ push する。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ADAPTER_SRC="${ADAPTER_SRC:-outputs/Qwen__Qwen3-8B-pairsplit-v2/adapter}"
STAGE="outputs/edit-sft/_dok_adapter"

test -s data/edit_sft_hunk_nopara/train.jsonl || {
  echo "data/edit_sft_hunk_nopara/train.jsonl が無い。先に make pairsplit-data" >&2
  exit 1
}
test -s data/a1_probe/japanese-tech-writing.md || {
  echo "data/a1_probe/japanese-tech-writing.md が無い。先に make a1-probe-items" >&2
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
test -n "${REGISTRY:-}" || {
  echo "REGISTRY=（コンテナレジストリ名）.sakuracr.jp を export してから実行" >&2
  exit 1
}

TAG="${TAG:-a1-probe-gen:latest}"
IMAGE="${REGISTRY}/${TAG}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}"
cp -a "${ADAPTER_SRC}/." "${STAGE}/"

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
cp -f .dockerignore.a1-probe .dockerignore

docker buildx build --platform linux/amd64 \
  -f Dockerfile.a1-probe \
  -t "${IMAGE}" \
  --push \
  .

cleanup
trap - EXIT

echo "pushed: ${IMAGE}"
echo "DOK: コマンド空。スモークは LIMIT=2 OUT_DIR=outputs/a1-probe-smoke。本番は LIMIT 未指定。"
