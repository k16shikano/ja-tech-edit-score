#!/usr/bin/env bash
# 使い方:
#   export REGISTRY=（コンテナレジストリ名）.sakuracr.jp
#   ./scripts/build_push_edit_sft_image.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

test -s data/edit_sft_all/train.jsonl || {
  echo "data/edit_sft_all/train.jsonl が無い。先に make edit-sft-export-keeps" >&2
  exit 1
}
if ! grep -q 'review_status' data/edit_sft_all/stats.json data/edit_sft_all/train.jsonl 2>/dev/null; then
  echo "WARNING: train.jsonl に review_status が見えない。export-keeps を確認すること" >&2
fi
test -n "${REGISTRY:-}" || {
  echo "REGISTRY=（コンテナレジストリ名）.sakuracr.jp を export してから実行" >&2
  exit 1
}

TAG="${TAG:-edit-sft:latest}"
IMAGE="${REGISTRY}/${TAG}"

IGNORE_BACKUP=""
cleanup_ignore() {
  if [[ -n "${IGNORE_BACKUP}" && -f "${IGNORE_BACKUP}" ]]; then
    mv -f "${IGNORE_BACKUP}" .dockerignore
  fi
}
trap cleanup_ignore EXIT
if [[ -f .dockerignore ]]; then
  IGNORE_BACKUP="$(mktemp)"
  cp -a .dockerignore "${IGNORE_BACKUP}"
fi
cp -f .dockerignore.edit-sft .dockerignore

docker buildx build --platform linux/amd64 \
  -f Dockerfile.edit-sft \
  -t "${IMAGE}" \
  --push \
  .

cleanup_ignore
trap - EXIT

echo "pushed: ${IMAGE}"
echo "DOK タスク: イメージ=${IMAGE} / レジストリ認証を登録 / GPU=V100 以上（不足なら H100）"
echo "スモークなら環境変数 LIMIT=64 EPOCHS=1"
