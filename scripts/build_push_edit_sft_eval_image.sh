#!/usr/bin/env bash
# 使い方:
#   export REGISTRY=あなたのレジストリ名.sakuracr.jp
#   ./scripts/build_push_edit_sft_eval_image.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

test -s data/edit_sft/heldout.jsonl || {
  echo "data/edit_sft/heldout.jsonl が無い。先に make edit-sft-data" >&2
  exit 1
}
test -d outputs/edit-sft/Qwen__Qwen3-8B/adapter || {
  echo "outputs/edit-sft/Qwen__Qwen3-8B/adapter が無い" >&2
  exit 1
}
test -s data/tech-writing-norms.md || {
  echo "data/tech-writing-norms.md が無い" >&2
  exit 1
}
test -n "${REGISTRY:-}" || {
  echo "REGISTRY=xxxx.sakuracr.jp を export してから実行" >&2
  exit 1
}

TAG="${TAG:-edit-sft-eval:latest}"
IMAGE="${REGISTRY}/${TAG}"

docker buildx build --platform linux/amd64 \
  -f Dockerfile.edit-sft-eval \
  -t "${IMAGE}" \
  --push \
  .

echo "pushed: ${IMAGE}"
echo "DOK: イメージ=${IMAGE} / LIMIT=64 でスモーク → 通ったら LIMIT=0"
echo "既定 EVAL_MODES='adapter base_norms'"
