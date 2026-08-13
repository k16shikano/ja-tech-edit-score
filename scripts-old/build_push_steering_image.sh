#!/usr/bin/env bash
# 使い方:
#   export REGISTRY=あなたのレジストリ名.sakuracr.jp
#   ./scripts/build_push_steering_image.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

test -s data/revision_pairs.jsonl || {
  echo "data/revision_pairs.jsonl が無い。先に make steering-pairs" >&2
  exit 1
}
test -n "${REGISTRY:-}" || {
  echo "REGISTRY=xxxx.sakuracr.jp を export してから実行" >&2
  exit 1
}

TAG="${TAG:-steering-phase-a:latest}"
IMAGE="${REGISTRY}/${TAG}"

docker buildx build --platform linux/amd64 \
  -f Dockerfile.steering \
  -t "${IMAGE}" \
  --push \
  .

echo "pushed: ${IMAGE}"
echo "DOK タスク: イメージ=${IMAGE} / レジストリ認証を登録 / GPU=V100 以上"
echo "スモークなら環境変数 LIMIT=64"
