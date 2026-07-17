#!/usr/bin/env bash
# 使い方:
#   export REGISTRY=あなたのレジストリ名.sakuracr.jp
#   ./scripts/build_push_pref_ce_image.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

test -s data/pref_dataset.jsonl || {
  echo "data/pref_dataset.jsonl が無い。先に make train（データ生成部）" >&2
  exit 1
}
test -s data/pref_split/train.jsonl || {
  echo "data/pref_split/train.jsonl が無い。先に make train（データ生成部）" >&2
  exit 1
}
test -n "${REGISTRY:-}" || {
  echo "REGISTRY=xxxx.sakuracr.jp を export してから実行" >&2
  exit 1
}

TAG="${TAG:-pref-ce:latest}"
IMAGE="${REGISTRY}/${TAG}"

docker buildx build --platform linux/amd64 \
  -f Dockerfile.pref-ce \
  -t "${IMAGE}" \
  --push \
  .

echo "pushed: ${IMAGE}"
echo "DOK: イメージ=${IMAGE}"
echo "  スモーク: MODE=xproject ONLY_PROJECTS=ir-system"
echo "  本番:     MODE=xproject（全 fold LOPO）→ 勝ったら MODE=train"
