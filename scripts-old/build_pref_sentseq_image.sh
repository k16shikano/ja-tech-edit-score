#!/usr/bin/env bash
# pref-sentseq 再学習用 Docker イメージをビルドする。
#   REGISTRY 未設定: ローカルタグ pref-sentseq:local のみ
#   REGISTRY 設定済み: ${REGISTRY}/pref-sentseq:latest を build & push
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

test -s data/pref_dataset.jsonl || {
  echo "data/pref_dataset.jsonl が無い。先に make section-pref-data" >&2
  exit 1
}
test -s data/pref_split/train.jsonl || {
  echo "data/pref_split/train.jsonl が無い。先に make section-pref-data" >&2
  exit 1
}

TAG="${TAG:-pref-sentseq:latest}"
if [[ -n "${REGISTRY:-}" ]]; then
  IMAGE="${REGISTRY}/${TAG}"
  echo "building and pushing ${IMAGE}"
  docker buildx build --platform linux/amd64 \
    -f Dockerfile.pref-sentseq \
    -t "${IMAGE}" \
    --push \
    .
  echo "pushed: ${IMAGE}"
else
  IMAGE="pref-sentseq:local"
  echo "building local image ${IMAGE} (set REGISTRY=... to push)"
  docker build --platform linux/amd64 \
    -f Dockerfile.pref-sentseq \
    -t "${IMAGE}" \
    .
  echo "built: ${IMAGE}"
fi

echo "DOK 用: イメージ=${IMAGE:-pref-sentseq:local}"
echo "  スモーク: MODE=xproject ONLY_PROJECTS=ir-system EPOCHS=2"
echo "  本番:     MODE=train"
