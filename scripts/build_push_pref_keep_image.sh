#!/usr/bin/env bash
# keep 分割の評価器学習用イメージを build / push する。
#   BT 用: pref_keep_split_hunk
#   文列用: pref_keep_split_section
#   REGISTRY 未設定: pref-keep:local
#   REGISTRY 設定済み: ${REGISTRY}/pref-keep:latest を push
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for f in \
  data/pref_keep_split_hunk/train.jsonl \
  data/pref_keep_split_hunk/valid.jsonl \
  data/pref_keep_split_section/train.jsonl \
  data/pref_keep_split_section/valid.jsonl
do
  test -s "$f" || {
    echo "$f が無い。先に make pref-keep-data" >&2
    exit 1
  }
done

TAG="${TAG:-pref-keep:latest}"
if [[ -n "${REGISTRY:-}" ]]; then
  IMAGE="${REGISTRY}/${TAG}"
  echo "building and pushing ${IMAGE}"
  docker buildx build --platform linux/amd64 \
    -f Dockerfile.pref-keep \
    -t "${IMAGE}" \
    --push \
    .
  echo "pushed: ${IMAGE}"
else
  IMAGE="pref-keep:local"
  echo "building local image ${IMAGE} (set REGISTRY=... to push)"
  docker build --platform linux/amd64 \
    -f Dockerfile.pref-keep \
    -t "${IMAGE}" \
    .
  echo "built: ${IMAGE}"
fi

echo "DOK: イメージ=${IMAGE}"
echo "  両学習: MODE=both（BT=hunk / sentseq=section）"
echo "  BT のみ: MODE=bt"
echo "  文列のみ: MODE=sentseq"
echo "  スモーク例: MODE=both SENTSEQ_EPOCHS=2 BT_EPOCHS=5"
