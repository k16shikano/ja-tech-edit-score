#!/usr/bin/env bash
# keep 分割の評価器学習用イメージを build / push する。
#   BT 用: pref_keep_split_hunk
#   文列用: pref_keep_split_section
#   必ず REGISTRY へ push する（未設定なら失敗。ローカルだけの箱は DOK で使えない）
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
    echo "$f が無い。先に make pairsplit-data" >&2
    exit 1
  }
done

test -n "${REGISTRY:-}" || {
  echo "REGISTRY 未設定。例: export REGISTRY=ja-tech-edit.sakuracr.jp" >&2
  echo "（未設定のままローカル build には落とさない。DOK では使えない箱になるため）" >&2
  exit 1
}

TAG="${TAG:-pref-keep:latest}"
IMAGE="${REGISTRY}/${TAG}"
echo "building and pushing ${IMAGE}"
docker buildx build --platform linux/amd64 \
  -f Dockerfile.pref-keep \
  -t "${IMAGE}" \
  --push \
  .
echo "pushed: ${IMAGE}"
echo "DOK: イメージ=${IMAGE}"
echo "  両学習: MODE=both（BT=hunk / sentseq=section）"
echo "  BT のみ: MODE=bt"
echo "  文列のみ: MODE=sentseq"
echo "  スモーク例: MODE=both SENTSEQ_EPOCHS=2 BT_EPOCHS=5"
