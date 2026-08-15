#!/usr/bin/env bash
# 節ペア三つ組みの InfoNCE 学習イメージを build / push する（PLAN 工程 8-nce / DOK）。
# 必ず REGISTRY へ push する（未設定なら失敗。ローカルだけの箱は DOK で使えない）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for f in \
  data/section_middle/pref_train.jsonl \
  data/section_middle/pref_valid.jsonl
do
  test -s "$f" || {
    echo "$f が無い。先に make section-middle-triples" >&2
    exit 1
  }
done

test -n "${REGISTRY:-}" || {
  echo "REGISTRY 未設定。例: export REGISTRY=ja-tech-edit.sakuracr.jp" >&2
  echo "（未設定のままローカル build には落とさない。DOK では使えない箱になるため）" >&2
  exit 1
}

TAG="${TAG:-pref-nce-section:latest}"
IMAGE="${REGISTRY}/${TAG}"
echo "building and pushing ${IMAGE}"
docker buildx build --platform linux/amd64 \
  -f Dockerfile.section-middle-nce \
  -t "${IMAGE}" \
  --push \
  .
echo "pushed: ${IMAGE}"
echo "DOK: イメージ=${IMAGE}"
echo "  本番: 環境変数なし（既定 40 epoch / lr 3e-4 / tau 0.07）"
echo "  スモーク: EPOCHS=2"
