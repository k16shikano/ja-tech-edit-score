#!/usr/bin/env bash
# 段階 1 から 7 の評価器学習イメージを build / push する
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for f in \
  data/section_middle/pref_train.jsonl \
  data/section_middle/pref_valid.jsonl \
  data/pref_keep_split_hunk/train.jsonl \
  data/pref_keep_split_section/train.jsonl \
  data/pref_keep_split_section/valid.jsonl \
  outputs/pref-bt-keep/model.joblib
do
  test -s "$f" || {
    echo "$f が無い" >&2
    exit 1
  }
done

test -n "${REGISTRY:-}" || {
  echo "REGISTRY 未設定。例: export REGISTRY=ja-tech-edit.sakuracr.jp" >&2
  exit 1
}

TAG="${TAG:-pref-multigranular:latest}"
IMAGE="${REGISTRY}/${TAG}"
echo "building and pushing ${IMAGE}"
docker buildx build --platform linux/amd64 \
  -f Dockerfile.pref-multigranular \
  -t "${IMAGE}" \
  --push \
  .
echo "pushed: ${IMAGE}"
echo "DOK: イメージ=${IMAGE}"
echo "  本番: 環境変数なし（段階 1-7、SEEDS=0,1,2、40 epoch）"
echo "  スモーク: EPOCHS=1 PHASE1_EPOCHS=1 PHASE2_EPOCHS=1 STAGES=2 SEEDS=0 MAX_TRAIN=8 MAX_VALID=8"
