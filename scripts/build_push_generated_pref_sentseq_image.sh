#!/usr/bin/env bash
# 生成文人手選好 sentseq 学習用イメージを build / push する（PLAN 工程 8c / DOK）。
# 必ず REGISTRY へ push する（未設定なら失敗。ローカルだけの箱は DOK で使えない）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for f in \
  data/pref_keep_split_section/train.jsonl \
  data/generated_pref_experiment/folds_section/fold_0/train.jsonl \
  data/generated_pref_experiment/folds_section/fold_0/valid.jsonl \
  data/generated_pref_experiment/folds_section/fold_1/train.jsonl \
  data/generated_pref_experiment/folds_section/fold_1/valid.jsonl \
  data/generated_pref_experiment/folds_section/fold_2/train.jsonl \
  data/generated_pref_experiment/folds_section/fold_2/valid.jsonl \
  data/generated_pref_experiment/folds_section/fold_3/train.jsonl \
  data/generated_pref_experiment/folds_section/fold_3/valid.jsonl \
  data/generated_pref_experiment/folds_section/fold_4/train.jsonl \
  data/generated_pref_experiment/folds_section/fold_4/valid.jsonl
do
  test -s "$f" || {
    echo "$f が無い。先に make generated-pref-data" >&2
    exit 1
  }
done

test -n "${REGISTRY:-}" || {
  echo "REGISTRY 未設定。例: export REGISTRY=ja-tech-edit.sakuracr.jp" >&2
  echo "（未設定のままローカル build には落とさない。DOK では使えない箱になるため）" >&2
  exit 1
}

TAG="${TAG:-generated-pref-sentseq:latest}"
IMAGE="${REGISTRY}/${TAG}"
echo "building and pushing ${IMAGE}"
docker buildx build --platform linux/amd64 \
  -f Dockerfile.generated-pref-sentseq \
  -t "${IMAGE}" \
  --push \
  .
echo "pushed: ${IMAGE}"
echo "DOK: イメージ=${IMAGE}"
echo "  5-fold（length off）: 環境変数なし（既定）"
echo "  length on: LENGTH_FEATURES=1"
echo "  1 fold だけ: FOLDS=0"
echo "  スモーク: EPOCHS=2 FOLDS=0"
