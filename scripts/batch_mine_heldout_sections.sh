#!/usr/bin/env bash
# 学習に使っていない held-out リポジトリから、マージ済み edit ブランチの節ペアを復元する。
# 出力: data/examples.section.heldout.jsonl（難試験 v2 の材料。学習には使わない）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

OUT="${OUT:-$ROOT/data/examples.section.heldout.jsonl}"

# 学習用 manifest（section_mining_manifest.json）に入っていないリポジトリのみ
REPOS=(
  /home/k16/work/Nmonthly/dev-with-type
  /home/k16/work/Nmonthly/lang-on-wasm
  /home/k16/work/Nmonthly/lean-by-example
  /home/k16/work/Nmonthly/ml-feature-store
  /home/k16/work/Nmonthly/nix
  /home/k16/work/Nmonthly/pfvm
  /home/k16/work/Nmonthly/picoruby
  /home/k16/work/Nmonthly/smtp-revisit
  /home/k16/work/Nmonthly/websocket-from-security
)

for repo in "${REPOS[@]}"; do
  if [[ ! -d "$repo/.git" ]]; then
    echo "skip (not a repo): $repo" >&2
    continue
  fi
  echo "== $(basename "$repo")"
  "$PYTHON" "$ROOT/scripts/mine_merged_section_pairs.py" \
    --repo "$repo" \
    --append "$OUT" || echo "failed: $repo" >&2
done

wc -l "$OUT"
"$PYTHON" "$ROOT/scripts/analyze_section_pairs.py" --input "$OUT"
