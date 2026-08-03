#!/usr/bin/env bash
# 旧: tip main..edit/* を直接掘っていた（推敲対として不正になりうる）。
# 現行: resolve_pre_merge_pair 経由の fork..edit のみ。batch_mine_hunks_premerge.sh と同じ。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "batch_import_repos.sh は廃止相当です。" >&2
echo "tip main との直接比較は禁止。次を使ってください:" >&2
echo "  MANIFEST=... OUT=... bash scripts/batch_mine_hunks_premerge.sh" >&2
echo "または scripts/verify_edit_revision_pairs.py のあと、manifest 付き premerge 採掘。" >&2
exit 1
