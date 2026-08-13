#!/usr/bin/env bash
# 使い方:
#   ./scripts/build_pref_ce_image.sh           # ローカル: pref-ce:local
#   export REGISTRY=あなたのレジストリ名.sakuracr.jp
#   ./scripts/build_push_pref_ce_image.sh      # push 版（後方互換）
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/build_pref_ce_image.sh" "$@"
