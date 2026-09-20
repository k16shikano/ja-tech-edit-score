#!/usr/bin/env bash
# GPM fold0-2 → BT fold2 まで順に実行し、fold 別 valid A_micro を報告ファイルに書く。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

REPORT="$ROOT/outputs/.jobs/qwen-fold0-2-report.txt"

make --no-print-directory pref-d-gpm-qwen3-8b-fg GPM_HEAD_DIM=16 MAX_FOLD=2
make --no-print-directory pref-d-bt-qwen3-8b-fg MAX_FOLD=2

"$ROOT/.venv/bin/python3" - <<'PY' | tee "$REPORT"
import json
from pathlib import Path

root = Path(".")
lines = ["Qwen3-8B D ペア fold0-2 valid A_micro", ""]
for label, sub in [("GPM", "pref-d-gpm-qwen3-8b"), ("BT", "pref-d-bt-qwen3-8b")]:
    out = root / "outputs" / sub
    lines.append(label)
    for fold in range(3):
        summary = json.loads((out / f"fold{fold}" / "summary.json").read_text(encoding="utf-8"))
        lines.append(
            f"  fold{fold}  epoch={summary['best_epoch']}  A_micro={summary['best_valid_A_micro']:.4f}"
        )
    lines.append("")
print("\n".join(lines))
PY

echo "report: $REPORT"
