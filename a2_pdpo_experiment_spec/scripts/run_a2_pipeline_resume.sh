#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-../.venv/bin/python3}"
CFG="config/experiment.yaml"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run() {
  echo "===== $(date -Is) $* ====="
  "$PY" "$@"
}

REF_META="data/reference_logps/.max_length"
WANT_LEN="$(../.venv/bin/python3 - <<'PY'
import yaml
from pathlib import Path
print(yaml.safe_load(Path("config/experiment.yaml").read_text())["training"]["max_length"])
PY
)"
if [[ ! -f data/reference_logps/train.jsonl ]] || [[ "$(cat "$REF_META" 2>/dev/null || true)" != "$WANT_LEN" ]]; then
  run scripts/precompute_reference_logps.py --config "$CFG"
  echo "$WANT_LEN" > "$REF_META"
fi
run scripts/train_pdpo.py --config "$CFG" --items-per-step 1
run scripts/evaluate_preferences.py --config "$CFG"
run scripts/generate_blind_ab.py --config "$CFG"
echo "===== $(date -Is) pipeline resume done ====="
