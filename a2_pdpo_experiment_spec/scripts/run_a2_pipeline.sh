#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-../.venv/bin/python3}"
CFG="config/experiment.yaml"
LOG_DIR="$ROOT/outputs/.jobs"
mkdir -p "$LOG_DIR"

run() {
  echo "===== $(date -Is) $* ====="
  "$PY" "$@"
}

run scripts/generate_generic.py --config "$CFG" --split train
run scripts/generate_generic.py --config "$CFG" --split dev
run scripts/build_preferences.py --config "$CFG"
run scripts/diagnose_length_leakage.py --config "$CFG"
run scripts/precompute_reference_logps.py --config "$CFG"
run scripts/train_pdpo.py --config "$CFG"
run scripts/evaluate_preferences.py --config "$CFG"
run scripts/generate_blind_ab.py --config "$CFG"
echo "===== $(date -Is) pipeline done ====="
