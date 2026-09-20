#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
E2="$ROOT/outputs/ead/data/e2.jsonl"
LOG="$ROOT/outputs/ead/work/e-pipeline.log"
EXPECTED=60
mkdir -p "$(dirname "$LOG")"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

n_e2="$(wc -l < "$E2" | tr -d ' ')"
log "E2 lines=$n_e2 (expected $EXPECTED)"
if [[ "$n_e2" -ne "$EXPECTED" ]]; then
  log "E2 incomplete; abort pipeline"
  exit 1
fi

log "starting E3"
make ead-generate-e3 >>"$LOG" 2>&1

n_e3="$(wc -l < "$ROOT/outputs/ead/data/e3.jsonl" | tr -d ' ')"
log "E3 lines=$n_e3 (expected $EXPECTED)"
if [[ "$n_e3" -ne "$EXPECTED" ]]; then
  log "E3 incomplete; abort pipeline"
  exit 1
fi

log "merge E1/E2/E3"
make ead-build-e >>"$LOG" 2>&1

log "validate E"
make ead-validate-e >>"$LOG" 2>&1
log "E pipeline complete"
