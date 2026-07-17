#!/usr/bin/env bash
# 節単位ペアから pref 学習データを作り、既存 hunk データとマージする。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

SECTION_RAW="${SECTION_RAW:-$ROOT/data/examples.section.raw.jsonl}"
SECTION_DB="${SECTION_DB:-$ROOT/data/examples.section.db}"
DPO_SECTION="${DPO_SECTION:-$ROOT/data/dpo_section.jsonl}"
DPO_SECTION_CURATED="${DPO_SECTION_CURATED:-$ROOT/data/dpo_section_curated.jsonl}"
PREF_SECTION="${PREF_SECTION:-$ROOT/data/pref_dataset_section.jsonl}"
PREF_HUNK="${PREF_HUNK:-$ROOT/data/pref_dataset.hunk.jsonl}"
PREF_MERGED="${PREF_MERGED:-$ROOT/data/pref_dataset.jsonl}"
PREF_SPLIT="${PREF_SPLIT:-$ROOT/data/pref_split}"
CURATE_REPORT="${CURATE_REPORT:-$ROOT/data/curate_section_report.json}"
MAX_CHARS="${MAX_CHARS:-4000}"

test -s "$SECTION_RAW" || {
  echo "missing $SECTION_RAW — run: make mine-sections" >&2
  exit 1
}

echo "[1/7] import section examples -> $SECTION_DB"
"$PYTHON" "$ROOT/scripts/import_examples.py" \
  --input "$SECTION_RAW" \
  --db "$SECTION_DB"

echo "[2/7] build DPO -> $DPO_SECTION"
"$PYTHON" "$ROOT/scripts/build_dpo_dataset.py" \
  --db "$SECTION_DB" \
  --out "$DPO_SECTION" \
  --accepted-only

echo "[3/7] curate DPO (max_chars=$MAX_CHARS) -> $DPO_SECTION_CURATED"
"$PYTHON" "$ROOT/scripts/curate_dpo_dataset.py" \
  --input "$DPO_SECTION" \
  --out "$DPO_SECTION_CURATED" \
  --max-chars "$MAX_CHARS" \
  --drop-citation-only \
  --report "$CURATE_REPORT"

echo "[4/7] build pref (swap) -> $PREF_SECTION"
"$PYTHON" "$ROOT/scripts/build_pref_dataset.py" \
  --input "$DPO_SECTION_CURATED" \
  --out "$PREF_SECTION" \
  --augment-swap

if [[ -f "$PREF_MERGED" && ! -f "$PREF_HUNK" ]]; then
  echo "[5/7] backup existing pref_dataset -> $PREF_HUNK"
  cp -a "$PREF_MERGED" "$PREF_HUNK"
elif [[ ! -f "$PREF_HUNK" ]]; then
  echo "[5/7] no existing hunk pref_dataset; section-only merge base"
  : > "$PREF_HUNK"
fi

echo "[6/7] merge hunk + section -> $PREF_MERGED"
"$PYTHON" - <<'PY' "$PREF_HUNK" "$PREF_SECTION" "$PREF_MERGED"
import json, sys
from pathlib import Path

hunk_path, section_path, out_path = map(Path, sys.argv[1:4])

def load(path: Path) -> list[dict]:
  if not path.is_file() or path.stat().st_size == 0:
    return []
  return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def key(rec: dict) -> tuple:
  return (
    rec.get('source_text', ''),
    rec.get('candidate_a', ''),
    rec.get('candidate_b', ''),
    int(rec.get('label', -1)),
  )

merged: list[dict] = []
seen: set[tuple] = set()
for path in (hunk_path, section_path):
  for rec in load(path):
    k = key(rec)
    if k in seen:
      continue
    seen.add(k)
    merged.append(rec)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', encoding='utf-8') as f:
  for rec in merged:
    f.write(json.dumps(rec, ensure_ascii=False) + '\n')

section_n = sum(1 for r in merged if 'section_pair_mined' in (r.get('meta', {}).get('labels') or []))
print(f"merged rows: {len(merged)} (section-tagged: {section_n})")
PY

echo "[7/7] split -> $PREF_SPLIT"
"$PYTHON" "$ROOT/scripts/split_pref_dataset.py" \
  --input "$PREF_MERGED" \
  --out-dir "$PREF_SPLIT" \
  --group-by base_id

wc -l "$PREF_MERGED" "$PREF_SPLIT/train.jsonl" "$PREF_SPLIT/valid.jsonl"
"$PYTHON" "$ROOT/scripts/analyze_section_pairs.py" --input "$SECTION_RAW" --compare "$PREF_HUNK"
