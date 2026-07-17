#!/usr/bin/env bash
# 節単位ペアを manifest に基づいて再採掘する。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
GIT="${GIT:-/usr/bin/git}"
MINER="$ROOT/scripts/mine_section_pairs.py"
ANALYZER="$ROOT/scripts/analyze_section_pairs.py"
MANIFEST="${MANIFEST:-$ROOT/data/section_mining_manifest.json}"
OUT="${OUT:-$ROOT/data/examples.section.raw.jsonl}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

resolve_ref() {
  local repo="$1"
  local ref="$2"
  if "$GIT" -C "$repo" rev-parse --verify "$ref" >/dev/null 2>&1; then
    echo "$ref"
    return 0
  fi
  if "$GIT" -C "$repo" rev-parse --verify "origin/$ref" >/dev/null 2>&1; then
    echo "origin/$ref"
    return 0
  fi
  return 1
}

"$PYTHON" - <<'PY' "$MANIFEST" "$MINER" "$OUT" "$GIT"
import json
import subprocess
import sys
from pathlib import Path

manifest_path, miner, out_path, git = sys.argv[1:5]

def resolve_ref(repo: str, ref: str) -> str | None:
  for candidate in (ref, f"origin/{ref}"):
    proc = subprocess.run(
      [git, "-C", repo, "rev-parse", "--verify", candidate],
      capture_output=True,
      text=True,
    )
    if proc.returncode == 0:
      return candidate
  return None

manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
for entry in manifest:
  repo = entry.get("repo")
  pid = entry["project_id"]
  if not repo or not Path(repo).is_dir() or not (Path(repo) / ".git").exists():
    print(f"[skip] missing repo: {pid} {repo}")
    continue
  for pair in entry.get("branch_pairs", []):
    base = resolve_ref(repo, pair["base"])
    edit = resolve_ref(repo, pair["edit"])
    if not base or not edit:
      print(f"[skip] cannot resolve refs: {pid} {pair['base']} -> {pair['edit']}")
      continue
    print(f"[mine-section] {pid} base={base} edit={edit} repo={repo}")
    proc = subprocess.run(
      [
        sys.executable,
        miner,
        "--repo", repo,
        "--base", base,
        "--edit", edit,
        "--project-id", pid,
        "--append", out_path,
      ],
      text=True,
    )
    if proc.returncode != 0:
      print(f"[warn] failed: {pid} {pair['base']} -> {pair['edit']}")
PY

if [[ -f "$OUT" ]]; then
  wc -l "$OUT"
  "$PYTHON" "$ANALYZER" --input "$OUT" --compare "$ROOT/data/examples.raw.jsonl"
fi
