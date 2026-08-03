#!/usr/bin/env bash
# 14 件以外の section 候補リポジトリから節ペアを採掘する。
# manifest: data/section_mining_manifest.extra.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
GIT="${GIT:-/usr/bin/git}"
MINER="$ROOT/scripts/mine_section_pairs.py"
MANIFEST="${MANIFEST:-$ROOT/data/section_mining_manifest.extra.json}"
OUT="${OUT:-$ROOT/data/examples.section.extra.raw.jsonl}"
EXCLUDE="${EXCLUDE:-$ROOT/data/section_extra_exclude.json}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

rm -f "$OUT"

"$PYTHON" - <<'PY' "$MANIFEST" "$MINER" "$OUT" "$GIT" "$EXCLUDE"
import json
import subprocess
import sys
from pathlib import Path

manifest_path, miner, out_path, git, exclude_path = sys.argv[1:6]
sys.path.insert(0, str(Path(miner).resolve().parent))
from git_pre_merge import detect_mainline, resolve_pre_merge_pair  # noqa: E402

exclude: set[str] = set()
if Path(exclude_path).is_file():
  ex = json.loads(Path(exclude_path).read_text(encoding="utf-8"))
  exclude = set((ex.get("exclude") or {}).keys())


def resolve_ref(repo: str, ref: str) -> str | None:
  if ref in ("main", "master"):
    detected = detect_mainline(repo)
    if detected:
      return detected
  for candidate in (f"origin/{ref}", ref):
    proc = subprocess.run(
      [git, "-C", repo, "rev-parse", "--verify", candidate],
      capture_output=True,
      text=True,
    )
    if proc.returncode == 0:
      return candidate
  return None


manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
n_ok = 0
n_skip = 0
for entry in manifest:
  repo = entry.get("repo")
  pid = entry["project_id"]
  if pid in exclude:
    print(f"[skip] excluded: {pid}")
    n_skip += 1
    continue
  if not repo or not Path(repo).is_dir() or not (Path(repo) / ".git").exists():
    print(f"[skip] missing repo: {pid} {repo}")
    n_skip += 1
    continue
  for pair in entry.get("branch_pairs", []):
    mainline = resolve_ref(repo, pair["base"])
    edit = resolve_ref(repo, pair["edit"])
    if not mainline or not edit:
      print(f"[skip] cannot resolve refs: {pid} {pair['base']} -> {pair['edit']}")
      n_skip += 1
      continue
    resolved = resolve_pre_merge_pair(Path(repo), mainline, edit)
    if not resolved:
      print(
        f"[skip] no pre-merge pair: {pid} {pair['edit']} "
        f"(reedit/summary / late unmerged after editorial merge / "
        f"main not older / already on mainline / empty diff)"
      )
      n_skip += 1
      continue
    base_sha, edit_sha, meta = resolved
    print(
      f"[mine-section-extra] {pid} method={meta['method']} "
      f"base={base_sha[:10]} edit={edit_sha[:10]} "
      f"mainline={mainline} branch={meta['edit_branch']}"
    )
    cmd = [
      sys.executable,
      miner,
      "--repo", repo,
      "--base", base_sha,
      "--edit", edit_sha,
      "--project-id", pid,
      "--append", out_path,
    ]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
      print(f"[warn] failed: {pid} {pair['edit']}")
      n_skip += 1
    else:
      n_ok += 1
print(f"done: ok={n_ok} skipped={n_skip}")
PY

if [[ -f "$OUT" ]]; then
  wc -l "$OUT"
fi
