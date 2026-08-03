#!/usr/bin/env bash
# hunk ペアを pre-merge 比較点で再採掘する。
#
# 運用: 対になる main 側（分岐点）が edit 側より古いこと。さもなければ推敲ではない。
# mainline は origin/main を優先。マージ済み・未マージとも fork..edit（three-dot）のみ。
# 第1親..edit の two-dot は禁止（main 側だけの推敲が逆に見える）。
#
# 既定: section_mining_manifest.json のリポジトリについて、全 edit/* を掘る。
# 出力: data/examples.raw.jsonl（呼び出し前に退避すること）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
GIT="${GIT:-/usr/bin/git}"
MINER="$ROOT/scripts/mine_branch_pair.py"
MANIFEST="${MANIFEST:-$ROOT/data/section_mining_manifest.json}"
OUT="${OUT:-$ROOT/data/examples.raw.jsonl}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

# 追記汚染を避ける。呼び出し前に OUT を退避すること。
# OUT が既にあれば上書きしないよう、空ファイルから始める（APPEND 明示時のみ追記）。
if [[ "${APPEND:-0}" != "1" ]]; then
  rm -f "$OUT"
fi

"$PYTHON" - <<'PY' "$MANIFEST" "$MINER" "$OUT" "$GIT"
import json
import subprocess
import sys
from pathlib import Path

manifest_path, miner, out_path, git = sys.argv[1:5]
sys.path.insert(0, str(Path(miner).resolve().parent))
from git_pre_merge import detect_mainline, resolve_pre_merge_pair  # noqa: E402


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


def list_edit_refs(repo: str) -> list[str]:
  proc = subprocess.run(
    [git, "-C", repo, "branch", "-a"],
    capture_output=True,
    text=True,
    check=True,
  )
  names: set[str] = set()
  for line in proc.stdout.splitlines():
    name = line.strip().lstrip("* ").strip()
    if name.startswith("remotes/origin/"):
      name = name[len("remotes/origin/") :]
    if name.startswith("edit/"):
      names.add(name)
  return sorted(names)


manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
n_ok = 0
n_skip = 0
n_pairs = 0
for entry in manifest:
  repo = entry.get("repo")
  pid = entry["project_id"]
  if not repo or not Path(repo).is_dir() or not (Path(repo) / ".git").exists():
    print(f"[skip] missing repo: {pid} {repo}")
    n_skip += 1
    continue
  mainline = detect_mainline(repo)
  if not mainline:
    print(f"[skip] no mainline: {pid} {repo}")
    n_skip += 1
    continue
  edits = list_edit_refs(repo)
  if not edits:
    print(f"[skip] no edit/*: {pid}")
    n_skip += 1
    continue
  for edit_name in edits:
    edit = resolve_ref(repo, edit_name)
    if not edit:
      print(f"[skip] cannot resolve: {pid} {edit_name}")
      n_skip += 1
      continue
    resolved = resolve_pre_merge_pair(Path(repo), mainline, edit)
    if not resolved:
      print(
        f"[skip] no pre-merge pair: {pid} {edit_name} "
        f"(reedit/summary / late unmerged after editorial merge / "
        f"main not older / already on mainline / empty diff)"
      )
      n_skip += 1
      continue
    base_sha, edit_sha, meta = resolved
    print(
      f"[mine-hunk] {pid} method={meta['method']} "
      f"base={base_sha[:10]} edit={edit_sha[:10]} "
      f"mainline={mainline} branch={meta['edit_branch']} repo={repo}"
    )
    proc = subprocess.run(
      [
        sys.executable,
        miner,
        "--repo",
        repo,
        "--base",
        base_sha,
        "--edit",
        edit_sha,
        "--project-id",
        pid,
        "--append",
        out_path,
        "--label",
        "branch_pair_mined_premerge",
        "--rationale",
        "mined from pre-merge fork..edit hunk diff",
      ],
      text=True,
    )
    if proc.returncode != 0:
      print(f"[warn] failed: {pid} {edit_name}")
      n_skip += 1
    else:
      n_ok += 1
      n_pairs += 1

print(f"done: mined_branch_pairs={n_pairs} ok={n_ok} skipped={n_skip}")
PY

if [[ -f "$OUT" ]]; then
  wc -l "$OUT"
fi
