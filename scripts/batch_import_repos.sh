#!/usr/bin/env bash
# Import edit/* branches for repos listed in a local config file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
GIT="${GIT:-/usr/bin/git}"
MINER="$ROOT/scripts/mine_branch_pair.py"
OUT="$ROOT/data/examples.raw.jsonl"
REPOS_FILE="${REPOS_FILE:-$ROOT/data/batch_import_repos.txt}"

detect_base() {
  local repo="$1"
  if "$GIT" -C "$repo" rev-parse --verify main >/dev/null 2>&1; then
    echo main
  elif "$GIT" -C "$repo" rev-parse --verify master >/dev/null 2>&1; then
    echo master
  else
    "$GIT" -C "$repo" symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@refs/remotes/origin/@@'
  fi
}

list_edit_refs() {
  local repo="$1"
  "$GIT" -C "$repo" branch -a \
    | sed 's/^[* ]*//' \
    | grep -E '(^edit/|remotes/origin/edit/)' \
    | sed 's@remotes/origin/@@' \
    | sort -u
}

import_repo() {
  local repo="$1"
  if [[ ! -d "$repo/.git" ]]; then
    echo "[skip] missing repo: $repo" >&2
    return 0
  fi
  local base
  base="$(detect_base "$repo")"
  if [[ -z "$base" ]]; then
    echo "[skip] no base branch: $repo" >&2
    return 0
  fi
  local project_id
  project_id="$(basename "$repo")"
  local edits
  edits="$(list_edit_refs "$repo" || true)"
  if [[ -z "$edits" ]]; then
    echo "[skip] no edit/* branches: $repo" >&2
    return 0
  fi
  while IFS= read -r edit; do
    [[ -z "$edit" ]] && continue
    local edt="$edit"
    if ! "$GIT" -C "$repo" rev-parse --verify "$edit" >/dev/null 2>&1; then
      edt="origin/$edit"
      if ! "$GIT" -C "$repo" rev-parse --verify "$edt" >/dev/null 2>&1; then
        echo "[skip] cannot resolve branch: $repo $edit" >&2
        continue
      fi
    fi
    echo "[mine] $repo base=$base edit=$edt project=$project_id"
    "$PYTHON" "$MINER" \
      --repo "$repo" \
      --base "$base" \
      --edit "$edt" \
      --project-id "$project_id" \
      --append "$OUT" || echo "[warn] failed: $repo $edt" >&2
  done <<< "$edits"
}

if [[ ! -f "$REPOS_FILE" ]]; then
  echo "repos file not found: $REPOS_FILE" >&2
  echo "copy data/batch_import_repos.example.txt and add one repo path per line" >&2
  exit 1
fi

while IFS= read -r repo || [[ -n "$repo" ]]; do
  repo="${repo%%#*}"
  repo="$(echo "$repo" | xargs)"
  [[ -z "$repo" ]] && continue
  import_repo "$repo"
done < "$REPOS_FILE"

wc -l "$OUT"
