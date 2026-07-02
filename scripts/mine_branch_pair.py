#!/usr/bin/env python3
"""Mine source/edited pairs from a base branch vs edit branch diff."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {".md", ".rst", ".txt", ".adoc", ".org"}


@dataclass
class Example:
  id: str
  project_id: str
  source_text: str
  edited_text: str
  source_reference: str
  created_at: str
  line_old: int | None = None
  line_new: int | None = None


def slug(s: str) -> str:
  return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-") or "x"


def clean_lines(lines: list[str]) -> list[str]:
  return [line.rstrip() for line in lines if line.rstrip()]


def is_usable_pair(source: str, edited: str) -> bool:
  if not source or not edited:
    return False
  if len(source) > 1800 or len(edited) > 1800:
    return False
  if not re.search(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9]", source + edited):
    return False
  return True


def pair_key(rec: dict) -> tuple[str, str, str, str]:
  return (
    rec.get("project_id", ""),
    rec.get("source_text", ""),
    rec.get("edited_text", ""),
    rec.get("source_reference", ""),
  )


def load_existing_keys(path: Path) -> set[tuple[str, str, str, str]]:
  if not path.exists():
    return set()
  keys: set[tuple[str, str, str, str]] = set()
  with path.open(encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      keys.add(pair_key(json.loads(line)))
  return keys


def git_executable() -> str:
  import os
  import shutil

  for candidate in (
    os.environ.get("GIT", ""),
    shutil.which("git"),
    "/usr/bin/git",
    "/bin/git",
  ):
    if candidate and Path(candidate).is_file():
      return candidate
  raise SystemExit("git executable not found; set GIT=/path/to/git")


def git_output(repo: Path, *args: str) -> str:
  cmd = [git_executable(), *args]
  return subprocess.check_output(cmd, cwd=repo, text=True, errors="replace")


def resolve_commit(repo: Path, ref: str) -> str:
  return git_output(repo, "rev-parse", ref).strip()


def list_changed_paths(repo: Path, base: str, edit: str, path_filter: str | None) -> list[str]:
  if path_filter:
    return [path_filter]
  out = git_output(repo, "diff", "--name-only", f"{base}..{edit}")
  paths = []
  for line in out.splitlines():
    line = line.strip()
    if not line:
      continue
    if Path(line).suffix.lower() in TEXT_SUFFIXES:
      paths.append(line)
  return sorted(set(paths))


HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_diff_to_examples(
  diff: str,
  *,
  path: str,
  project_id: str,
  base_label: str,
  edit_label: str,
  base_commit: str,
  edit_commit: str,
) -> list[Example]:
  """Parse unified diff text into source/edited hunk pairs."""
  records: list[Example] = []
  minus: list[str] = []
  plus: list[str] = []
  line_old: int | None = None
  line_new: int | None = None
  file_slug = slug(path.replace("/", "__"))
  ref_prefix = f"{path}:{base_label}@{base_commit[:8]}->{edit_label}@{edit_commit[:8]}"

  def flush() -> None:
    nonlocal minus, plus, line_old, line_new
    source = "\n".join(clean_lines(minus))
    edited = "\n".join(clean_lines(plus))
    old_line, new_line = line_old, line_new
    minus = []
    plus = []
    line_old = None
    line_new = None
    if not is_usable_pair(source, edited):
      return
    content_hash = hashlib.sha256((source + "\n---\n" + edited).encode("utf-8")).hexdigest()[:12]
    records.append(
      Example(
        id=f"{project_id}-{file_slug}-{content_hash}",
        project_id=project_id,
        source_text=source,
        edited_text=edited,
        source_reference=ref_prefix,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        line_old=old_line,
        line_new=new_line,
      )
    )

  for line in diff.splitlines():
    if line.startswith("@@ "):
      flush()
      match = HUNK_HEADER_RE.match(line)
      if match:
        line_old = int(match.group(1))
        line_new = int(match.group(2))
      continue
    if line.startswith("diff --git ") or line.startswith("--- ") or line.startswith("+++ "):
      continue
    if line.startswith("-"):
      minus.append(line[1:])
      continue
    if line.startswith("+"):
      plus.append(line[1:])
      continue

  flush()
  return records


def mine_worktree_diff(
  repo: Path,
  *,
  base: str,
  path: str,
  project_id: str,
  edit_branch: str,
) -> list[Example]:
  """Diff base branch tip against current working tree (includes uncommitted edits)."""
  base_commit = resolve_commit(repo, base)
  edit_label, edit_token = _worktree_edit_token(repo, path, edit_branch)
  diff = git_output(repo, "diff", "--unified=0", base, "--", path)
  return parse_diff_to_examples(
    diff,
    path=path,
    project_id=project_id,
    base_label=base,
    edit_label=edit_label,
    base_commit=base_commit,
    edit_commit=edit_token,
  )


def _worktree_edit_token(repo: Path, path: str, edit_branch: str) -> tuple[str, str]:
  from git_branch_utils import head_commit, worktree_dirty_for

  head = head_commit(repo)
  dirty = worktree_dirty_for(repo, path)
  token = head[:8] + ("+dirty" if dirty else "")
  if edit_branch:
    label = f"{edit_branch}+working-tree" if dirty else f"{edit_branch}@working-tree"
  else:
    label = "working-tree" + ("*" if dirty else "")
  return label, token


def mine_file_diff(
  repo: Path,
  *,
  base: str,
  edit: str,
  path: str,
  project_id: str,
  base_commit: str,
  edit_commit: str,
) -> list[Example]:
  diff = git_output(repo, "diff", "--unified=0", f"{base}..{edit}", "--", path)
  return parse_diff_to_examples(
    diff,
    path=path,
    project_id=project_id,
    base_label=base,
    edit_label=edit,
    base_commit=base_commit,
    edit_commit=edit_commit,
  )


def example_to_dict(example: Example) -> dict:
  return {
    "id": example.id,
    "project_id": example.project_id,
    "source_text": example.source_text,
    "edited_text": example.edited_text,
    "source_reference": example.source_reference,
    "rationale": "mined from base..edit branch diff",
    "labels": ["branch_pair_mined"],
    "author": "human",
    "review_result": "accepted",
    "created_at": example.created_at,
  }


def append_records(out_path: Path, records: Iterable[dict], existing_keys: set[tuple[str, str, str, str]]) -> tuple[int, int]:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  appended = 0
  skipped = 0
  mode = "a" if out_path.exists() else "w"
  with out_path.open(mode, encoding="utf-8") as handle:
    for rec in records:
      key = pair_key(rec)
      if key in existing_keys:
        skipped += 1
        continue
      handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
      existing_keys.add(key)
      appended += 1
  return appended, skipped


def infer_project_id(repo: Path, explicit: str | None) -> str:
  if explicit:
    return explicit
  return slug(repo.name) or "project"


def main() -> None:
  parser = argparse.ArgumentParser(description="Mine training pairs from git base..edit diff")
  parser.add_argument("--repo", required=True, help="git repository path")
  parser.add_argument("--base", required=True, help="base branch (ORG)")
  parser.add_argument("--edit", required=True, help="edit branch (EDT)")
  parser.add_argument("--project-id", default="", help="project id (default: repo directory name)")
  parser.add_argument("--path", default="", help="single file path inside repo (default: all changed text files)")
  parser.add_argument("--append", required=True, help="output JSONL path to append examples.raw.jsonl")
  args = parser.parse_args()

  repo = Path(args.repo).resolve()
  if not (repo / ".git").exists():
    raise SystemExit(f"not a git repository: {repo}")

  project_id = infer_project_id(repo, args.project_id or None)
  base_commit = resolve_commit(repo, args.base)
  edit_commit = resolve_commit(repo, args.edit)
  paths = list_changed_paths(repo, args.base, args.edit, args.path or None)
  if not paths:
    print("no changed text files", file=sys.stderr)
    return

  out_path = Path(args.append)
  existing_keys = load_existing_keys(out_path)

  all_records: list[dict] = []
  for path in paths:
    mined = mine_file_diff(
      repo,
      base=args.base,
      edit=args.edit,
      path=path,
      project_id=project_id,
      base_commit=base_commit,
      edit_commit=edit_commit,
    )
    for example in mined:
      all_records.append(example_to_dict(example))

  appended, skipped = append_records(out_path, all_records, existing_keys)
  print(f"paths: {len(paths)}")
  print(f"mined: {len(all_records)}")
  print(f"appended: {appended}")
  print(f"skipped_duplicates: {skipped}")
  print(f"output: {out_path}")


if __name__ == "__main__":
  main()
