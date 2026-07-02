#!/usr/bin/env python3
"""Git helpers shared by mining and revision check."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


TEXT_SUFFIXES = {".md", ".rst", ".txt", ".adoc", ".org"}


def git_executable() -> str:
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


def git_output_optional(repo: Path, *args: str) -> str | None:
  cmd = [git_executable(), *args]
  proc = subprocess.run(cmd, cwd=repo, text=True, errors="replace", capture_output=True)
  if proc.returncode != 0:
    return None
  return proc.stdout


def resolve_commit(repo: Path, ref: str) -> str:
  return git_output(repo, "rev-parse", "--verify", ref).strip()


def repo_root_for(path: Path) -> Path:
  out = git_output(path.parent if path.is_file() else path, "rev-parse", "--show-toplevel").strip()
  return Path(out)


def current_branch(repo: Path) -> str:
  return git_output(repo, "branch", "--show-current").strip()


def infer_base_from_edit_branch(edit_branch: str) -> str | None:
  if edit_branch.startswith("edit/"):
    return edit_branch[len("edit/") :]
  return None


def detect_base_branch(repo: Path, edit_branch: str) -> str:
  env_base = os.environ.get("JA_TECH_EDIT_SCORE_BASE", "").strip()
  if env_base:
    resolve_commit(repo, env_base)
    return env_base

  inferred = infer_base_from_edit_branch(edit_branch)
  if inferred and git_output_optional(repo, "rev-parse", "--verify", inferred):
    return inferred

  for candidate in ("main", "master"):
    if git_output_optional(repo, "rev-parse", "--verify", candidate):
      return candidate

  raise SystemExit(
    f"cannot infer base branch for edit branch {edit_branch!r}; "
    "pass --base or set JA_TECH_EDIT_SCORE_BASE"
  )


def rel_path_in_repo(repo: Path, file_arg: str) -> str:
  path = Path(file_arg).expanduser()
  if not path.is_absolute():
    path = (Path.cwd() / path).resolve()
  else:
    path = path.resolve()
  if not path.is_file():
    raise SystemExit(f"file not found: {path}")
  try:
    return str(path.relative_to(repo.resolve()))
  except ValueError as exc:
    raise SystemExit(f"file is outside git repo {repo}: {path}") from exc


def head_commit(repo: Path) -> str:
  return resolve_commit(repo, "HEAD")


def worktree_dirty_for(repo: Path, rel_path: str) -> bool:
  out = git_output(repo, "status", "--porcelain", "--", rel_path).strip()
  return bool(out)


def describe_edit_side(repo: Path, rel_path: str, *, branch: str) -> tuple[str, str]:
  """Return (human label, commit/id token for metadata)."""
  head = head_commit(repo)
  dirty = worktree_dirty_for(repo, rel_path)
  branch_part = branch or "HEAD"
  if dirty:
    return f"作業ツリー ({branch_part}, 未コミットあり)", f"{head[:8]}+dirty"
  return f"作業ツリー ({branch_part})", head[:8]
