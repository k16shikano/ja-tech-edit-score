#!/usr/bin/env python3
"""main にマージ済みの edit ブランチから節ペアを復元する。

mine_section_pairs.py は「現行ブランチと main の差分」を見るため、
マージ済みブランチでは差分が 0 になり採掘できない。
本脚本は main のマージコミットを走査し、
第2親（マージ時点の edit ブランチ先端）と fork 点（merge-base）の対から
節ペアを復元する。学習に使っていない held-out リポジトリの
実編集ペア収集（難試験 v2 の材料）を想定する。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from mine_branch_pair import (
  TEXT_SUFFIXES,
  git_output,
  infer_project_id,
  load_existing_keys,
)
from mine_section_pairs import (
  append_records,
  example_to_dict,
  file_exists_at_ref,
  mine_file_sections,
)

MERGE_SUBJECT_RE = re.compile(
  r"Merge (?:pull request #\d+ from [^ ]*?(?P<pr_branch>edit/[^\s']+)"
  r"|branch '(?P<local_branch>edit/[^']+)')"
)


def list_merges(repo: Path, mainline: str) -> list[tuple[str, str, str, str]]:
  """main 上のマージコミットを (hash, parent1, parent2, subject) で返す。"""
  out = git_output(
    repo, "log", "--merges", "--first-parent", "--format=%H%x00%P%x00%s", mainline, "--"
  )
  merges = []
  for line in out.splitlines():
    parts = line.split("\x00")
    if len(parts) != 3:
      continue
    commit, parents, subject = parts
    parent_list = parents.split()
    if len(parent_list) != 2:
      continue
    merges.append((commit, parent_list[0], parent_list[1], subject))
  return merges


def branch_name_from_subject(subject: str) -> str | None:
  m = MERGE_SUBJECT_RE.search(subject)
  if not m:
    return None
  return m.group("pr_branch") or m.group("local_branch")


def changed_text_paths(repo: Path, base: str, edit: str, only_path: str | None) -> list[str]:
  out = git_output(repo, "diff", "--name-only", f"{base}..{edit}")
  paths = [p for p in out.splitlines() if p.strip()]
  if only_path:
    paths = [p for p in paths if p == only_path]
  return [p for p in paths if Path(p).suffix.lower() in TEXT_SUFFIXES]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", required=True, help="git repository path")
  parser.add_argument("--mainline", default="", help="mainline branch (default: HEAD branch)")
  parser.add_argument("--project-id", default="", help="project id")
  parser.add_argument("--path", default="", help="single file path filter")
  parser.add_argument(
    "--branch-pattern",
    default="edit/",
    help="only merges of branches matching this prefix (default: edit/)",
  )
  parser.add_argument("--append", required=True, help="output JSONL path")
  parser.add_argument("--max-chars", type=int, default=12000)
  args = parser.parse_args()

  repo = Path(args.repo).resolve()
  if not (repo / ".git").exists():
    raise SystemExit(f"not a git repository: {repo}")

  mainline = args.mainline
  if not mainline:
    mainline = git_output(repo, "symbolic-ref", "--short", "HEAD").strip()

  project_id = infer_project_id(repo, args.project_id or None)
  out_path = Path(args.append)
  existing_keys = load_existing_keys(out_path)

  all_records: list[dict] = []
  n_merges = 0
  for commit, p1, p2, subject in list_merges(repo, mainline):
    branch = branch_name_from_subject(subject)
    if not branch or not branch.startswith(args.branch_pattern):
      continue
    try:
      fork = git_output(repo, "merge-base", p1, p2).strip()
    except Exception:
      continue
    if not fork or fork == p2:
      continue
    paths = changed_text_paths(repo, fork, p2, args.path or None)
    if not paths:
      continue
    n_merges += 1
    for path in paths:
      if not file_exists_at_ref(repo, fork, path) or not file_exists_at_ref(repo, p2, path):
        continue
      mined = mine_file_sections(
        repo,
        base=fork,
        edit=p2,
        path=path,
        project_id=project_id,
        base_commit=fork,
        edit_commit=p2,
        max_chars=args.max_chars,
      )
      for example in mined:
        rec = example_to_dict(example)
        rec["meta"]["merged_branch"] = branch
        rec["meta"]["merge_commit"] = commit
        all_records.append(rec)

  appended, skipped = append_records(out_path, all_records, existing_keys)
  print(f"project: {project_id}")
  print(f"merges_with_diff: {n_merges}")
  print(f"mined: {len(all_records)}")
  print(f"appended: {appended}")
  print(f"skipped_duplicates: {skipped}")
  print(f"output: {out_path}")


if __name__ == "__main__":
  main()
