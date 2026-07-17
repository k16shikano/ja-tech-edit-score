#!/usr/bin/env python3
"""節（見出し単位）の source/edited ペアを base..edit 差分から採掘する。

hunk 採掘（mine_branch_pair.py）では空行が落ち、段落境界や節全体の再構成が失われる。
本脚本はファイルを見出し単位に分割し、同じ見出しキーの節どうしをペア化する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from markdown_sections import line_count, paragraph_count, split_sections
from mine_branch_pair import (
  TEXT_SUFFIXES,
  git_output,
  infer_project_id,
  list_changed_paths,
  load_existing_keys,
  pair_key,
  resolve_commit,
  slug,
)


@dataclass
class SectionExample:
  id: str
  project_id: str
  source_text: str
  edited_text: str
  source_reference: str
  created_at: str
  section_key: str
  path: str
  paragraph_count_source: int
  paragraph_count_edited: int
  line_count_source: int
  line_count_edited: int


def read_file_at_ref(repo: Path, ref: str, path: str) -> str | None:
  try:
    git_output(repo, "cat-file", "-e", f"{ref}:{path}")
    return git_output(repo, "show", f"{ref}:{path}")
  except Exception:
    return None


def file_exists_at_ref(repo: Path, ref: str, path: str) -> bool:
  try:
    git_output(repo, "cat-file", "-e", f"{ref}:{path}")
    return True
  except Exception:
    return False


def is_usable_section_pair(source: str, edited: str, *, max_chars: int) -> bool:
  if not source.strip() or not edited.strip():
    return False
  if source.strip() == edited.strip():
    return False
  if len(source) > max_chars or len(edited) > max_chars:
    return False
  if not re.search(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9]", source + edited):
    return False
  return True


def mine_file_sections(
  repo: Path,
  *,
  base: str,
  edit: str,
  path: str,
  project_id: str,
  base_commit: str,
  edit_commit: str,
  max_chars: int,
) -> list[SectionExample]:
  base_text = read_file_at_ref(repo, base, path)
  edit_text = read_file_at_ref(repo, edit, path)
  if base_text is None or edit_text is None:
    return []

  base_sections = split_sections(base_text)
  edit_sections = split_sections(edit_text)
  keys = sorted(set(base_sections) | set(edit_sections))

  file_slug = slug(path.replace("/", "__"))
  ref_prefix = (
    f"{path}:section:{base}@{base_commit[:8]}->{edit}@{edit_commit[:8]}"
  )
  created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
  records: list[SectionExample] = []

  for section_key in keys:
    source = base_sections.get(section_key, "")
    edited = edit_sections.get(section_key, "")
    if not is_usable_section_pair(source, edited, max_chars=max_chars):
      continue
    content_hash = hashlib.sha256(
      (section_key + "\n---\n" + source + "\n---\n" + edited).encode("utf-8")
    ).hexdigest()[:12]
    key_slug = slug(section_key)[:48] or "section"
    records.append(
      SectionExample(
        id=f"{project_id}-{file_slug}-{key_slug}-{content_hash}",
        project_id=project_id,
        source_text=source,
        edited_text=edited,
        source_reference=f"{ref_prefix}:{section_key}",
        created_at=created_at,
        section_key=section_key,
        path=path,
        paragraph_count_source=paragraph_count(source),
        paragraph_count_edited=paragraph_count(edited),
        line_count_source=line_count(source),
        line_count_edited=line_count(edited),
      )
    )
  return records


def example_to_dict(example: SectionExample) -> dict:
  return {
    "id": example.id,
    "project_id": example.project_id,
    "source_text": example.source_text,
    "edited_text": example.edited_text,
    "source_reference": example.source_reference,
    "rationale": "mined from base..edit section diff",
    "labels": ["section_pair_mined"],
    "author": "human",
    "review_result": "accepted",
    "created_at": example.created_at,
    "meta": {
      "granularity": "section",
      "section_key": example.section_key,
      "path": example.path,
      "paragraph_count_source": example.paragraph_count_source,
      "paragraph_count_edited": example.paragraph_count_edited,
      "line_count_source": example.line_count_source,
      "line_count_edited": example.line_count_edited,
    },
  }


def append_records(
  out_path: Path,
  records: list[dict],
  existing_keys: set[tuple[str, str, str, str]],
) -> tuple[int, int]:
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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", required=True, help="git repository path")
  parser.add_argument("--base", required=True, help="base branch")
  parser.add_argument("--edit", required=True, help="edit branch")
  parser.add_argument("--project-id", default="", help="project id")
  parser.add_argument("--path", default="", help="single file path (default: all changed text files)")
  parser.add_argument("--append", required=True, help="output JSONL path")
  parser.add_argument(
    "--max-chars",
    type=int,
    default=12000,
    help="skip sections longer than this (default: 12000)",
  )
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
    if Path(path).suffix.lower() not in TEXT_SUFFIXES:
      continue
    if not file_exists_at_ref(repo, base, path) or not file_exists_at_ref(repo, edit, path):
      continue
    mined = mine_file_sections(
      repo,
      base=args.base,
      edit=args.edit,
      path=path,
      project_id=project_id,
      base_commit=base_commit,
      edit_commit=edit_commit,
      max_chars=args.max_chars,
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
