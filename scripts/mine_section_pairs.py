#!/usr/bin/env python3
"""節（見出し単位）の source/edited ペアを base..edit 差分から採掘する。

hunk 採掘（mine_branch_pair.py）では空行が落ち、段落境界や節全体の再構成が失われる。
本脚本はファイルを見出し単位に分割し、同じ見出しキーの節どうしをペア化する。
キーが一致しない場合は、本文類似度で 1 対 1 対応を取る。
"""

from __future__ import annotations

import argparse
import difflib
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

# 見出しキー不一致時の本文マッチ下限（無関係節の袋詰めを避ける）
DEFAULT_MIN_CONTENT_SIM = 0.45


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


def align_section_pairs(
  base_sections: dict[str, str],
  edit_sections: dict[str, str],
  *,
  min_content_sim: float,
) -> list[tuple[str, str, str, str, str]]:
  """(pair_key, edit_key_unused, source, edited, match_kind) を返す。"""
  out: list[tuple[str, str, str, str, str]] = []
  matched_base: set[str] = set()
  matched_edit: set[str] = set()

  for key in sorted(set(base_sections) & set(edit_sections)):
    out.append((key, key, base_sections[key], edit_sections[key], "exact_key"))
    matched_base.add(key)
    matched_edit.add(key)

  remaining_base = [(k, v) for k, v in base_sections.items() if k not in matched_base]
  remaining_edit = {k: v for k, v in edit_sections.items() if k not in matched_edit}
  remaining_base.sort(key=lambda kv: len(kv[1]), reverse=True)

  def leaf(key: str) -> str:
    return key.split(">")[-1].strip()

  for b_key, b_text in remaining_base:
    best_key: str | None = None
    best_sim = 0.0
    best_heading = 0.0
    for e_key, e_text in remaining_edit.items():
      # 長さが大きく違う節同士は親子の誤対応が多い
      lr = len(e_text) / max(len(b_text), 1)
      if lr < 0.4 or lr > 2.5:
        continue
      heading_sim = difflib.SequenceMatcher(None, leaf(b_key), leaf(e_key)).ratio()
      if heading_sim < 0.45:
        continue
      sim = difflib.SequenceMatcher(None, b_text, e_text).ratio()
      if sim > best_sim:
        best_sim = sim
        best_key = e_key
        best_heading = heading_sim
    if best_key is None or best_sim < min_content_sim:
      continue
    e_text = remaining_edit.pop(best_key)
    pair_key_label = f"{b_key}<=>{best_key}"
    out.append(
      (
        pair_key_label,
        best_key,
        b_text,
        e_text,
        f"content:{best_sim:.3f}:h{best_heading:.2f}",
      )
    )
    matched_base.add(b_key)
    matched_edit.add(best_key)

  return out


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
  min_content_sim: float = DEFAULT_MIN_CONTENT_SIM,
) -> list[SectionExample]:
  base_text = read_file_at_ref(repo, base, path)
  edit_text = read_file_at_ref(repo, edit, path)
  if base_text is None or edit_text is None:
    return []

  base_sections = split_sections(base_text)
  edit_sections = split_sections(edit_text)
  aligned = align_section_pairs(
    base_sections,
    edit_sections,
    min_content_sim=min_content_sim,
  )

  file_slug = slug(path.replace("/", "__"))
  ref_prefix = (
    f"{path}:section:{base}@{base_commit[:8]}->{edit}@{edit_commit[:8]}"
  )
  created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
  records: list[SectionExample] = []

  for section_key, _edit_key, source, edited, match_kind in aligned:
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
        source_reference=f"{ref_prefix}:{section_key}|{match_kind}",
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
    "review_result": "pending",
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
  from git_pre_merge import assert_structural_edit_pair

  try:
    assert_structural_edit_pair(repo, base_commit, edit_commit)
  except ValueError as exc:
    raise SystemExit(f"invalid revision pair for section mining: {exc}") from exc
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
    if not file_exists_at_ref(repo, args.base, path) or not file_exists_at_ref(repo, args.edit, path):
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
