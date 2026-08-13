#!/usr/bin/env python3
"""完成稿（edit ブランチ）から段落遷移の正例・負例ペアを抽出する。

正例: 同一節内の隣接段落対。
負例: 同一書籍内の非隣接段落対（同一節内 distance>=2 と別節を混ぜる）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from build_hard_eval_v2 import MARKERS, paragraphs, split_heading_body
from markdown_sections import split_sections
from mine_section_pairs import read_file_at_ref
from mine_branch_pair import git_output, slug

HEADING_RE = re.compile(r"^#{1,6}\s")
MIN_PARA_LEN = 30

HELDOUT_REPOS = (
  "/home/k16/work/Nmonthly/dev-with-type",
  "/home/k16/work/Nmonthly/lang-on-wasm",
  "/home/k16/work/Nmonthly/lean-by-example",
  "/home/k16/work/Nmonthly/ml-feature-store",
  "/home/k16/work/Nmonthly/nix",
  "/home/k16/work/Nmonthly/pfvm",
  "/home/k16/work/Nmonthly/picoruby",
  "/home/k16/work/Nmonthly/smtp-revisit",
  "/home/k16/work/Nmonthly/websocket-from-security",
)
HELDOUT_PROJECT_IDS = {Path(p).name for p in HELDOUT_REPOS}


@dataclass(frozen=True)
class ParaRef:
  text: str
  path: str
  section_key: str
  index: int


@dataclass(frozen=True)
class SectionParas:
  path: str
  section_key: str
  edit_ref: str
  paragraphs: tuple[str, ...]


def resolve_ref(repo: Path, ref: str) -> str | None:
  for candidate in (ref, f"origin/{ref}"):
    try:
      git_output(repo, "rev-parse", "--verify", candidate)
      return candidate
    except Exception:
      continue
  return None


def has_markers(text: str) -> bool:
  return any(marker in text for marker in MARKERS)


def is_usable_paragraph(text: str) -> bool:
  stripped = text.strip()
  if len(stripped) < MIN_PARA_LEN:
    return False
  if HEADING_RE.match(stripped):
    return False
  if has_markers(stripped):
    return False
  if not re.search(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9]", stripped):
    return False
  return True


def filtered_paragraphs(text: str) -> list[str]:
  """節本文から学習・評価で共通のフィルタを通した段落列を返す。"""
  _heading, body = split_heading_body(text)
  return [p for p in paragraphs(body) if is_usable_paragraph(p)]


def pair_id(project_id: str, path: str, section_key: str, text_a: str, text_b: str, label: int) -> str:
  digest = hashlib.sha1(f"{text_a}\0{text_b}".encode("utf-8")).hexdigest()[:12]
  return f"{project_id}__{slug(path)}__{slug(section_key)}__{label}__{digest}"


def collect_sections(repo: Path, edit_ref: str, path: str) -> list[SectionParas]:
  text = read_file_at_ref(repo, edit_ref, path)
  if text is None:
    return []

  sections: list[SectionParas] = []
  for section_key, section_text in split_sections(text).items():
    _heading, body = split_heading_body(section_text)
    usable = tuple(p for p in paragraphs(body) if is_usable_paragraph(p))
    if len(usable) >= 2:
      sections.append(
        SectionParas(
          path=path,
          section_key=section_key,
          edit_ref=edit_ref,
          paragraphs=usable,
        )
      )
  return sections


def positive_pairs(project_id: str, sections: list[SectionParas]) -> list[dict]:
  rows: list[dict] = []
  for section in sections:
    for i in range(len(section.paragraphs) - 1):
      text_a = section.paragraphs[i]
      text_b = section.paragraphs[i + 1]
      rows.append(
        {
          "id": pair_id(project_id, section.path, section.section_key, text_a, text_b, 1),
          "project_id": project_id,
          "text_a": text_a,
          "text_b": text_b,
          "label": 1,
          "meta": {
            "path": section.path,
            "section_key": section.section_key,
            "distance": 1,
            "edit_ref": section.edit_ref,
          },
        }
      )
  return rows


def negative_pool(project_id: str, sections: list[SectionParas]) -> list[tuple[ParaRef, ParaRef, int]]:
  """非隣接候補 (A, B, distance) を列挙する。"""
  refs: list[ParaRef] = []
  for section in sections:
    for idx, text in enumerate(section.paragraphs):
      refs.append(
        ParaRef(
          text=text,
          path=section.path,
          section_key=section.section_key,
          index=idx,
        )
      )

  pool: list[tuple[ParaRef, ParaRef, int]] = []
  for i, a in enumerate(refs):
    for j, b in enumerate(refs):
      if i >= j:
        continue
      same_section = a.path == b.path and a.section_key == b.section_key
      if same_section:
        distance = abs(a.index - b.index)
        if distance < 2:
          continue
      else:
        distance = -1
      pool.append((a, b, distance))
  return pool


def sample_negatives(
  project_id: str,
  pool: list[tuple[ParaRef, ParaRef, int]],
  n: int,
  rng: random.Random,
) -> list[dict]:
  if not pool or n <= 0:
    return []
  picks = [pool[rng.randrange(len(pool))] for _ in range(n)]
  rows: list[dict] = []
  for a, b, distance in picks:
    rows.append(
      {
        "id": pair_id(project_id, a.path, a.section_key, a.text, b.text, 0),
        "project_id": project_id,
        "text_a": a.text,
        "text_b": b.text,
        "label": 0,
        "meta": {
          "path": a.path,
          "section_key": a.section_key,
          "distance": distance,
        },
      }
    )
  return rows


def extract_from_manifest(manifest_path: Path, *, seed: int) -> tuple[list[dict], dict]:
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  rng = random.Random(seed)
  all_rows: list[dict] = []
  stats = {
    "projects": 0,
    "skipped_heldout": 0,
    "positive": 0,
    "negative": 0,
    "sections": 0,
  }

  for entry in manifest:
    project_id = entry["project_id"]
    if project_id in HELDOUT_PROJECT_IDS:
      stats["skipped_heldout"] += 1
      continue

    repo = Path(entry["repo"])
    if not repo.is_dir() or not (repo / ".git").exists():
      print(f"[skip] missing repo: {project_id} {repo}", file=sys.stderr)
      continue

    sections: list[SectionParas] = []
    paths = entry.get("paths") or []
    for pair in entry.get("branch_pairs", []):
      edit_ref = resolve_ref(repo, pair["edit"])
      if not edit_ref:
        print(f"[skip] unresolved edit ref: {project_id} {pair['edit']}", file=sys.stderr)
        continue
      for path in paths:
        sections.extend(collect_sections(repo, edit_ref, path))

    if not sections:
      print(f"[skip] no sections: {project_id}", file=sys.stderr)
      continue

    stats["projects"] += 1
    stats["sections"] += len(sections)

    pos_rows = positive_pairs(project_id, sections)
    pool = negative_pool(project_id, sections)
    neg_rows = sample_negatives(project_id, pool, len(pos_rows), rng)

    all_rows.extend(pos_rows)
    all_rows.extend(neg_rows)
    stats["positive"] += len(pos_rows)
    stats["negative"] += len(neg_rows)
    print(
      f"{project_id:28s} sections={len(sections):4d} "
      f"pos={len(pos_rows):5d} neg={len(neg_rows):5d} pool={len(pool):6d}"
    )

  return all_rows, stats


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--manifest",
    default="data/section_mining_manifest.json",
    help="学習用リポジトリ manifest",
  )
  parser.add_argument(
    "--out",
    default="data/paragraph_transitions.jsonl",
    help="出力 JSONL",
  )
  parser.add_argument("--seed", type=int, default=42, help="負例サンプリング seed")
  args = parser.parse_args()

  root = Path(__file__).resolve().parent.parent
  manifest_path = Path(args.manifest)
  if not manifest_path.is_absolute():
    manifest_path = root / manifest_path
  out_path = Path(args.out)
  if not out_path.is_absolute():
    out_path = root / out_path

  rows, stats = extract_from_manifest(manifest_path, seed=args.seed)
  if not rows:
    raise SystemExit("no transition pairs extracted")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")

  print("-" * 64)
  print(
    f"projects={stats['projects']} sections={stats['sections']} "
    f"positive={stats['positive']} negative={stats['negative']}"
  )
  print(f"wrote: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
  main()
