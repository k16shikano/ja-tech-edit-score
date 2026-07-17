#!/usr/bin/env python3
"""Markdown プレビュー内の候補順を Hard Eval の人手順位へ変換する。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ITEM_HEADING = re.compile(r"^## (he-[^\s]+)\s*$")
CANDIDATE_HEADING = re.compile(r"^### `([^`]+)` \([^)]+\)\s*$")


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
      if not line.strip():
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
  return rows


def normalize_body(lines: list[str]) -> str:
  return "\n".join(lines).strip()


def parse_preview(path: Path) -> dict[str, list[tuple[str, str]]]:
  """Return item id -> [(candidate id, body)] in Markdown order."""
  items: dict[str, list[tuple[str, str]]] = {}
  current_item: str | None = None
  current_candidate: str | None = None
  body: list[str] = []

  def finish_candidate() -> None:
    nonlocal current_candidate, body
    if current_item is None or current_candidate is None:
      return
    items[current_item].append((current_candidate, normalize_body(body)))
    current_candidate = None
    body = []

  for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    item_match = ITEM_HEADING.match(line)
    if item_match:
      finish_candidate()
      current_item = item_match.group(1)
      if current_item in items:
        raise ValueError(f"{path}:{line_no}: duplicate item heading: {current_item}")
      items[current_item] = []
      continue

    candidate_match = CANDIDATE_HEADING.match(line)
    if candidate_match:
      if current_item is None:
        raise ValueError(f"{path}:{line_no}: candidate outside item")
      finish_candidate()
      current_candidate = candidate_match.group(1)
      continue

    if current_candidate is not None:
      body.append(line)

  finish_candidate()
  return items


def apply_ranks(rows: list[dict], preview: dict[str, list[tuple[str, str]]]) -> list[dict]:
  row_ids = [row.get("id") for row in rows]
  if len(row_ids) != len(set(row_ids)):
    raise ValueError("source JSONL contains duplicate item ids")
  if set(row_ids) != set(preview):
    missing = sorted(set(row_ids) - set(preview))
    extra = sorted(set(preview) - set(row_ids))
    raise ValueError(f"item mismatch: missing={missing}, extra={extra}")

  output: list[dict] = []
  for row in rows:
    item_id = row["id"]
    candidates = row.get("candidates") or []
    candidate_by_id = {candidate.get("id"): candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
      raise ValueError(f"{item_id}: duplicate candidate ids in source JSONL")

    ordered_blocks = preview[item_id]
    rank = [candidate_id for candidate_id, _ in ordered_blocks]
    if len(rank) != len(set(rank)):
      raise ValueError(f"{item_id}: duplicate candidate heading in preview")
    if set(rank) != set(candidate_by_id):
      missing = sorted(set(candidate_by_id) - set(rank))
      extra = sorted(set(rank) - set(candidate_by_id))
      raise ValueError(f"{item_id}: candidate mismatch: missing={missing}, extra={extra}")

    for candidate_id, body in ordered_blocks:
      expected = str(candidate_by_id[candidate_id].get("text") or "").strip()
      if body != expected:
        raise ValueError(
          f"{item_id}/{candidate_id}: candidate body differs from source JSONL; "
          "move the entire subsection without editing it"
        )

    labeled = dict(row)
    labeled["human"] = {
      **(row.get("human") or {}),
      "best_id": rank[0],
      "rank": rank,
    }
    labeled["status"] = "labeled"
    output.append(labeled)
  return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--preview", required=True, help="順位どおりに並べ替えた Markdown")
  parser.add_argument("--source", required=True, help="候補本文を持つ元 JSONL")
  parser.add_argument("--output", help="ラベル済み JSONL の出力先")
  parser.add_argument(
    "--check-only",
    action="store_true",
    help="Markdown と元 JSONL の整合性だけを検査し、出力しない",
  )
  args = parser.parse_args()

  if not args.check_only and not args.output:
    parser.error("--output is required unless --check-only is set")

  rows = load_jsonl(Path(args.source))
  preview = parse_preview(Path(args.preview))
  labeled = apply_ranks(rows, preview)

  if args.check_only:
    print(f"valid: {len(labeled)} items, no output written")
    return

  output = Path(args.output)
  write_jsonl(output, labeled)
  print(f"wrote {len(labeled)} labeled items: {output}")


if __name__ == "__main__":
  main()
