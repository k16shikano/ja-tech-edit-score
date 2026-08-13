#!/usr/bin/env python3
"""Hard Eval v2c: human / machine(composer) / copy の3候補試験を組み立てる。

目的は切り分け。機械負例（composer-2.5 生成）で再学習したモデルが、
「負例と同じ生成モデルの推敲案」だけを人間より下げるのか、
それとも生成モデルを問わず流暢な機械推敲案を下げるのかを見る。

mode=prepare: v2b 試験の下書き（base_text）を generate_machine_revisions.py の
入力形式（id / project_id / draft）に変換する。
mode=build:   生成結果と v2b を突き合わせ、human / machine / copy の試験 JSONL を書く。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GOLD_RANK = ["human", "machine", "copy"]


def load_jsonl(path: Path) -> list[dict]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def prepare(v2b_path: Path, out_path: Path) -> None:
  rows = load_jsonl(v2b_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(
        json.dumps(
          {
            "id": row["id"],
            "project_id": str(row.get("seed_meta", {}).get("project_id") or ""),
            "draft": row["base_text"],
          },
          ensure_ascii=False,
        )
        + "\n"
      )
  print(f"wrote {len(rows)} drafts: {out_path}")


def build(
  v2b_path: Path,
  revisions_path: Path,
  out_path: Path,
  preview_path: Path,
) -> None:
  rows = load_jsonl(v2b_path)
  revisions = load_jsonl(revisions_path)
  text_by_draft_id: dict[str, str] = {}
  rejected: list[str] = []
  for rev in revisions:
    draft_id = str(rev.get("draft_id") or "")
    if rev.get("status") == "rejected":
      rejected.append(f"{draft_id} ({rev.get('reject_reason')})")
      continue
    text_by_draft_id[draft_id] = str(rev.get("text") or "").strip("\n") + "\n"

  out_rows: list[dict] = []
  missing: list[str] = []
  for row in rows:
    item_id = row["id"]
    machine_text = text_by_draft_id.get(item_id)
    if machine_text is None:
      missing.append(item_id)
      continue
    human = next(c["text"] for c in row["candidates"] if c["id"] == "human")
    copy = row["base_text"]
    generator = next(
      (str(r.get("generator") or "") for r in revisions if r.get("draft_id") == item_id),
      "machine",
    )
    out_rows.append(
      {
        "id": item_id,
        "seed_text": "",
        "seed_meta": {
          **row.get("seed_meta", {}),
          "note": "v2c human/machine(composer)/copy; rank is reference expectation",
          "gold_rank_rule": "human > machine > copy (reference; margins matter)",
        },
        "base_text": copy,
        "base_generator": "draft",
        "candidates": [
          {"id": "human", "text": human, "generator": "human", "prompt_tag": "real-edit"},
          {
            "id": "machine",
            "text": machine_text,
            "generator": generator,
            "prompt_tag": "plain-v1",
          },
          {"id": "copy", "text": copy, "generator": "copy", "prompt_tag": "identity"},
        ],
        "human": {
          "best_id": "human",
          "rank": list(GOLD_RANK),
          "notes": (
            "reference expectation human>machine>copy; "
            "primary interest is score margins, not hard fail on inversions"
          ),
        },
        "status": "labeled",
      }
    )

  if rejected:
    print(f"skipped rejected generations: {rejected}", file=sys.stderr)
  if missing:
    print(f"warning: no machine revision for: {missing}", file=sys.stderr)
  if not out_rows:
    raise SystemExit("no items built; run the generation step first")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as f:
    for row in out_rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  lines = [
    "# Hard Eval v2c: human / machine(composer) / copy",
    "",
    "参考順位: `human > machine > copy`",
    "主眼は順位の成否ではなく、スコア差（どの程度よいか）が見えるか。",
    "",
  ]
  for row in out_rows:
    lines.append(f"## {row['id']}")
    lines.append("")
    for cand in row["candidates"]:
      lines.append(f"### Candidate: `{cand['id']}` ({cand['generator']})")
      lines.append("")
      lines.append(cand["text"].strip())
      lines.append("")
  preview_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote {len(out_rows)} items: {out_path}")
  print(f"wrote {preview_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=["prepare", "build"], required=True)
  parser.add_argument(
    "--v2b",
    default="data/hard_eval/bases_v2b_human_fable_copy.jsonl",
    help="v2b（human/fable/copy）試験 JSONL",
  )
  parser.add_argument(
    "--input-out",
    default="data/hard_eval/bases_v2c_composer_input.jsonl",
    help="prepare: 生成用入力 JSONL の出力先",
  )
  parser.add_argument(
    "--revisions",
    default="data/hard_eval/bases_v2c_composer_revisions.jsonl",
    help="build: generate_machine_revisions.py の出力 JSONL",
  )
  parser.add_argument(
    "--out",
    default="data/hard_eval/bases_v2c_human_machine_copy.jsonl",
  )
  parser.add_argument(
    "--preview",
    default="data/hard_eval/bases_v2c_human_machine_copy_preview.md",
  )
  args = parser.parse_args()

  if args.mode == "prepare":
    prepare(Path(args.v2b), Path(args.input_out))
  else:
    build(Path(args.v2b), Path(args.revisions), Path(args.out), Path(args.preview))


if __name__ == "__main__":
  main()
