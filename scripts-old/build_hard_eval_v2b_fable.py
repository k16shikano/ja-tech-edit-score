#!/usr/bin/env python3
"""Hard Eval v2b: human / fable / copy の3候補試験。

期待順位（参考）: human > fable > copy
主眼は Top-1 の成否ではなく、スコア差（マージン）が見えるかどうか。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GOLD_RANK = ["human", "fable", "copy"]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source",
    default="data/hard_eval/bases_v2_labeled.jsonl",
    help="human/copy を持つ v2 labeled JSONL",
  )
  parser.add_argument(
    "--fable",
    default="data/hard_eval/bases_v2_fable_revisions.json",
    help="id -> fable 推敲本文の JSON",
  )
  parser.add_argument(
    "--out",
    default="data/hard_eval/bases_v2b_human_fable_copy.jsonl",
  )
  parser.add_argument(
    "--preview",
    default="data/hard_eval/bases_v2b_human_fable_copy_preview.md",
  )
  args = parser.parse_args()

  fable = json.loads(Path(args.fable).read_text(encoding="utf-8"))
  rows = [
    json.loads(line)
    for line in Path(args.source).read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]

  out_rows: list[dict] = []
  missing: list[str] = []
  for row in rows:
    item_id = row["id"]
    if item_id not in fable:
      missing.append(item_id)
      continue
    human = next(c["text"] for c in row["candidates"] if c["id"] == "human")
    copy = row["base_text"]
    fable_text = str(fable[item_id]).strip("\n") + "\n"
    candidates = [
      {"id": "human", "text": human, "generator": "human", "prompt_tag": "real-edit"},
      {
        "id": "fable",
        "text": fable_text,
        "generator": "fable",
        "prompt_tag": "revise-from-copy",
      },
      {"id": "copy", "text": copy, "generator": "copy", "prompt_tag": "identity"},
    ]
    out_rows.append(
      {
        "id": item_id,
        "seed_text": "",
        "seed_meta": {
          **row.get("seed_meta", {}),
          "note": "v2b human/fable/copy; rank is reference expectation",
          "gold_rank_rule": "human > fable > copy (reference; margins matter)",
        },
        "base_text": copy,
        "base_generator": "draft",
        "candidates": candidates,
        "human": {
          "best_id": "human",
          "rank": list(GOLD_RANK),
          "notes": (
            "reference expectation human>fable>copy; "
            "primary interest is score margins, not hard fail on inversions"
          ),
        },
        "status": "labeled",
      }
    )

  if missing:
    raise SystemExit(f"missing fable revisions: {missing}")

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for row in out_rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  lines = [
    "# Hard Eval v2b: human / fable / copy",
    "",
    "参考順位: `human > fable > copy`",
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
  Path(args.preview).write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote {len(out_rows)} items: {out}")
  print(f"wrote {args.preview}")


if __name__ == "__main__":
  main()
