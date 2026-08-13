#!/usr/bin/env python3
"""劣化していないと付けた生成を、節ペアの三つ組み選好に書く。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from section_middle_utils import (
  build_pref_from_judgments,
  judgments_by_id,
  load_jsonl,
  revisions_by_id,
  write_jsonl,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--items", default="data/revision_corpus/keep_section.jsonl")
  parser.add_argument("--revisions", default="data/section_middle/revisions.jsonl")
  parser.add_argument("--judgments", default="data/section_middle/judgments.jsonl")
  parser.add_argument("--out-dir", default="data/section_middle")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  items = load_jsonl(root / args.items)
  revisions = revisions_by_id(load_jsonl(root / args.revisions))
  judgments = judgments_by_id(load_jsonl(root / args.judgments))
  train, valid, stats = build_pref_from_judgments(items, revisions, judgments)
  out_dir = root / args.out_dir
  write_jsonl(out_dir / "pref_train.jsonl", train)
  write_jsonl(out_dir / "pref_valid.jsonl", valid)
  (out_dir / "triple_report.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(stats, ensure_ascii=False))
  print(f"wrote {out_dir / 'pref_train.jsonl'} and {out_dir / 'pref_valid.jsonl'}")


if __name__ == "__main__":
  main()
