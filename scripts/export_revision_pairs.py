#!/usr/bin/env python3
"""dpo_curated から下書き/推敲後の対照ペアを書き出す。

activation steering と編集モデル実験の共通前処理。
[参照] ブロックは本文から除去する。原稿パスは meta に載せない。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from steering_utils import (
  iter_revision_pairs_from_dpo,
  load_jsonl,
  write_jsonl,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", default="data/dpo_curated.jsonl", help="DPO curated JSONL")
  parser.add_argument("--out", default="data/revision_pairs.jsonl", help="output revision pairs")
  parser.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
  args = parser.parse_args()

  in_path = Path(args.input)
  if not in_path.is_file():
    raise SystemExit(f"missing input: {in_path}")

  pairs = iter_revision_pairs_from_dpo(load_jsonl(in_path))
  if args.limit > 0:
    pairs = pairs[: args.limit]

  out_path = Path(args.out)
  write_jsonl(out_path, pairs)

  projects = sorted({p["project_id"] for p in pairs})
  print(f"exported: {len(pairs)} pairs -> {out_path}")
  print(f"projects: {len(projects)}")


if __name__ == "__main__":
  main()
