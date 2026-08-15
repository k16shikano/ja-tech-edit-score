#!/usr/bin/env python3
"""B の検証 50 件について、人間の推敲と Composer の推敲のブラインド対を作る。

下書きは文脈だけに使い、左右の役割は乱択する。評価器が選んだ文は使わない。
既存の C（人間の推敲対アダプタ選抜）の pairs.jsonl は上書きしない。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from pref_static_utils import load_jsonl
from setwise_triple_utils import reconstruct_triples_from_pref_rows

COMPARE_TYPE = "gold_vs_composer"
SRC_GOLD = "gold"
SRC_COMPOSER = "composer"


def build_pairs(triples: list, *, seed: int) -> list[dict]:
  ordered = sorted(triples, key=lambda t: t.item_id)
  rng = random.Random(seed)
  pairs: list[dict] = []
  for triple in ordered:
    swap = rng.random() < 0.5
    left_text, right_text = triple.human, triple.composer
    left_src, right_src = SRC_GOLD, SRC_COMPOSER
    if swap:
      left_text, right_text = right_text, left_text
      left_src, right_src = right_src, left_src
    pairs.append(
      {
        "pair_id": f"{triple.item_id}::{COMPARE_TYPE}",
        "item_id": triple.item_id,
        "compare_type": COMPARE_TYPE,
        "context_draft": triple.draft,
        "a_text": left_text,
        "b_text": right_text,
        "a_source": left_src,
        "b_source": right_src,
        "swapped": swap,
      }
    )
  rng.shuffle(pairs)
  for i, row in enumerate(pairs):
    row["order"] = i
  return pairs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--valid-file", default="data/section_middle/pref_valid.jsonl")
  parser.add_argument(
    "--out",
    default="data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl",
  )
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  triples = reconstruct_triples_from_pref_rows(load_jsonl(args.valid_file))
  if not triples:
    raise SystemExit(f"no triples in {args.valid_file}")
  pairs = build_pairs(triples, seed=args.seed)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for row in pairs:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  n_swap = sum(1 for row in pairs if row["swapped"])
  print(
    f"wrote {out} pairs={len(pairs)} swapped={n_swap} seed={args.seed}",
    flush=True,
  )


if __name__ == "__main__":
  main()
