#!/usr/bin/env python3
"""独立人手判定（PLAN 8d）の下書き 40 件を、学習前に固定する。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pref_static_utils import load_jsonl
from setwise_triple_utils import reconstruct_triples_from_pref_rows

INSTRUCTION = (
  "次の下書きを、意味を保ったまま日本語の技術文書として推敲せよ。\n"
  "行頭の # で始まる見出しと、【図】・【コード】などの置き場所を示す要素は原稿の構成要素である。"
  "推敲の一部として位置を動かすのはよいが、省いてはならない。\n"
  "文体（ですます調・である調）は下書きのまま保て。変えてはならない。\n"
  "出力は推敲後の本文のみ。前置き・後書き・変更点の説明・「このようにすると…」のようなメタ文言は書くな。"
)


def _triple_ids(path: Path) -> set[str]:
  if not path.is_file():
    return set()
  return {t.item_id for t in reconstruct_triples_from_pref_rows(load_jsonl(str(path)))}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--hunk-heldout", default="data/pref_keep_split_hunk/valid.jsonl")
  parser.add_argument("--section-train", default="data/section_middle/pref_train.jsonl")
  parser.add_argument("--section-valid", default="data/section_middle/pref_valid.jsonl")
  parser.add_argument("--out-items", default="data/blind_eval/8d_items.jsonl")
  parser.add_argument("--out-protocol", default="data/blind_eval/8d_protocol.json")
  parser.add_argument("--n", type=int, default=40)
  parser.add_argument("--min-chars", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  excluded = _triple_ids(Path(args.section_train)) | _triple_ids(Path(args.section_valid))
  pool: list[dict] = []
  for row in load_jsonl(args.hunk_heldout):
    meta = row.get("meta") or {}
    if str(meta.get("pair_order") or "chosen_first") != "chosen_first":
      continue
    if int(row.get("label", 0)) != 1:
      continue
    item_id = str(row.get("id") or "")
    if not item_id or item_id in excluded:
      continue
    draft = str(row.get("candidate_b") or "")
    gold = str(row.get("candidate_a") or "")
    if len(draft) < args.min_chars:
      continue
    pool.append(
      {
        "id": item_id,
        "draft": draft,
        "gold": gold,
        "unit": str(meta.get("unit") or "hunk"),
      }
    )

  rng = random.Random(args.seed)
  rng.shuffle(pool)
  selected = pool[: args.n]
  if len(selected) < args.n:
    raise SystemExit(f"need {args.n} items, got {len(selected)} after filters")

  items_path = Path(args.out_items)
  items_path.parent.mkdir(parents=True, exist_ok=True)
  with items_path.open("w", encoding="utf-8") as f:
    for row in selected:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  protocol = {
    "n_items": args.n,
    "seed": args.seed,
    "min_chars": args.min_chars,
    "source": args.hunk_heldout,
    "excluded": "data/section_middle/pref_train.jsonl と pref_valid.jsonl の item_id",
    "instruction": INSTRUCTION,
    "candidates": [
      {
        "id": "composer_seed1",
        "generator": "composer-2.5",
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 1,
      },
      {
        "id": "composer_seed2",
        "generator": "composer-2.5",
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 2,
      },
    ],
    "judgment": {
      "view": "左右の役割を見せない。下書きと二案を並べ、どちらがよいかを人が付ける",
      "options": ["a", "b", "tie"],
      "scorer_selection": False,
    },
    "aggregation": {
      "primary": "人が付けた勝敗と、段階 5 の評価器が二案に付けた点の上下が一致した件数",
      "denominator": "人が tie としなかった件",
    },
    "items_file": str(items_path),
  }
  Path(args.out_protocol).write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"wrote {items_path} n={len(selected)}", flush=True)
  print(f"wrote {args.out_protocol}", flush=True)


if __name__ == "__main__":
  main()
