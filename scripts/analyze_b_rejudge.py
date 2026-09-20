#!/usr/bin/env python3
"""B 付け直し判定の choice 内訳を集計する（本文は出さない）。"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from pref_static_utils import load_jsonl


def analyze(pairs: list[dict], judgments: list[dict]) -> dict:
  by_pair = {str(p["pair_id"]): p for p in pairs}
  choice_counts: Counter[str] = Counter()
  reason_counts: Counter[str] = Counter()
  compare_counts: Counter[str] = Counter()
  items_done: set[str] = set()
  judged_pair_ids: set[str] = set()

  for row in judgments:
    pid = str(row.get("pair_id") or "")
    if not pid:
      continue
    judged_pair_ids.add(pid)
    choice = str(row.get("choice") or "")
    choice_counts[choice] += 1
    if choice == "incomparable":
      reason_counts[str(row.get("incomparable_reason") or "other")] += 1
    pair = by_pair.get(pid)
    if pair:
      items_done.add(str(pair.get("item_id") or ""))
      compare_counts[str(pair.get("compare_type") or "")] += 1

  items_total = len({str(p.get("item_id") or "") for p in pairs})
  pairs_per_item = defaultdict(int)
  for p in pairs:
    pairs_per_item[str(p.get("item_id") or "")] += 1

  return {
    "n_pairs_total": len(pairs),
    "n_pairs_judged": len(judged_pair_ids),
    "n_items_total": items_total,
    "n_items_any_judged": len(items_done),
    "n_items_complete": sum(
      1
      for iid, n in pairs_per_item.items()
      if sum(
        1
        for j in judgments
        if str(by_pair.get(str(j.get("pair_id") or ""), {}).get("item_id") or "") == iid
      )
      >= n
    ),
    "choice_counts": dict(sorted(choice_counts.items())),
    "incomparable_reason_counts": dict(sorted(reason_counts.items())),
    "compare_type_counts": dict(sorted(compare_counts.items())),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/section_middle/pairs_rejudge.jsonl")
  parser.add_argument(
    "--judgments",
    default="data/section_middle/judgments_rejudge.jsonl",
  )
  parser.add_argument(
    "--out",
    default="data/section_middle/b_rejudge_analysis.json",
  )
  args = parser.parse_args()

  pairs = load_jsonl(args.pairs)
  judgments = load_jsonl(args.judgments)
  report = analyze(pairs, judgments)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
  main()
