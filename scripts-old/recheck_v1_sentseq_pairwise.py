#!/usr/bin/env python3
"""v1 pairwise 268/300 の再確認（学習率 1e-4 ep40 vs 3e-4 ep40）。

集計済み Markdown ではなく、レポート内の候補スコアからペア判定を再計算する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pairwise_events(human_rank: list[str], scores: dict[str, float]) -> list[tuple]:
  events = []
  for i, better in enumerate(human_rank):
    for worse in human_rank[i + 1 :]:
      events.append((better, worse, scores[better] > scores[worse]))
  return events


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--a",
    type=Path,
    default=ROOT / "outputs/hard_eval_v1_report_sentseq40.json",
    help="lr=1e-4 ep40 report",
  )
  parser.add_argument(
    "--b",
    type=Path,
    default=ROOT / "outputs/hard_eval_v1_report_sentseq3e4.json",
    help="lr=3e-4 ep40 report",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=ROOT / "results/raw/v1_sentseq_268_recheck.json",
  )
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()

  if args.out.exists() and not args.overwrite:
    raise SystemExit(f"exists (pass --overwrite): {args.out}")
  for path in (args.a, args.b):
    if not path.is_file():
      raise SystemExit(f"missing: {path}")

  ra = json.loads(args.a.read_text(encoding="utf-8"))
  rb = json.loads(args.b.read_text(encoding="utf-8"))
  by_a = {it["id"]: it for it in ra["items"]}
  by_b = {it["id"]: it for it in rb["items"]}
  if set(by_a) != set(by_b):
    raise SystemExit("item id sets differ")

  agree_a = agree_b = 0
  total = 0
  disagree_pairs = 0
  identical_pair_judgments = True
  top1_a = []
  top1_b = []
  top1_both = []
  top1_a_only = []
  top1_b_only = []

  per_item = []
  for item_id in sorted(by_a):
    ia, ib = by_a[item_id], by_b[item_id]
    if ia["human_rank"] != ib["human_rank"]:
      raise SystemExit(f"{item_id}: human_rank mismatch")
    ea = pairwise_events(ia["human_rank"], ia["scores"])
    eb = pairwise_events(ib["human_rank"], ib["scores"])
    if len(ea) != len(eb):
      raise SystemExit(f"{item_id}: pair count mismatch")
    item_disagree = 0
    for (ba, wa, oka), (bb, wb, okb) in zip(ea, eb):
      assert (ba, wa) == (bb, wb)
      total += 1
      if oka:
        agree_a += 1
      if okb:
        agree_b += 1
      if oka != okb:
        disagree_pairs += 1
        item_disagree += 1
        identical_pair_judgments = False
    if ia["top1_hit"]:
      top1_a.append(item_id)
    if ib["top1_hit"]:
      top1_b.append(item_id)
    if ia["top1_hit"] and ib["top1_hit"]:
      top1_both.append(item_id)
    elif ia["top1_hit"]:
      top1_a_only.append(item_id)
    elif ib["top1_hit"]:
      top1_b_only.append(item_id)
    per_item.append(
      {
        "item_id": item_id,
        "top1_a": ia["top1_hit"],
        "top1_b": ib["top1_hit"],
        "pairwise_a": sum(1 for *_, ok in ea if ok),
        "pairwise_b": sum(1 for *_, ok in eb if ok),
        "pairwise_total": len(ea),
        "pairwise_judgment_disagreements": item_disagree,
      }
    )

  result = {
    "model_a": {
      "report": str(args.a),
      "checkpoint": ra["summary"]["model"],
      "model_config": "lr1e-4-ep40",
      "top1_correct_items": top1_a,
      "top1_hits": len(top1_a),
      "pairwise_agree": agree_a,
      "pairwise_total": total,
    },
    "model_b": {
      "report": str(args.b),
      "checkpoint": rb["summary"]["model"],
      "model_config": "lr3e-4-ep40",
      "top1_correct_items": top1_b,
      "top1_hits": len(top1_b),
      "pairwise_agree": agree_b,
      "pairwise_total": total,
    },
    "both_top1_correct_items": top1_both,
    "a_only_top1_items": top1_a_only,
    "b_only_top1_items": top1_b_only,
    "pairwise_judgment_disagreements": disagree_pairs,
    "identical_pairwise_judgments": identical_pair_judgments,
    "same_pairwise_count_but_different_judgments": (
      agree_a == agree_b and not identical_pair_judgments
    ),
    "per_item": per_item,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
  print(json.dumps({k: v for k, v in result.items() if k != "per_item"}, ensure_ascii=False, indent=2))
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
