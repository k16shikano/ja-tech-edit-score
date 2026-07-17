#!/usr/bin/env python3
"""Hard Eval レポートから候補間のスコア差（マージン）を集計する。

v2b（human/fable/copy）向け: 「どっちがよい」だけでなく「どの程度」を見る。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mean(xs: list[float]) -> float:
  return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
  if not xs:
    return float("nan")
  ys = sorted(xs)
  n = len(ys)
  mid = n // 2
  if n % 2:
    return ys[mid]
  return 0.5 * (ys[mid - 1] + ys[mid])


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--report", required=True, help="score_hard_eval の JSON レポート")
  parser.add_argument(
    "--pairs",
    default="human,fable;fable,copy;human,copy",
    help="比較ペア A,B;...（A-B のマージンを集計）",
  )
  parser.add_argument("--out", help="マージン分析 JSON の出力先（省略時は stdout 要約のみ）")
  args = parser.parse_args()

  report = json.loads(Path(args.report).read_text(encoding="utf-8"))
  items = report["items"]
  pairs = []
  for chunk in args.pairs.split(";"):
    a, b = [x.strip() for x in chunk.split(",")]
    pairs.append((a, b))

  pair_stats: dict[str, dict] = {}
  for a, b in pairs:
    margins = []
    wins = 0
    for it in items:
      scores = it["scores"]
      if a not in scores or b not in scores:
        continue
      m = scores[a] - scores[b]
      margins.append(m)
      if m > 0:
        wins += 1
    key = f"{a}-{b}"
    pair_stats[key] = {
      "n": len(margins),
      "win_rate": wins / len(margins) if margins else float("nan"),
      "mean_margin": mean(margins),
      "median_margin": median(margins),
      "min_margin": min(margins) if margins else float("nan"),
      "max_margin": max(margins) if margins else float("nan"),
      "margins": margins,
    }

  # 項目ごとのスコアと正規化位置（copy=0, human=1 のとき fable はどこか）
  # human≈copy のときは分母が潰れるので位置は出さない。
  positions = []
  per_item = []
  for it in items:
    s = it["scores"]
    if not all(k in s for k in ("human", "fable", "copy")):
      continue
    span = s["human"] - s["copy"]
    if span > 1e-3:
      pos = (s["fable"] - s["copy"]) / span
    else:
      pos = float("nan")
    if not math.isnan(pos):
      positions.append(pos)
    per_item.append(
      {
        "id": it["id"],
        "scores": {k: s[k] for k in ("human", "fable", "copy")},
        "margin_human_fable": s["human"] - s["fable"],
        "margin_fable_copy": s["fable"] - s["copy"],
        "margin_human_copy": s["human"] - s["copy"],
        "fable_position": pos,
        "model_best": it.get("best_model"),
        "order": sorted(("human", "fable", "copy"), key=lambda k: -s[k]),
      }
    )

  finite_pos = [p for p in positions if not math.isnan(p)]
  summary = {
    "report": str(args.report),
    "model": report.get("summary", {}).get("model"),
    "n": len(per_item),
    "pair_stats": {
      k: {kk: vv for kk, vv in v.items() if kk != "margins"}
      for k, v in pair_stats.items()
    },
    "fable_position": {
      "mean": mean(finite_pos),
      "median": median(finite_pos),
      "between_0_1": sum(1 for p in finite_pos if 0 < p < 1) / len(finite_pos)
      if finite_pos
      else float("nan"),
      "note": "0=copy, 1=human。0〜1 なら human>fable>copy の間にいる",
    },
    "items": per_item,
  }

  print(f"model: {summary['model']}")
  print(f"n: {summary['n']}")
  for k, v in summary["pair_stats"].items():
    print(
      f"  {k}: win={v['win_rate']:.3f}  "
      f"meanΔ={v['mean_margin']:.2f}  medianΔ={v['median_margin']:.2f}  "
      f"range=[{v['min_margin']:.2f}, {v['max_margin']:.2f}]"
    )
  fp = summary["fable_position"]
  print(
    f"  fable_position (0=copy,1=human): "
    f"mean={fp['mean']:.3f} median={fp['median']:.3f} "
    f"in(0,1)={fp['between_0_1']:.3f}"
  )

  if args.out:
    Path(args.out).write_text(
      json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
