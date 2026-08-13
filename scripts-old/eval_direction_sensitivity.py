#!/usr/bin/env python3
"""方向感度の検証: 順方向・逆方向・劣化版の self 基準マージンを測る。

検証セットの実ペア（下書き、人間編集）に対して、任意の採点モデルで
次の 3 つを測る。マージンはすべて margin = s(基準, 候補) - s(基準, 基準)。

- 順方向: 基準=下書き、候補=人間編集。正が正解。
- 逆方向: 基準=人間編集、候補=下書き。負が正解。
- 劣化版: 基準=下書き、候補=冗長な言い回しを機械挿入した版。負が正解。

劣化版は診断用のプローブであり、学習・評価の能力指標にはしない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pref_scorer import load_scorer
from pref_static_utils import load_jsonl
from train_pref_bt import unique_preference_pairs


def degrade(text: str) -> str:
  t = text.replace("。", "ということになります。", 3)
  t = t.replace("、", "、まあ、", 2)
  return t


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, help="採点モデルのディレクトリ")
  parser.add_argument("--pairs", default="data/pref_split/valid.jsonl")
  parser.add_argument("--n-degrade", type=int, default=60)
  parser.add_argument("--limit", type=int, default=0, help="ペア数の上限（0で全件）")
  parser.add_argument("--report", default="", help="JSON 出力先（省略時は表示のみ）")
  args = parser.parse_args()

  rows = unique_preference_pairs(load_jsonl(args.pairs))
  if args.limit > 0:
    rows = rows[: args.limit]
  scorer = load_scorer(Path(args.model))

  def self_margin(base: str, cand: str) -> float:
    s = scorer.score(base, [cand, base], batch_size=4)
    return float(s[0] - s[1])

  fwd, rev, deg = [], [], []
  for i, r in enumerate(rows):
    draft, edit = r["candidate_b"], r["candidate_a"]
    fwd.append(self_margin(draft, edit))
    rev.append(self_margin(edit, draft))
    if i < args.n_degrade:
      bad = degrade(draft)
      if bad != draft:
        deg.append(self_margin(draft, bad))

  f, v, g = np.array(fwd), np.array(rev), np.array(deg)
  report = {
    "model": args.model,
    "scorer": scorer.kind,
    "n_pairs": len(rows),
    "forward": {"positive_rate": float((f > 0).mean()), "median": float(np.median(f))},
    "reverse": {"negative_rate": float((v < 0).mean()), "median": float(np.median(v))},
    "degrade": {
      "negative_rate": float((g < 0).mean()),
      "median": float(np.median(g)),
      "n": int(g.size),
    },
  }
  print(json.dumps(report, ensure_ascii=False, indent=2))
  if args.report:
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
