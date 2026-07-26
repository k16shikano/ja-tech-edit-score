#!/usr/bin/env python3
"""合格閾値の較正: 人間編集が下書きから稼ぐマージンの分布を測る。

検証セットの「下書き → 人間編集」ペアを主モデル（pref-sentseq）とゲート
（pref-bt）で採点し、margin = s(source, 人間編集) - s(source, 下書き) の分布
（パーセンタイル）を出す。この分布との比較で「人間の編集に十分近いか」を
判定する閾値を選ぶ。

下書きには候補欄の棄却側（candidate_b、クリーンな下書きテキスト）を使う。
source_text には参照情報の後注が付いており、運用時（make rank に渡す
クリーンな下書き）のマージンと一致しないため。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pref_scorer import load_scorer
from pref_static_utils import load_jsonl


PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


def human_edit_rows(path: str) -> list[dict]:
  rows = []
  for row in load_jsonl(path):
    meta = row.get("meta", {})
    if meta.get("pair_order", "chosen_first") != "chosen_first":
      continue
    if int(row["label"]) != 1:
      continue
    rows.append(row)
  return rows


def margins_for_model(model_dir: Path, rows: list[dict]) -> dict:
  scorer = load_scorer(model_dir)
  margins: list[float] = []
  for row in rows:
    source = row["source_text"]
    edit = row["candidate_a"]
    draft = row["candidate_b"]
    s_edit, s_draft = scorer.score(source, [edit, draft], batch_size=4)
    margins.append(float(s_edit - s_draft))
  arr = np.asarray(margins, dtype=np.float64)
  return {
    "scorer": scorer.kind,
    "model_dir": str(model_dir),
    "n_pairs": int(arr.size),
    "positive_rate": float((arr > 0).mean()),
    "mean": float(arr.mean()),
    "percentiles": {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES},
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/pref_split/valid.jsonl")
  parser.add_argument("--primary-model", default="outputs/pref-sentseq-3e4")
  parser.add_argument("--gate-model", default="outputs/pref-bt")
  parser.add_argument("--out", default="outputs/acceptance_margin_calibration.json")
  args = parser.parse_args()

  rows = human_edit_rows(args.pairs)
  if not rows:
    raise SystemExit("no human-edit pairs found")

  report = {
    "pairs_file": args.pairs,
    "n_pairs": len(rows),
    "primary": margins_for_model(Path(args.primary_model), rows),
    "gate": margins_for_model(Path(args.gate_model), rows),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
