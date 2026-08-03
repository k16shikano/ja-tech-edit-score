#!/usr/bin/env python3
"""合格閾値の較正: 人間編集が下書きから稼ぐマージンの分布を測る。

検証セットの「下書き → 人間編集」ペアを主モデル（pref-sentseq-keep）とゲート
（pref-bt-keep）で採点し、2 種類のマージン分布（パーセンタイル）を出す。

- pair 基準: s(source_text, 人間編集) - s(source_text, 下書き)。
  学習・検証と同じ形（共通の基準に対する 2 候補の差）。
- self 基準: s(下書き, 人間編集) - s(下書き, 下書き)。
  運用時（revise ループ・採点 Web サービス）の定義。基準と候補が同一という
  学習にない入力を含むため、「変更しただけで加点」の偏りが乗る。閾値は
  この偏りを超える水準に置く必要がある。

下書きには候補欄の棄却側（candidate_b、クリーンな下書きテキスト）を使う。
source_text には参照情報の後注が付いており、運用時のテキストと一致しないため。
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


def distribution(margins: list[float]) -> dict:
  arr = np.asarray(margins, dtype=np.float64)
  return {
    "n_pairs": int(arr.size),
    "positive_rate": float((arr > 0).mean()),
    "mean": float(arr.mean()),
    "percentiles": {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES},
  }


def margins_for_model(model_dir: Path, rows: list[dict]) -> dict:
  scorer = load_scorer(model_dir)
  pair_margins: list[float] = []
  self_margins: list[float] = []
  for row in rows:
    source = row["source_text"]
    edit = row["candidate_a"]
    draft = row["candidate_b"]
    s_edit, s_draft = scorer.score(source, [edit, draft], batch_size=4)
    pair_margins.append(float(s_edit - s_draft))
    s_edit_self, s_draft_self = scorer.score(draft, [edit, draft], batch_size=4)
    self_margins.append(float(s_edit_self - s_draft_self))
  return {
    "scorer": scorer.kind,
    "model_dir": str(model_dir),
    **distribution(pair_margins),
    "self_baseline": distribution(self_margins),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/pref_keep_split_section/valid.jsonl")
  parser.add_argument("--primary-model", default="outputs/pref-sentseq-keep")
  parser.add_argument("--gate-model", default="outputs/pref-bt-keep")
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
