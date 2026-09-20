#!/usr/bin/env python3
"""データ C（人間の推敲 vs 選抜生成 60 件）で D ペア BT/GPM と人手優劣の一致を集計する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from eval_pref_c_human_agreement import load_merged
from eval_pref_multigranular_blind60 import agreement_for_pairs, score_blind_rows
from pref_d_pair_score import make_score_fn, resolve_model_dir

DEFAULT_PAIRS = "data/blind_eval/pairs_gold_vs_adapter_selected.jsonl"
DEFAULT_JUDGMENTS = "data/blind_eval/judgments_gold_vs_adapter_selected.jsonl"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cv-dir", type=Path, required=True)
  parser.add_argument("--fold", type=int, default=0)
  parser.add_argument("--pairs", type=Path, default=Path(DEFAULT_PAIRS))
  parser.add_argument("--judgments", type=Path, default=Path(DEFAULT_JUDGMENTS))
  parser.add_argument("--out", type=Path, default="")
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  args = parser.parse_args()

  if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("--device cuda requested but CUDA is not available")

  model_dir = resolve_model_dir(args.cv_dir, args.fold)
  merged = load_merged(args.pairs, args.judgments)
  score_fn, model_meta = make_score_fn(model_dir, device=args.device)
  scored = score_blind_rows(merged, score_fn)
  stats = agreement_for_pairs(scored)
  report = {
    "cv_dir": str(args.cv_dir),
    "model_dir": str(model_dir),
    "fold": args.fold,
    "pairs": str(args.pairs),
    "judgments": str(args.judgments),
    "compare_type": "5_gold_vs_adapter_selected",
    **model_meta,
    "human_agreement": {k: stats[k] for k in stats if k != "items"},
  }
  text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
  print(text, end="")


if __name__ == "__main__":
  main()
