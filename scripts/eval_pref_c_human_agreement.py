#!/usr/bin/env python3
"""データ C（人間の推敲 vs 選抜生成 60 件）で、評価器と人手優劣の一致を集計する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import merge_rows
from eval_pref_multigranular_blind60 import agreement_for_pairs, score_blind_rows
from pref_bt_runtime import load_bt_model, score_candidates_bt
from pref_sentseq_runtime import load_sentseq_model, score_candidates_sentseq
from pref_static_utils import load_jsonl

COMPARE_TYPE = "5_gold_vs_adapter_selected"
DEFAULT_PAIRS = "data/blind_eval/pairs_gold_vs_adapter_selected.jsonl"
DEFAULT_JUDGMENTS = "data/blind_eval/judgments_gold_vs_adapter_selected.jsonl"


def make_score_fn(model_dir: Path, kind: str, *, device: str):
  if kind == "bt":
    loaded = load_bt_model(model_dir, device=device)

    def score(source: str, candidates: list[str]) -> list[float]:
      return score_candidates_bt(loaded, source, candidates)

    return score

  if kind == "sentseq":
    loaded = load_sentseq_model(model_dir, device=device)

    def score(source: str, candidates: list[str]) -> list[float]:
      return score_candidates_sentseq(loaded, source, candidates)

    return score

  raise ValueError(kind)


def load_merged(pairs_path: Path, judgments_path: Path) -> list[dict]:
  pairs = [p for p in load_jsonl(str(pairs_path)) if p.get("compare_type") == COMPARE_TYPE]
  judgments = load_jsonl(str(judgments_path))
  merged = merge_rows(pairs, judgments)
  if len(merged) != len(pairs):
    raise SystemExit(f"merge mismatch: pairs={len(pairs)} merged={len(merged)}")
  return merged


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, required=True)
  parser.add_argument("--kind", choices=["bt", "sentseq"], default="sentseq")
  parser.add_argument("--pairs", type=Path, default=Path(DEFAULT_PAIRS))
  parser.add_argument("--judgments", type=Path, default=Path(DEFAULT_JUDGMENTS))
  parser.add_argument("--out", type=Path, default="")
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  args = parser.parse_args()

  if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("--device cuda requested but CUDA is not available")

  merged = load_merged(args.pairs, args.judgments)
  score_fn = make_score_fn(args.model_dir, args.kind, device=args.device)
  scored = score_blind_rows(merged, score_fn)
  stats = agreement_for_pairs(scored)
  report = {
    "model_dir": str(args.model_dir),
    "kind": args.kind,
    "pairs": str(args.pairs),
    "judgments": str(args.judgments),
    "compare_type": COMPARE_TYPE,
    "human_agreement": {k: stats[k] for k in stats if k != "items"},
  }
  text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
  print(text, end="")


if __name__ == "__main__":
  main()
