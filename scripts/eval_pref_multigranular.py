#!/usr/bin/env python3
"""検証 50 件の三つ組みを、段階ごとの評価器で採点して集計する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pref_multigranular_runtime import (
  load_multigranular_model,
  score_candidates_multigranular,
)
from pref_scorer import load_scorer
from pref_static_utils import load_jsonl
from setwise_model import STRICT_COMPARE_EPS, pair_checkpoint_selection_key
from setwise_triple_utils import reconstruct_triples_from_pref_rows


def _strictly_greater(a: float, b: float) -> bool:
  return a > b + STRICT_COMPARE_EPS


def _summarize(rows: list[dict]) -> dict[str, float]:
  n = len(rows)
  if n == 0:
    raise ValueError("no rows to summarize")
  human_over_composer = sum(1 for r in rows if r["human_over_composer"]) / n
  human_over_draft = sum(1 for r in rows if r["human_over_draft"]) / n
  human_among_top = sum(1 for r in rows if r["human_among_top"]) / n
  human_top1 = sum(1 for r in rows if r["human_top1"]) / n
  return {
    "n": float(n),
    "human_among_top": human_among_top,
    "human_top1": human_top1,
    "human_over_composer": human_over_composer,
    "human_over_draft": human_over_draft,
    "min_human_over_pairs": min(human_over_draft, human_over_composer),
  }


def _score_triple(score_fn, triple) -> dict:
  names = ["human", "composer", "draft"]
  texts = [triple.human, triple.composer, triple.draft]
  scores = score_fn(triple.source_text, texts)
  by_name = {name: float(scores[i]) for i, name in enumerate(names)}
  human = by_name["human"]
  composer = by_name["composer"]
  draft = by_name["draft"]
  others = [composer, draft]
  human_top1 = all(_strictly_greater(human, other) for other in others)
  human_among_top = not any(_strictly_greater(other, human) for other in others)
  return {
    "item_id": triple.item_id,
    "scores": by_name,
    "human_over_composer": _strictly_greater(human, composer),
    "human_over_draft": _strictly_greater(human, draft),
    "human_among_top": human_among_top,
    "human_top1": human_top1,
  }


def _sentseq_score_fn(model_dir: Path):
  scorer = load_scorer(model_dir)

  def score(source: str, candidates: list[str]) -> list[float]:
    raw = scorer.score(source, candidates + [source], batch_size=4)
    self_s = raw[-1]
    return [float(v - self_s) for v in raw[:-1]]

  return score


def _multigranular_score_fn(model_dir: Path):
  loaded = load_multigranular_model(model_dir)

  def score(source: str, candidates: list[str]) -> list[float]:
    return score_candidates_multigranular(loaded, source, candidates)["logits"]

  return score


def _gated_score_fn(pair_model_dir: Path, gate_model_dir: Path, gate_min_margin: float):
  pair_score = _multigranular_score_fn(pair_model_dir)
  gate = load_scorer(gate_model_dir)

  def score(source: str, candidates: list[str]) -> list[float]:
    pair_logits = pair_score(source, candidates)
    gate_raw = gate.score(source, candidates + [source], batch_size=4)
    self_g = gate_raw[-1]
    out: list[float] = []
    for i, cand_score in enumerate(pair_logits):
      margin = float(gate_raw[i] - self_g)
      if margin < gate_min_margin:
        out.append(float("-inf"))
      else:
        out.append(cand_score)
    return out

  return score


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--eval-file", default="data/section_middle/pref_valid.jsonl")
  parser.add_argument("--kind", required=True, choices=["sentseq", "multigranular", "gated"])
  parser.add_argument("--model", required=True)
  parser.add_argument("--gate-model", default="outputs/pref-bt-keep")
  parser.add_argument("--gate-min-margin", type=float, default=0.0)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  triples = reconstruct_triples_from_pref_rows(load_jsonl(args.eval_file))
  if args.kind == "sentseq":
    score_fn = _sentseq_score_fn(Path(args.model))
  elif args.kind == "multigranular":
    score_fn = _multigranular_score_fn(Path(args.model))
  else:
    score_fn = _gated_score_fn(
      Path(args.model),
      Path(args.gate_model),
      args.gate_min_margin,
    )

  rows = [_score_triple(score_fn, triple) for triple in triples]
  summary = _summarize(rows)
  summary["checkpoint_key"] = list(pair_checkpoint_selection_key(summary))
  payload = {"summary": summary, "items": rows}
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
  main()
