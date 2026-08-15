#!/usr/bin/env python3
"""B の検証 50 件について、人間の推敲対 Composer の人手選択と評価器の点の上下の一致を集計する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import human_pref, merge_rows
from eval_pref_multigranular_blind60 import score_blind_rows, scorer_pref_from_scores
from pref_scorer import detect_scorer_kind, load_scorer
from pref_static_utils import load_jsonl

COMPARE_TYPE = "gold_vs_composer"
DEFAULT_MODELS = (
  "outputs/pref-detect-section",
  "outputs/pref-detect-cd-section",
  "outputs/pref-nce-section",
  "outputs/pref-sentseq-section-triples",
)


def gold_scores_higher(row: dict, score_a: float, score_b: float) -> bool:
  a_src = str(row.get("a_source") or "")
  b_src = str(row.get("b_source") or "")
  if a_src == "gold" and b_src == "composer":
    return score_a > score_b
  if b_src == "gold" and a_src == "composer":
    return score_b > score_a
  raise ValueError(
    f"expected gold vs composer for pair {row.get('pair_id')!r}: "
    f"a_source={a_src!r} b_source={b_src!r}"
  )


def human_picked_source(row: dict) -> str | None:
  hp = human_pref(row)
  if hp == "a":
    return str(row.get("a_source") or "")
  if hp == "b":
    return str(row.get("b_source") or "")
  return None


def agreement_for_pairs(rows: list[dict]) -> dict:
  n_total = len(rows)
  human_tie = 0
  human_gold = 0
  human_composer = 0
  scorer_tie = 0
  comparable_n = 0
  agree_n = 0
  gold_higher_n = 0
  agree_when_human_gold = 0
  agree_when_human_composer = 0
  items: list[dict] = []

  for row in rows:
    score_a = float(row["score_a"])
    score_b = float(row["score_b"])
    hp = human_pref(row)
    sp = scorer_pref_from_scores(score_a, score_b)
    picked = human_picked_source(row)
    gold_higher = gold_scores_higher(row, score_a, score_b)
    if gold_higher:
      gold_higher_n += 1

    if row.get("choice") == "tie":
      human_tie += 1
    if picked == "gold":
      human_gold += 1
    elif picked == "composer":
      human_composer += 1
    if hp is not None and sp is None:
      scorer_tie += 1

    agree: bool | None = None
    if hp is not None and sp is not None:
      comparable_n += 1
      agree = hp == sp
      if agree:
        agree_n += 1
        if picked == "gold":
          agree_when_human_gold += 1
        elif picked == "composer":
          agree_when_human_composer += 1

    items.append(
      {
        "pair_id": row["pair_id"],
        "item_id": row.get("item_id"),
        "human_pref": hp,
        "human_source": picked,
        "scorer_pref": sp,
        "score_a": score_a,
        "score_b": score_b,
        "gold_higher": gold_higher,
        "agree": agree,
        "a_broken": bool(row.get("a_broken")),
        "b_broken": bool(row.get("b_broken")),
      }
    )

  return {
    "n_total": n_total,
    "human_tie": human_tie,
    "human_gold": human_gold,
    "human_composer": human_composer,
    "scorer_tie": scorer_tie,
    "comparable_n": comparable_n,
    "agree_n": agree_n,
    "agree_when_human_gold": agree_when_human_gold,
    "agree_when_human_composer": agree_when_human_composer,
    "gold_higher_n": gold_higher_n,
    "items": items,
  }


def evaluate_model(model_dir: Path, rows: list[dict]) -> dict:
  kind = detect_scorer_kind(model_dir)
  scorer = load_scorer(model_dir)

  def score_fn(source: str, candidates: list[str]) -> list[float]:
    return scorer.score(source, candidates, batch_size=4)

  scored = score_blind_rows(rows, score_fn)
  stats = agreement_for_pairs(scored)
  return {
    "model": str(model_dir),
    "kind": kind,
    **{k: stats[k] for k in stats if k != "items"},
    "items": stats["items"],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--pairs",
    default="data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl",
  )
  parser.add_argument(
    "--judgments",
    default="data/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl",
  )
  parser.add_argument(
    "--out",
    default="outputs/blind50_pref_valid_gold_vs_composer.json",
  )
  parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
  args = parser.parse_args()

  pairs = load_jsonl(args.pairs)
  judgments = load_jsonl(args.judgments)
  compare_pairs = [p for p in pairs if p.get("compare_type") == COMPARE_TYPE]
  merged = merge_rows(compare_pairs, judgments)
  if len(merged) != 50:
    raise SystemExit(f"expected 50 merged rows, got {len(merged)}")

  runs = []
  for rel in args.models:
    model_dir = Path(rel)
    print(f"scoring {model_dir} kind={detect_scorer_kind(model_dir)}", flush=True)
    run = evaluate_model(model_dir, merged)
    print(
      f"  agree {run['agree_n']}/{run['comparable_n']} "
      f"human_gold={run['human_gold']} human_composer={run['human_composer']} "
      f"tie={run['human_tie']} gold_higher={run['gold_higher_n']}",
      flush=True,
    )
    runs.append(run)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "compare_type": COMPARE_TYPE,
    "pairs": args.pairs,
    "judgments": args.judgments,
    "n_pairs": len(merged),
    "runs": runs,
  }
  out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {out}", flush=True)


if __name__ == "__main__":
  main()
