#!/usr/bin/env python3
"""データ A で学習した BT / SentSeq（3 系統エンコーダ）を B 検証 50 と C 60 で評価する。"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import merge_rows
from eval_pref_multigranular import _strictly_greater
from eval_pref_multigranular_blind60 import agreement_for_pairs as agreement_c
from eval_pref_multigranular_blind60 import score_blind_rows
from eval_pref_valid50_gold_vs_composer import agreement_for_pairs as agreement_b
from pref_bt_runtime import load_bt_model, score_candidates_bt
from pref_sentseq_runtime import load_sentseq_model, score_candidates_sentseq
from pref_static_utils import load_jsonl
from setwise_triple_utils import reconstruct_triples_from_pref_rows

B_VALID = "data/section_middle/pref_valid.jsonl"
B_PAIRS = "data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl"
B_JUDGMENTS = "data/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl"
C_PAIRS = "data/blind_eval/pairs_gold_vs_adapter_selected.jsonl"
C_JUDGMENTS = "data/blind_eval/judgments_gold_vs_adapter_selected.jsonl"
COMPARE_B = "gold_vs_composer"
COMPARE_C = "5_gold_vs_adapter_selected"


def make_score_fn(model_dir: Path, kind: str, *, device: str) -> Callable:
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


def eval_b_triples(score_fn: Callable, triples: list) -> dict:
  rows: list[dict] = []
  draft_over_human = 0
  composer_over_human = 0
  for triple in triples:
    scores = score_fn(triple.source_text, [triple.human, triple.composer, triple.draft])
    human, composer, draft = (float(v) for v in scores)
    human_over_draft = _strictly_greater(human, draft)
    human_over_composer = _strictly_greater(human, composer)
    if _strictly_greater(draft, human):
      draft_over_human += 1
    if _strictly_greater(composer, human):
      composer_over_human += 1
    rows.append(
      {
        "item_id": triple.item_id,
        "scores": {"human": human, "composer": composer, "draft": draft},
        "human_over_draft": human_over_draft,
        "human_over_composer": human_over_composer,
      }
    )
  n = len(rows)
  return {
    "n": n,
    "draft_over_human_n": draft_over_human,
    "composer_over_human_n": composer_over_human,
    "human_over_draft_n": sum(1 for r in rows if r["human_over_draft"]),
    "human_over_composer_n": sum(1 for r in rows if r["human_over_composer"]),
    "items": rows,
  }


def eval_blind_b(score_fn: Callable, merged: list[dict]) -> dict:
  scored = score_blind_rows(merged, score_fn)
  stats = agreement_b(scored)
  return {k: stats[k] for k in stats if k != "items"}


def eval_blind_c(score_fn: Callable, merged: list[dict]) -> dict:
  scored = score_blind_rows(merged, score_fn)
  stats = agreement_c(scored)
  return {k: stats[k] for k in stats if k != "items"}


def load_merged_blind(pairs_path: Path, judgments_path: Path, compare_type: str) -> list[dict]:
  pairs = [p for p in load_jsonl(str(pairs_path)) if p.get("compare_type") == compare_type]
  judgments = load_jsonl(str(judgments_path))
  merged = merge_rows(pairs, judgments)
  if len(merged) != len(pairs):
    raise SystemExit(
      f"merge mismatch for {compare_type}: pairs={len(pairs)} merged={len(merged)}"
    )
  return merged


def model_specs(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
  return [
    ("bt_ruri_a", args.bt_ruri, "bt"),
    ("bt_modernbert_a", args.bt_modernbert, "bt"),
    ("bt_modernbert_d_a", args.bt_modernbert_d, "bt"),
    ("sentseq_ruri_a", args.sentseq_ruri, "sentseq"),
    ("sentseq_modernbert_a", args.sentseq_modernbert, "sentseq"),
    ("sentseq_modernbert_d_a", args.sentseq_modernbert_d, "sentseq"),
  ]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--b-valid-file", type=Path, default=Path(B_VALID))
  parser.add_argument("--b-pairs", type=Path, default=Path(B_PAIRS))
  parser.add_argument("--b-judgments", type=Path, default=Path(B_JUDGMENTS))
  parser.add_argument("--c-pairs", type=Path, default=Path(C_PAIRS))
  parser.add_argument("--c-judgments", type=Path, default=Path(C_JUDGMENTS))
  parser.add_argument("--out", type=Path, default=Path("outputs/pref-a-bc-eval.json"))
  parser.add_argument("--bt-ruri", type=Path, default=Path("outputs/pref-bt-a-ruri"))
  parser.add_argument("--bt-modernbert", type=Path, default=Path("outputs/pref-bt-a-modernbert"))
  parser.add_argument("--bt-modernbert-d", type=Path, default=Path("outputs/pref-bt-a-modernbert-d"))
  parser.add_argument("--sentseq-ruri", type=Path, default=Path("outputs/pref-sentseq-a-ruri"))
  parser.add_argument(
    "--sentseq-modernbert", type=Path, default=Path("outputs/pref-sentseq-a-modernbert")
  )
  parser.add_argument(
    "--sentseq-modernbert-d", type=Path, default=Path("outputs/pref-sentseq-a-modernbert-d")
  )
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  args = parser.parse_args()

  if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("--device cuda requested but CUDA is not available")

  triples = reconstruct_triples_from_pref_rows(load_jsonl(str(args.b_valid_file)))
  merged_b = load_merged_blind(args.b_pairs, args.b_judgments, COMPARE_B)
  merged_c = load_merged_blind(args.c_pairs, args.c_judgments, COMPARE_C)

  models: dict[str, dict] = {}
  for name, model_dir, kind in model_specs(args):
    print(f"evaluating {name} ({kind}) from {model_dir}", flush=True)
    score_fn = make_score_fn(model_dir, kind, device=args.device)
    b_triples = eval_b_triples(score_fn, triples)
    blind_b = eval_blind_b(score_fn, merged_b)
    blind_c = eval_blind_c(score_fn, merged_c)
    models[name] = {
      "model_dir": str(model_dir),
      "kind": kind,
      "b_valid50_triples": {
        k: b_triples[k]
        for k in (
          "n",
          "draft_over_human_n",
          "composer_over_human_n",
          "human_over_draft_n",
          "human_over_composer_n",
        )
      },
      "b_blind50_gold_vs_composer": blind_b,
      "c_blind60_gold_vs_adapter_selected": blind_c,
    }
    print(
      f"  B triples: draft>human {b_triples['draft_over_human_n']}/50 "
      f"composer>human {b_triples['composer_over_human_n']}/50",
      flush=True,
    )
    print(
      f"  B blind agree {blind_b['agree_n']}/{blind_b['comparable_n']} "
      f"gold_higher={blind_b['gold_higher_n']}",
      flush=True,
    )
    print(
      f"  C blind agree {blind_c['agree_n']}/{blind_c['comparable_n']} "
      f"gold_higher={blind_c['gold_higher_n']}",
      flush=True,
    )

  report = {
    "b_valid_file": str(args.b_valid_file),
    "b_pairs": str(args.b_pairs),
    "b_judgments": str(args.b_judgments),
    "c_pairs": str(args.c_pairs),
    "c_judgments": str(args.c_judgments),
    "human_baseline_b_blind": {
      "human_gold": 43,
      "human_composer": 4,
      "human_tie": 3,
      "n": 50,
    },
    "reference_pref_sentseq_section_triples_b_valid50": {
      "draft_over_human_n": 0,
      "composer_over_human_n": 3,
    },
    "models": models,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
  main()
