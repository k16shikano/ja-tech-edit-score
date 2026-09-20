#!/usr/bin/env python3
"""SentSeq + D ModernBERT を下書き零点（Δ = s(x,y) - s(x,x)）で B/C/難試験する。"""
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

from eval_pref_a_bc import (
  B_JUDGMENTS,
  B_PAIRS,
  B_VALID,
  C_JUDGMENTS,
  C_PAIRS,
  COMPARE_B,
  COMPARE_C,
  eval_b_triples,
  eval_blind_b,
  eval_blind_c,
  load_merged_blind,
)
from pref_sentseq_runtime import load_sentseq_model, score_candidates_sentseq_delta
from score_hard_eval import score_hard_eval
from setwise_triple_utils import reconstruct_triples_from_pref_rows
from pref_static_utils import load_jsonl

DEFAULT_MODEL = Path("outputs/pref-sentseq-a-modernbert-d")
V2B = "data/hard_eval/bases_v2b_human_fable_copy.jsonl"
V2C = "data/hard_eval/bases_v2c_human_machine_copy.jsonl"


def make_delta_score_fn(model_dir: Path, *, device: str) -> Callable:
  loaded = load_sentseq_model(model_dir, device=device)

  def score(source: str, candidates: list[str]) -> list[float]:
    return score_candidates_sentseq_delta(loaded, source, candidates)

  return score


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--out-bc", type=Path, default=Path("outputs/pref-a-bc-eval-sentseq-modernbert-d-delta.json"))
  parser.add_argument(
    "--out-hard",
    type=Path,
    default=Path("outputs/pref-a-hard-eval-sentseq-modernbert-d-delta.json"),
  )
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  args = parser.parse_args()

  if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("--device cuda requested but CUDA is not available")

  score_fn = make_delta_score_fn(args.model, device=args.device)
  triples = reconstruct_triples_from_pref_rows(load_jsonl(B_VALID))
  merged_b = load_merged_blind(Path(B_PAIRS), Path(B_JUDGMENTS), COMPARE_B)
  merged_c = load_merged_blind(Path(C_PAIRS), Path(C_JUDGMENTS), COMPARE_C)

  bc_report = {
    "model_dir": str(args.model),
    "kind": "sentseq",
    "calibrate_draft_zero": True,
    "score": "delta(source, candidate) = s(source, candidate) - s(source, source)",
    "b_valid50_triples": eval_b_triples(score_fn, triples),
    "b_blind50_gold_vs_composer": eval_blind_b(score_fn, merged_b),
    "c_blind60_gold_vs_adapter_selected": eval_blind_c(score_fn, merged_c),
  }
  args.out_bc.parent.mkdir(parents=True, exist_ok=True)
  args.out_bc.write_text(json.dumps(bc_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {args.out_bc}", flush=True)

  hard_report = {
    "model_dir": str(args.model),
    "kind": "sentseq",
    "calibrate_draft_zero": True,
    "score": "delta(source, candidate) = s(source, candidate) - s(source, source)",
    "datasets": {
      "v2b": score_hard_eval(
        input_path=Path(V2B),
        score_fn=score_fn,
        scorer="sentseq",
        model_dir=args.model,
        mid_id="fable",
      ),
      "v2c": score_hard_eval(
        input_path=Path(V2C),
        score_fn=score_fn,
        scorer="sentseq",
        model_dir=args.model,
        mid_id="machine",
      ),
    },
  }
  args.out_hard.parent.mkdir(parents=True, exist_ok=True)
  args.out_hard.write_text(json.dumps(hard_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {args.out_hard}", flush=True)

  s = bc_report["b_valid50_triples"]
  print(
    f"B triples: draft>human {s['draft_over_human_n']}/50 "
    f"composer>human {s['composer_over_human_n']}/50",
    flush=True,
  )
  b = bc_report["b_blind50_gold_vs_composer"]
  print(f"B blind agree {b['agree_n']}/{b['comparable_n']}", flush=True)
  c = bc_report["c_blind60_gold_vs_adapter_selected"]
  print(f"C blind agree {c['agree_n']}/{c['comparable_n']}", flush=True)
  for ds_name in ("v2b", "v2c"):
    hs = hard_report["datasets"][ds_name]["summary"]
    print(
      f"hard {ds_name}: top1 {hs['top1_hits']}/24 "
      f"human>copy {hs['human_over_draft_hits']}/24 "
      f"human>{hs['mid_id']} {hs['human_over_mid_hits']}/24",
      flush=True,
    )


if __name__ == "__main__":
  main()
