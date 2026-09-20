#!/usr/bin/env python3
"""D ペア BT/GPM を難試験 v2b / v2c（24 件）で採点する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from pref_d_pair_score import make_score_fn, resolve_model_dir
from score_hard_eval import score_hard_eval

V2B = "data/hard_eval/bases_v2b_human_fable_copy.jsonl"
V2C = "data/hard_eval/bases_v2c_human_machine_copy.jsonl"


def summarize_dataset(report: dict) -> dict:
  s = report["summary"]
  return {
    "input": s["input"],
    "n_labeled": s["n_labeled"],
    "top1_hits": s["top1_hits"],
    "top1_accuracy": s["top1_accuracy"],
    "human_over_draft_hits": s["human_over_draft_hits"],
    "human_over_mid_hits": s["human_over_mid_hits"],
    "mid_over_draft_hits": s["mid_over_draft_hits"],
    "mean_score_length_spearman": s["mean_score_length_spearman"],
    "pairwise_accuracy": s["pairwise_accuracy"],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cv-dir", type=Path, required=True)
  parser.add_argument("--fold", type=int, default=0)
  parser.add_argument("--v2b-input", type=Path, default=Path(V2B))
  parser.add_argument("--v2c-input", type=Path, default=Path(V2C))
  parser.add_argument("--out", type=Path, default="")
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  args = parser.parse_args()

  if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("--device cuda requested but CUDA is not available")

  model_dir = resolve_model_dir(args.cv_dir, args.fold)
  score_fn, model_meta = make_score_fn(model_dir, device=args.device)
  datasets = [
    ("v2b", args.v2b_input, "fable"),
    ("v2c", args.v2c_input, "machine"),
  ]
  ds_reports: dict[str, dict] = {}
  for ds_name, ds_path, mid_id in datasets:
    print(f"scoring {ds_name} ({ds_path})", flush=True)
    report = score_hard_eval(
      input_path=ds_path,
      score_fn=score_fn,
      scorer=model_meta["mode"],
      model_dir=model_dir,
      mid_id=mid_id,
    )
    ds_reports[ds_name] = report
    s = report["summary"]
    print(
      f"  top1 {s['top1_hits']}/{s['n_labeled']} "
      f"human>copy {s['human_over_draft_hits']}/{s['human_over_draft_total']} "
      f"human>{mid_id} {s['human_over_mid_hits']}/{s['human_over_mid_total']} "
      f"{mid_id}>copy {s['mid_over_draft_hits']}/{s['mid_over_draft_total']}",
      flush=True,
    )

  payload = {
    "cv_dir": str(args.cv_dir),
    "model_dir": str(model_dir),
    "fold": args.fold,
    "v2b_input": str(args.v2b_input),
    "v2c_input": str(args.v2c_input),
    "note": (
      "24 件の段落窓。base_text=下書きを文脈に候補へ点を付ける。"
      "BT は f(x,y)=g(x,y)-g(x,x)。GPM は候補ベクトル間の gpm_score 合計で順位付け。"
    ),
    **model_meta,
    "reference_sentseq_section_triples_v2b": {
      "top1_hits": 17,
      "human_over_draft_hits": 20,
      "human_over_mid_hits": 17,
      "mid_over_draft_hits": 24,
    },
    "reference_sentseq_section_triples_v2c": {
      "top1_hits": 15,
      "human_over_draft_hits": 20,
      "human_over_mid_hits": 15,
      "mid_over_draft_hits": 23,
    },
    "datasets": ds_reports,
    "summary": {
      ds_name: summarize_dataset(report) for ds_name, report in ds_reports.items()
    },
  }
  text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
  else:
    print(text, end="")


if __name__ == "__main__":
  main()
