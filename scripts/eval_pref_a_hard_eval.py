#!/usr/bin/env python3
"""データ A で学習した 6 モデルを難試験 v2b / v2c で採点する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from score_hard_eval import make_score_fn, score_hard_eval

V2B = "data/hard_eval/bases_v2b_human_fable_copy.jsonl"
V2C = "data/hard_eval/bases_v2c_human_machine_copy.jsonl"


def model_specs(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
  return [
    ("bt_ruri_a", args.bt_ruri, "bt"),
    ("bt_modernbert_a", args.bt_modernbert, "bt"),
    ("bt_modernbert_d_a", args.bt_modernbert_d, "bt"),
    ("sentseq_ruri_a", args.sentseq_ruri, "sentseq"),
    ("sentseq_modernbert_a", args.sentseq_modernbert, "sentseq"),
    ("sentseq_modernbert_d_a", args.sentseq_modernbert_d, "sentseq"),
  ]


def summarize_run(name: str, report: dict) -> dict:
  s = report["summary"]
  return {
    "name": name,
    "scorer": s["scorer"],
    "model": s["model"],
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
  parser.add_argument("--v2b-input", type=Path, default=Path(V2B))
  parser.add_argument("--v2c-input", type=Path, default=Path(V2C))
  parser.add_argument("--out", type=Path, default=Path("outputs/pref-a-hard-eval.json"))
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

  datasets = [
    ("v2b", args.v2b_input, "fable"),
    ("v2c", args.v2c_input, "machine"),
  ]
  models_out: dict[str, dict] = {}
  for name, model_dir, kind in model_specs(args):
    print(f"scoring {name} ({kind})", flush=True)
    score_fn = make_score_fn(model_dir, kind, device=args.device)
    entry: dict = {"kind": kind, "model_dir": str(model_dir), "datasets": {}}
    for ds_name, ds_path, mid_id in datasets:
      report = score_hard_eval(
        input_path=ds_path,
        score_fn=score_fn,
        scorer=kind,
        model_dir=model_dir,
        mid_id=mid_id,
      )
      entry["datasets"][ds_name] = report
      s = report["summary"]
      print(
        f"  {ds_name}: top1 {s['top1_hits']}/{s['n_labeled']} "
        f"human>copy {s['human_over_draft_hits']}/{s['human_over_draft_total']} "
        f"human>{mid_id} {s['human_over_mid_hits']}/{s['human_over_mid_total']} "
        f"{mid_id}>copy {s['mid_over_draft_hits']}/{s['mid_over_draft_total']}",
        flush=True,
      )
    models_out[name] = entry

  payload = {
    "v2b_input": str(args.v2b_input),
    "v2c_input": str(args.v2c_input),
    "note": (
      "24 件の段落窓。base_text=下書きを文脈に候補へ点を付ける。"
      "v2b は human/fable/copy、v2c は human/machine/copy。"
    ),
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
    "models": models_out,
    "table": [
      summarize_run(name, models_out[name]["datasets"]["v2b"])
      for name in models_out
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
  main()
