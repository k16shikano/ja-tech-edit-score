#!/usr/bin/env python3
"""データ A 検証 260 ペアで、3 系統エンコーダ × BT / SentSeq を比較する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pref_bt_runtime import load_bt_model, score_candidates_bt
from pref_sentseq_runtime import load_sentseq_model, score_candidates_sentseq
from pref_static_utils import load_jsonl
from train_pref_bt import unique_preference_pairs


def eval_pairwise(
  model_dir: Path,
  rows: list[dict],
  *,
  kind: str,
  device: str,
) -> dict:
  draft_over = 0
  if kind == "bt":
    loaded = load_bt_model(model_dir, device=device)
    for row in rows:
      sh, sd = score_candidates_bt(
        loaded, row["source_text"], [row["candidate_a"], row["candidate_b"]]
      )
      if sd > sh:
        draft_over += 1
  elif kind == "sentseq":
    loaded = load_sentseq_model(model_dir, device=device)
    for row in rows:
      sh, sd = score_candidates_sentseq(
        loaded, row["source_text"], [row["candidate_a"], row["candidate_b"]]
      )
      if sd > sh:
        draft_over += 1
  else:
    raise ValueError(kind)
  n = len(rows)
  return {
    "pair_accuracy": (n - draft_over) / n,
    "draft_over_human_n": draft_over,
    "n": n,
  }


def load_metrics(model_dir: Path) -> dict:
  path = model_dir / "metrics.json"
  if path.is_file():
    return json.loads(path.read_text(encoding="utf-8"))
  return {}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--valid-file", type=Path, default=Path("data/pref_a_split/valid.jsonl"))
  parser.add_argument("--out", type=Path, default=Path("outputs/pref-a-bt-sentseq-compare.json"))
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

  rows = unique_preference_pairs(load_jsonl(str(args.valid_file)))
  specs = [
    ("bt_ruri_a", args.bt_ruri, "bt"),
    ("bt_modernbert_a", args.bt_modernbert, "bt"),
    ("bt_modernbert_d_a", args.bt_modernbert_d, "bt"),
    ("sentseq_ruri_a", args.sentseq_ruri, "sentseq"),
    ("sentseq_modernbert_a", args.sentseq_modernbert, "sentseq"),
    ("sentseq_modernbert_d_a", args.sentseq_modernbert_d, "sentseq"),
  ]
  models: dict[str, dict] = {}
  for name, model_dir, kind in specs:
    entry = {
      "model_dir": str(model_dir),
      "kind": kind,
      **eval_pairwise(model_dir, rows, kind=kind, device=args.device),
    }
    entry["train_metrics"] = load_metrics(model_dir)
    models[name] = entry

  report = {
    "valid_file": str(args.valid_file),
    "n_pairs": len(rows),
    "metric": "pair_accuracy（人間の推敲 ≻ 下書き）; draft_over_human_n は下書き≻人間の件数",
    "models": models,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
