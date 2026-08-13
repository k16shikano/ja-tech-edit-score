#!/usr/bin/env python3
"""leave-one-project-out で BT 報酬モデルを評価する。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from pref_static_utils import (
  collect_unique_texts,
  encode_text_map,
  load_jsonl,
  normalize_truncate_dim,
)
from train_pref_bt import (
  build_pointwise_matrix,
  eval_pair_accuracy,
  train_bt_head,
  unique_preference_pairs,
)


def project_of(row: dict) -> str:
  return str(row.get("meta", {}).get("project_id", "") or "(unknown)")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="preference jsonl")
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--lr", type=float, default=1e-2)
  parser.add_argument("--weight-decay", type=float, default=1e-3)
  parser.add_argument("--report", default="")
  args = parser.parse_args()

  rows = unique_preference_pairs(load_jsonl(args.input))
  if len(rows) < 2:
    raise SystemExit("need preference pairs")

  indices_by_project: dict[str, list[int]] = defaultdict(list)
  for i, row in enumerate(rows):
    indices_by_project[project_of(row)].append(i)
  projects = sorted(indices_by_project, key=lambda p: -len(indices_by_project[p]))
  if len(projects) < 2:
    raise SystemExit("need at least 2 projects")

  truncate_dim = normalize_truncate_dim(args.truncate_dim)
  encoder = SentenceTransformer(args.model, device="cpu", truncate_dim=truncate_dim)
  if args.max_seq_length > 0:
    encoder.max_seq_length = args.max_seq_length
  text_to_embedding = encode_text_map(
    encoder,
    collect_unique_texts(rows),
    batch_size=args.batch_size,
    normalize_embeddings=True,
    text_prefix=args.text_prefix,
  )
  x_w = build_pointwise_matrix(rows, text_to_embedding, key="candidate_a")
  x_l = build_pointwise_matrix(rows, text_to_embedding, key="candidate_b")

  folds = []
  for held_out in projects:
    eval_idx = np.asarray(indices_by_project[held_out], dtype=np.int64)
    train_mask = np.ones(len(rows), dtype=bool)
    train_mask[eval_idx] = False
    head, scaler, _ = train_bt_head(
      x_w[train_mask],
      x_l[train_mask],
      epochs=args.epochs,
      lr=args.lr,
      weight_decay=args.weight_decay,
      seed=0,
    )
    metrics = eval_pair_accuracy(head, scaler, x_w[eval_idx], x_l[eval_idx])
    fold = {
      "project_id": held_out,
      "eval_pairs": int(eval_idx.size),
      "train_pairs": int(train_mask.sum()),
      **metrics,
    }
    folds.append(fold)
    print(
      f"{held_out:28s} n={fold['eval_pairs']:5d} "
      f"acc={fold['pair_accuracy']:.4f} bt_loss={fold['bt_loss']:.4f} "
      f"margin={fold['mean_margin']:.3f}"
    )

  total = sum(f["eval_pairs"] for f in folds)
  micro_acc = sum(f["pair_accuracy"] * f["eval_pairs"] for f in folds) / total
  macro_acc = float(np.mean([f["pair_accuracy"] for f in folds]))
  micro_loss = sum(f["bt_loss"] * f["eval_pairs"] for f in folds) / total
  print("-" * 64)
  print(f"projects: {len(folds)}  pairs: {total}")
  print(f"micro pair accuracy: {micro_acc:.4f}")
  print(f"macro pair accuracy: {macro_acc:.4f}")
  print(f"micro bt_loss: {micro_loss:.4f}")

  if args.report:
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
      json.dumps(
        {
          "embedding_model": args.model,
          "text_prefix": args.text_prefix,
          "max_seq_length": args.max_seq_length if args.max_seq_length > 0 else None,
          "total_pairs": total,
          "micro_pair_accuracy": micro_acc,
          "macro_pair_accuracy": macro_acc,
          "micro_bt_loss": micro_loss,
          "folds": folds,
        },
        ensure_ascii=False,
        indent=2,
      ),
      encoding="utf-8",
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
  main()
