#!/usr/bin/env python3
"""leave-one-project-out で文列 Transformer BT 報酬モデルを評価する。

fold ごとに学習し直す。pref-bt（凍結 ruri + 線形ヘッド、LOPO micro 0.975）との比較が目的。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pref_static_utils import load_jsonl, normalize_truncate_dim
from train_pref_sentseq import (
  SentSeqTrainConfig,
  encode_unique_sentences,
  eval_sentseq_pairs,
  resolve_device,
  train_sentseq_model,
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
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--num-layers", type=int, default=2)
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  parser.add_argument(
    "--only-projects",
    default="",
    help="カンマ区切りで fold を絞る（スモーク用）。空なら全 fold",
  )
  parser.add_argument("--report", default="")
  args = parser.parse_args()

  device = resolve_device(args.device)
  rows = unique_preference_pairs(load_jsonl(args.input))
  if len(rows) < 2:
    raise SystemExit("need preference pairs")

  indices_by_project: dict[str, list[int]] = defaultdict(list)
  for i, row in enumerate(rows):
    indices_by_project[project_of(row)].append(i)
  projects = sorted(indices_by_project, key=lambda p: -len(indices_by_project[p]))
  if len(projects) < 2:
    raise SystemExit("need at least 2 projects")

  targets = projects
  if args.only_projects.strip():
    wanted = {p.strip() for p in args.only_projects.split(",") if p.strip()}
    unknown = wanted - set(projects)
    if unknown:
      raise SystemExit(f"unknown projects: {sorted(unknown)}")
    targets = [p for p in projects if p in wanted]

  cfg = SentSeqTrainConfig(
    sentence_model_name=args.model,
    truncate_dim=normalize_truncate_dim(args.truncate_dim),
    text_prefix=args.text_prefix,
    max_seq_length=args.max_seq_length,
    encode_batch_size=args.encode_batch_size,
    d_model=args.d_model,
    num_layers=args.num_layers,
    max_sents=args.max_sents,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
  )

  # 埋め込みモデルは凍結しており fold に依存しないので、全行ぶんを一度だけ計算して
  # 全 fold で使い回す（fold ごとの再計算は所要時間の大半を占めていた）。
  print(f"encoding sentence embeddings once for {len(rows)} rows ...", flush=True)
  shared_embeddings = encode_unique_sentences(rows, cfg, device=device)

  folds = []
  for held_out in targets:
    eval_idx = set(indices_by_project[held_out])
    train_rows = [r for i, r in enumerate(rows) if i not in eval_idx]
    eval_rows = [rows[i] for i in sorted(eval_idx)]

    model, _, _, prepared = train_sentseq_model(
      train_rows,
      eval_rows,
      cfg,
      device=device,
      log_prefix=f"[{held_out}] ",
      precomputed_embeddings=shared_embeddings,
    )
    metrics = eval_sentseq_pairs(
      model,
      eval_rows,
      prepared,
      device=device,
      batch_size=cfg.batch_size,
    )
    fold = {
      "project_id": held_out,
      "eval_pairs": len(eval_rows),
      "train_pairs": len(train_rows),
      **metrics,
    }
    folds.append(fold)
    print(
      f"{held_out:28s} n={fold['eval_pairs']:5d} "
      f"acc={fold['pair_accuracy']:.4f} bt_loss={fold['bt_loss']:.4f} "
      f"margin={fold['mean_margin']:.3f}",
      flush=True,
    )

    del model
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

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
          "sentence_model_name": cfg.sentence_model_name,
          "max_seq_length": cfg.max_seq_length,
          "max_sents": cfg.max_sents,
          "d_model": cfg.d_model,
          "num_layers": cfg.num_layers,
          "epochs": cfg.epochs,
          "lr": cfg.lr,
          "batch_size": cfg.batch_size,
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
