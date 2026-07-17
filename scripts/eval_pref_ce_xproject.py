#!/usr/bin/env python3
"""leave-one-project-out で cross-encoder BT 報酬モデルを評価する（GPU 向け）。

fold ごとにベースモデルから学習し直すので、fold 数 × 学習時間がかかる。
pref-bt（凍結 ruri + 線形ヘッド、LOPO micro 0.975）との比較が目的。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_pref_ce import (
  CeTrainConfig,
  eval_ce_pairs,
  load_jsonl,
  train_ce_model,
  unique_preference_pairs,
)


def project_of(row: dict) -> str:
  return str(row.get("meta", {}).get("project_id", "") or "(unknown)")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="preference jsonl")
  parser.add_argument("--base-model", default="sbintuitions/modernbert-ja-130m")
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--lr", type=float, default=3e-5)
  parser.add_argument("--weight-decay", type=float, default=0.01)
  parser.add_argument("--warmup-ratio", type=float, default=0.1)
  parser.add_argument("--grad-accum", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"]
  )
  parser.add_argument("--gradient-checkpointing", action="store_true")
  parser.add_argument(
    "--only-projects",
    default="",
    help="カンマ区切りで fold を絞る（スモーク用）。空なら全 fold",
  )
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

  targets = projects
  if args.only_projects.strip():
    wanted = {p.strip() for p in args.only_projects.split(",") if p.strip()}
    unknown = wanted - set(projects)
    if unknown:
      raise SystemExit(f"unknown projects: {sorted(unknown)}")
    targets = [p for p in projects if p in wanted]

  cfg = CeTrainConfig(
    base_model=args.base_model,
    max_length=args.max_length,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    grad_accum=args.grad_accum,
    seed=args.seed,
    precision=args.precision,
    gradient_checkpointing=args.gradient_checkpointing,
  )

  folds = []
  for held_out in targets:
    eval_idx = set(indices_by_project[held_out])
    train_rows = [r for i, r in enumerate(rows) if i not in eval_idx]
    eval_rows = [rows[i] for i in sorted(eval_idx)]

    model, tokenizer, _ = train_ce_model(
      train_rows, cfg, log_prefix=f"[{held_out}] "
    )
    metrics = eval_ce_pairs(
      model, tokenizer, eval_rows,
      max_length=cfg.max_length, batch_size=cfg.batch_size,
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

    # fold ごとにモデルを解放（GPU メモリ）
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
          "base_model": args.base_model,
          "max_length": args.max_length,
          "epochs": args.epochs,
          "lr": args.lr,
          "batch_size": args.batch_size,
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
