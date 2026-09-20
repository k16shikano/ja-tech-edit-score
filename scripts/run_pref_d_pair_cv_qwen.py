#!/usr/bin/env python3
"""Qwen D ペア BT/GPM の 5 分割 CV を実行し eval_cv_report.json を出す。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pref_d_pair_utils import (
  group_rows_by_item,
  qwen_only_items,
  rank_pair_metrics_gpm,
  rank_pair_metrics_scalar,
)
from pref_static_utils import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python3"
TRAIN_SCRIPT = ROOT / "scripts" / "train_pref_d_pair_qwen.py"


def run_train(
  *,
  mode: str,
  base_model: str,
  fold: int,
  epochs: int,
  output_dir: Path,
  data_dir: Path,
  ledger_file: Path,
  head_dim: int,
  batch_items: int,
  pair_batch_size: int,
  max_length: int,
  learning_rate: float,
  base_quantization: str,
  seed: int,
) -> dict:
  folds_dir = data_dir / "folds"
  cmd = [
    str(PYTHON),
    str(TRAIN_SCRIPT),
    "--mode",
    mode,
    "--base-model",
    base_model,
    "--train-file",
    str(folds_dir / f"fold{fold}_train.jsonl"),
    "--valid-file",
    str(folds_dir / f"fold{fold}_valid.jsonl"),
    "--ledger-file",
    str(ledger_file),
    "--output-dir",
    str(output_dir / f"fold{fold}"),
    "--fold",
    str(fold),
    "--epochs",
    str(epochs),
    "--head-dim",
    str(head_dim),
    "--batch-items",
    str(batch_items),
    "--pair-batch-size",
    str(pair_batch_size),
    "--max-length",
    str(max_length),
    "--learning-rate",
    str(learning_rate),
    "--base-quantization",
    base_quantization,
    "--seed",
    str(seed),
  ]
  print(" ".join(cmd), flush=True)
  subprocess.run(cmd, check=True)
  return json.loads((output_dir / f"fold{fold}" / "summary.json").read_text(encoding="utf-8"))


def load_fold_summary(output_dir: Path, fold: int) -> dict | None:
  path = output_dir / f"fold{fold}" / "summary.json"
  if not path.is_file():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def run_train_or_skip(**kwargs) -> dict:
  output_dir = kwargs["output_dir"]
  fold = kwargs["fold"]
  existing = load_fold_summary(output_dir, fold)
  if existing is not None:
    print(
      f"skip fold {fold} (summary exists best_epoch={existing.get('best_epoch')})",
      flush=True,
    )
    return existing
  return run_train(**kwargs)


def load_oof_predictions(cv_dir: Path) -> list[dict]:
  rows: list[dict] = []
  for fold in range(5):
    path = cv_dir / f"fold{fold}" / "best_valid_predictions.jsonl"
    if not path.is_file():
      raise FileNotFoundError(path)
    rows.extend(load_jsonl(str(path)))
  if len(rows) != 800:
    raise ValueError(f"expected 800 OOF rows, got {len(rows)}")
  return rows


def aggregate_pair_metrics(oof_rows: list[dict], *, mode: str) -> dict:
  items = group_rows_by_item(oof_rows)
  if mode == "bt":
    scores = {r["row_id"]: r.get("delta") for r in oof_rows}
    all_metrics = rank_pair_metrics_scalar(items, scores)
    qwen_metrics = rank_pair_metrics_scalar(qwen_only_items(items), scores)
  else:
    vectors = {r["row_id"]: r.get("vector") for r in oof_rows}
    all_metrics = rank_pair_metrics_gpm(items, vectors)
    qwen_metrics = rank_pair_metrics_gpm(qwen_only_items(items), vectors)
  return {"all": all_metrics, "qwen_only": qwen_metrics}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=["bt", "gpm"], required=True)
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "d")
  parser.add_argument("--ledger-file", type=Path, default=ROOT / "data" / "d" / "pair_ledger.jsonl")
  parser.add_argument("--output-dir", type=Path, default="")
  parser.add_argument("--max-epochs", type=int, default=40)
  parser.add_argument("--head-dim", type=int, default=32)
  parser.add_argument("--batch-items", type=int, default=8)
  parser.add_argument("--pair-batch-size", type=int, default=8)
  parser.add_argument("--max-length", type=int, default=2048)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--base-quantization", choices=["8bit", "4bit"], default="8bit")
  parser.add_argument("--max-fold", type=int, default=4, help="last fold index to run (0-4)")
  args = parser.parse_args()

  if args.mode == "gpm" and args.head_dim % 2 != 0:
    raise SystemExit("--head-dim must be even for GPM")
  if not 0 <= args.max_fold <= 4:
    raise SystemExit("--max-fold must be between 0 and 4")

  default_name = "pref-d-bt-qwen3-8b" if args.mode == "bt" else "pref-d-gpm-qwen3-8b"
  output_dir = Path(args.output_dir) if args.output_dir else (ROOT / "outputs" / default_name)
  output_dir.mkdir(parents=True, exist_ok=True)

  report_path = output_dir / "eval_cv_report.json"
  folds_done = all(load_fold_summary(output_dir, fold) for fold in range(args.max_fold + 1))
  if args.max_fold == 4 and report_path.is_file() and folds_done:
    print(f"skip CV (all folds done): {report_path}", flush=True)
    print(report_path.read_text(encoding="utf-8"))
    return
  if args.max_fold < 4 and folds_done:
    print(f"skip CV (folds 0-{args.max_fold} already done)", flush=True)
    return

  train_kwargs = dict(
    mode=args.mode,
    base_model=args.base_model,
    output_dir=output_dir,
    data_dir=args.data_dir,
    ledger_file=args.ledger_file,
    head_dim=args.head_dim,
    batch_items=args.batch_items,
    pair_batch_size=args.pair_batch_size,
    max_length=args.max_length,
    learning_rate=args.learning_rate,
    base_quantization=args.base_quantization,
    seed=args.seed,
  )

  fold0 = run_train_or_skip(fold=0, epochs=args.max_epochs, **train_kwargs)
  selected_epoch = int(fold0["best_epoch"])
  if selected_epoch <= 0:
    raise SystemExit("fold 0 did not produce a valid best_epoch")

  fold_summaries = [fold0]
  for fold in range(1, args.max_fold + 1):
    fold_summaries.append(run_train_or_skip(fold=fold, epochs=selected_epoch, **train_kwargs))

  if args.max_fold < 4:
    out = {
      "mode": args.mode,
      "max_fold": args.max_fold,
      "selected_epoch_from_fold0": selected_epoch,
      "fold_summaries": fold_summaries,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return

  oof_rows = load_oof_predictions(output_dir)
  pair_metrics = aggregate_pair_metrics(oof_rows, mode=args.mode)
  out = {
    "mode": args.mode,
    "backend": f"causal_lora_{args.base_quantization}",
    "base_model": args.base_model,
    "base_quantization": args.base_quantization,
    "head_dim": args.head_dim if args.mode == "gpm" else 1,
    "max_length": args.max_length,
    "learning_rate": args.learning_rate,
    "selected_epoch_from_fold0": selected_epoch,
    "fold_summaries": fold_summaries,
    "oof_n": len(oof_rows),
    "pair_metrics": pair_metrics,
  }
  report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
