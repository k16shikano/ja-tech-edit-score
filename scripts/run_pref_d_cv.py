#!/usr/bin/env python3
"""D 実験の 5 分割 CV を実行し、800 行連結の eval_cv_report.json を出す。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pref_d_interval_utils import eval_report, load_d_rows
from pref_static_utils import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python3"
TRAIN_SCRIPT = ROOT / "scripts" / "train_pref_d_interval.py"
DEFAULT_DATA = ROOT / "data" / "d"
DEFAULT_FOLDS = DEFAULT_DATA / "folds"


def run_train(
  *,
  backend: str,
  base_model: str,
  fold: int,
  epochs: int,
  output_dir: Path,
  data_dir: Path,
  batch_size: int,
  seed: int,
) -> dict:
  folds_dir = data_dir / "folds"
  cmd = [
    str(PYTHON),
    str(TRAIN_SCRIPT),
    "--backend",
    backend,
    "--base-model",
    base_model,
    "--train-file",
    str(folds_dir / f"fold{fold}_train.jsonl"),
    "--valid-file",
    str(folds_dir / f"fold{fold}_valid.jsonl"),
    "--output-dir",
    str(output_dir / f"fold{fold}"),
    "--fold",
    str(fold),
    "--epochs",
    str(epochs),
    "--batch-size",
    str(batch_size),
    "--seed",
    str(seed),
  ]
  print(" ".join(cmd), flush=True)
  subprocess.run(cmd, check=True)
  return json.loads((output_dir / f"fold{fold}" / "summary.json").read_text(encoding="utf-8"))


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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=["modernbert", "ruri"], default="modernbert")
  parser.add_argument("--base-model", default="")
  parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--output-dir", type=Path, default="")
  parser.add_argument("--max-epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  base_model = args.base_model or (
    "sbintuitions/modernbert-ja-310m"
    if args.backend == "modernbert"
    else "cl-nagoya/ruri-v3-30m"
  )
  output_dir = Path(args.output_dir) if args.output_dir else (
    ROOT / "outputs" / ("pref-d-modernbert-cv" if args.backend == "modernbert" else "pref-d-ruri-cv")
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  fold0 = run_train(
    backend=args.backend,
    base_model=base_model,
    fold=0,
    epochs=args.max_epochs,
    output_dir=output_dir,
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    seed=args.seed,
  )
  selected_epoch = int(fold0["best_epoch"])
  if selected_epoch <= 0:
    raise SystemExit("fold 0 did not produce a valid best_epoch")

  fold_summaries = [fold0]
  for fold in range(1, 5):
    fold_summaries.append(
      run_train(
        backend=args.backend,
        base_model=base_model,
        fold=fold,
        epochs=selected_epoch,
        output_dir=output_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
      )
    )

  oof_rows = load_oof_predictions(output_dir)
  f_values = [float(r["f"]) for r in oof_rows]
  report = eval_report(oof_rows, f_values)
  out = {
    "backend": args.backend,
    "base_model": base_model,
    "selected_epoch_from_fold0": selected_epoch,
    "fold_summaries": fold_summaries,
    "oof_n": len(oof_rows),
    "metrics": report,
  }
  report_path = output_dir / "eval_cv_report.json"
  report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
