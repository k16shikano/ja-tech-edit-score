#!/usr/bin/env python3
"""pref-bt LOPO JSON を公開用 CSV に整形し、同一 6274 組の pref-sentseq LOPO と並べる。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--bt-report",
    type=Path,
    default=ROOT / "results/raw/eval_bt_xproject_6274.json",
  )
  parser.add_argument(
    "--sentseq-report",
    type=Path,
    default=ROOT / "outputs/eval_sentseq_xproject.json",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=ROOT / "results/pref_bt_lopo_6274.csv",
  )
  parser.add_argument(
    "--compare-out",
    type=Path,
    default=ROOT / "results/raw/lopo_6274_bt_vs_sentseq.json",
  )
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()

  for out in (args.out, args.compare_out):
    if out.exists() and not args.overwrite:
      raise SystemExit(f"exists (pass --overwrite): {out}")
  if not args.bt_report.is_file():
    raise SystemExit(f"missing bt LOPO report: {args.bt_report}")
  if not args.sentseq_report.is_file():
    raise SystemExit(f"missing sentseq LOPO report: {args.sentseq_report}")

  bt = json.loads(args.bt_report.read_text(encoding="utf-8"))
  ss = json.loads(args.sentseq_report.read_text(encoding="utf-8"))
  if bt.get("total_pairs") != 6274:
    raise SystemExit(f"bt total_pairs={bt.get('total_pairs')} expected 6274")
  if ss.get("total_pairs") != 6274:
    raise SystemExit(f"sentseq total_pairs={ss.get('total_pairs')} expected 6274")

  rows = []
  for fold in bt["folds"]:
    n = int(fold["eval_pairs"])
    acc = float(fold["pair_accuracy"])
    # correct は整数で再構成（丸め誤差に注意）
    correct = int(round(acc * n))
    rows.append(
      {
        "held_out_project": fold["project_id"],
        "num_train_pairs": fold["train_pairs"],
        "num_test_pairs": n,
        "accuracy": acc,
        "correct": correct,
        "total": n,
        "bt_loss": fold.get("bt_loss", ""),
        "mean_margin": fold.get("mean_margin", ""),
        "model": "pref-bt",
        "data_version": "pref_dataset_unique_6274",
        "seed": 0,
      }
    )

  # project-level bootstrap CI for macro (exploratory secondary)
  accs = np.asarray([r["accuracy"] for r in rows], dtype=np.float64)
  rng = np.random.default_rng(0)
  boots = []
  for _ in range(10000):
    sample = rng.choice(accs, size=len(accs), replace=True)
    boots.append(float(sample.mean()))
  boots_arr = np.asarray(boots)
  project_boot = {
    "metric": "macro_accuracy_over_projects",
    "observed": float(accs.mean()),
    "ci_95_lower": float(np.quantile(boots_arr, 0.025)),
    "ci_95_upper": float(np.quantile(boots_arr, 0.975)),
    "n_bootstrap": 10000,
    "seed": 0,
    "note": "project as resampling unit; not pair-level",
  }

  write_csv(
    args.out,
    rows,
    [
      "held_out_project",
      "num_train_pairs",
      "num_test_pairs",
      "accuracy",
      "correct",
      "total",
      "bt_loss",
      "mean_margin",
      "model",
      "data_version",
      "seed",
    ],
  )

  ss_by = {f["project_id"]: f for f in ss["folds"]}
  bt_by = {f["project_id"]: f for f in bt["folds"]}
  if set(ss_by) != set(bt_by):
    raise SystemExit(f"project mismatch: {set(ss_by) ^ set(bt_by)}")

  compare = {
    "data_version": "pref_dataset unique preference pairs = 6274",
    "pref_bt": {
      "report": str(args.bt_report),
      "total_pairs": bt["total_pairs"],
      "micro_pair_accuracy": bt["micro_pair_accuracy"],
      "macro_pair_accuracy": bt["macro_pair_accuracy"],
      "n_folds": len(bt["folds"]),
      "embedding_model": bt.get("embedding_model"),
      "seed": 0,
      "project_macro_bootstrap": project_boot,
    },
    "pref_sentseq": {
      "report": str(args.sentseq_report),
      "total_pairs": ss["total_pairs"],
      "micro_pair_accuracy": ss["micro_pair_accuracy"],
      "macro_pair_accuracy": ss["macro_pair_accuracy"],
      "n_folds": len(ss["folds"]),
      "d_model": ss.get("d_model"),
      "num_layers": ss.get("num_layers"),
      "epochs": ss.get("epochs"),
      "lr": ss.get("lr"),
    },
    "per_project": [
      {
        "project_id": p,
        "n_test": bt_by[p]["eval_pairs"],
        "bt_accuracy": bt_by[p]["pair_accuracy"],
        "sentseq_accuracy": ss_by[p]["pair_accuracy"],
        "difference_bt_minus_sentseq": (
          bt_by[p]["pair_accuracy"] - ss_by[p]["pair_accuracy"]
        ),
      }
      for p in sorted(bt_by, key=lambda x: -bt_by[x]["eval_pairs"])
    ],
    "legacy_bt_lopo_note": (
      "outputs/eval_bt_xproject.json has total_pairs=5887 (pre-section-pair). "
      "Do not compare that file to sentseq 6274."
    ),
  }
  args.compare_out.parent.mkdir(parents=True, exist_ok=True)
  args.compare_out.write_text(json.dumps(compare, ensure_ascii=False, indent=2) + "\n")
  print(f"wrote {args.out}")
  print(f"wrote {args.compare_out}")
  print(
    f"bt micro={bt['micro_pair_accuracy']:.4f} macro={bt['macro_pair_accuracy']:.4f} | "
    f"sentseq micro={ss['micro_pair_accuracy']:.4f} macro={ss['macro_pair_accuracy']:.4f}"
  )


if __name__ == "__main__":
  main()
