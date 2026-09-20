#!/usr/bin/env python3
"""R11: H3 ablation — s_den 単独 vs G2 から s_den 除去 vs R7 全入力。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import ead_out, ead_reports, ead_work, fmt_pct, load_jsonl, md_table, repo_root, write_json
from ead.gate_level import (
  INPUT_MODES,
  attach_surface_features,
  cache_features,
  load_floor_metrics,
  run_cv_and_valid,
)

try:
  import torch
except ImportError as exc:
  raise SystemExit(f"torch required: {exc}") from exc


ABLATION_MODES = ("s_den_only", "s_den_surface", "no_s_den", "r7")


def build_ablation_report(results: dict[str, Any], floor: dict[str, float], *, legacy_recall: float) -> str:
  lines = [
    "# ead-gate-level-ablation",
    "",
    "R11: H3 ablation（D valid, 3 seed 平均, norm=s_sum）。",
    f"床（表層特徴）: ラベル a 再現率 {floor['label_a_recall']:.3f}。",
    f"legacy G2（hidden + s_den のみ）: ラベル a 再現率 {legacy_recall:.3f}。",
    "",
    md_table(
      ["入力", "RPS", "ラベル a 再現率", "4値精度", "床超え（a）"],
      [
        [
          mode,
          fmt_pct(results[mode]["aggregate_valid"]["rps_mean"]),
          fmt_pct(results[mode]["aggregate_valid"]["label_a_recall_mean"]),
          fmt_pct(results[mode]["aggregate_valid"]["accuracy_4_mean"]),
          "✓"
          if results[mode]["aggregate_valid"]["label_a_recall_mean"] > floor["label_a_recall"]
          else "✗",
        ]
        for mode in ABLATION_MODES
      ],
    ),
    "",
    "## H3 判定",
    "",
  ]
  s_only = results["s_den_only"]["aggregate_valid"]["label_a_recall_mean"]
  s_surf = results["s_den_surface"]["aggregate_valid"]
  no_den = results["no_s_den"]["aggregate_valid"]["label_a_recall_mean"]
  full = results["r7"]["aggregate_valid"]["label_a_recall_mean"]
  if abs(s_only - legacy_recall) < 0.03:
    lines.append(
      f"- `s_den_only` の a 再現率 {s_only:.3f} は legacy {legacy_recall:.3f} に近い → **H3 は偽**（G2 は s_den の再表現に近い）。"
    )
  else:
    lines.append(
      f"- `s_den_only` の a 再現率 {s_only:.3f} は legacy {legacy_recall:.3f} と離れている → s_den 単独では legacy G2 を再現できない。"
    )
  lines.extend(
    [
      f"- `s_den_surface`（線形）: RPS {s_surf['rps_mean']:.3f}、a 再現率 {s_surf['label_a_recall_mean']:.3f}。床 RPS {floor['rps']:.3f}。",
      f"- `no_s_den` の a 再現率 {no_den:.3f}。`r7` 全入力 {full:.3f}。",
    ]
  )
  return "\n".join(lines) + "\n"


def load_legacy_recall(root: Path) -> float:
  path = ead_reports() / "ead-gate-level.json"
  if path.is_file():
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj["norms"]["s_sum"]["aggregate_valid"]["label_a_recall_mean"])
  return float("nan")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--adapter-dir", type=Path, default=None)
  parser.add_argument("--fit-train-jsonl", type=Path, default=None)
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--train-device", default="cpu")
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--trust-remote-code", action="store_true")
  parser.add_argument("--seeds", default="0,1,2")
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=1e-3)
  args = parser.parse_args()

  root = args.root.resolve()
  train_device = args.train_device
  if train_device == "cuda" and not torch.cuda.is_available():
    train_device = "cpu"

  train_raw = load_jsonl(root / "data/d/train.jsonl")
  valid_raw = load_jsonl(root / "data/d/valid.jsonl")
  for row in train_raw:
    row["_split"] = "train"
  for row in valid_raw:
    row["_split"] = "valid"
  all_rows = train_raw + valid_raw

  adapter_dir = args.adapter_dir or (ead_out() / "adapters" / "ead-den-a-excl")
  fit_path = args.fit_train_jsonl or (ead_work() / "ead-den-a-excl-train.jsonl")
  if not fit_path.is_absolute():
    fit_path = root / fit_path

  cache_path = ead_work() / "gate_level_features.npz"
  features = cache_features(
    all_rows,
    adapter_dir=adapter_dir,
    fit_train_jsonl=fit_path,
    base_model=args.base_model,
    device=args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu",
    max_seq_len=args.max_seq_length,
    trust_remote_code=args.trust_remote_code,
    cache_path=cache_path,
  )

  train_rows = [r for r in features if r.get("split") == "train" or r.get("_split") == "train"]
  valid_rows = [r for r in features if r.get("split") == "valid" or r.get("_split") == "valid"]
  attach_surface_features(train_rows + valid_rows, all_rows)

  floor = load_floor_metrics()
  legacy_recall = load_legacy_recall(root)
  seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

  results: dict[str, Any] = {}
  for mode in ABLATION_MODES:
    if mode not in INPUT_MODES:
      raise SystemExit(f"unknown ablation mode {mode}")
    mlp_hidden = 0 if mode == "s_den_surface" else None
    kwargs: dict = {
      "train_rows": train_rows,
      "valid_rows": valid_rows,
      "norm": "s_sum",
      "input_mode": mode,
      "adapter_subdir": f"ead-gate-level-ablation-{mode}",
      "seeds": seeds,
      "epochs": args.epochs,
      "lr": args.lr,
      "device": train_device,
      "floor_recall_a": floor["label_a_recall"],
      "floor_rps": floor["rps"],
    }
    if mlp_hidden is not None:
      kwargs["mlp_hidden"] = mlp_hidden
    results[mode] = run_cv_and_valid(**kwargs)

  reports = ead_reports()
  write_json(
    reports / "ead-gate-level-ablation.json",
    {"floor": floor, "legacy_recall": legacy_recall, "modes": results},
  )
  (reports / "ead-gate-level-ablation.md").write_text(
    build_ablation_report(results, floor, legacy_recall=legacy_recall),
    encoding="utf-8",
  )
  print(
    json.dumps(
      {
        "wrote": str(reports / "ead-gate-level-ablation.md"),
        "s_den_only_recall": results["s_den_only"]["aggregate_valid"]["label_a_recall_mean"],
        "no_s_den_recall": results["no_s_den"]["aggregate_valid"]["label_a_recall_mean"],
        "r7_recall": results["r7"]["aggregate_valid"]["label_a_recall_mean"],
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
