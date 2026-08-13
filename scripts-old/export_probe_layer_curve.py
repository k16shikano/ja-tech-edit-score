#!/usr/bin/env python3
"""Qwen3-8B 内部表現プローブの全層結果を公開用 CSV / 図に整理する。

既存の probe_report.json / probe_paired_diff.json を再集計するだけで、
活性値の再抽出は行わない。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
    "--steering-dir",
    type=Path,
    default=ROOT / "outputs/steering/Qwen__Qwen3-8B",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=ROOT / "results/probe_layer_results.csv",
  )
  parser.add_argument(
    "--figure-pdf",
    type=Path,
    default=ROOT / "results/figures/probe_layer_accuracy.pdf",
  )
  parser.add_argument(
    "--figure-svg",
    type=Path,
    default=ROOT / "results/figures/probe_layer_accuracy.svg",
  )
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()

  for out in (args.out, args.figure_pdf, args.figure_svg):
    if out.exists() and not args.overwrite:
      raise SystemExit(f"exists (pass --overwrite): {out}")

  meta_path = args.steering_dir / "meta.json"
  single_path = args.steering_dir / "probe_report.json"
  paired_path = args.steering_dir / "probe_paired_diff.json"
  for path in (meta_path, single_path, paired_path):
    if not path.is_file():
      raise SystemExit(f"missing: {path}")

  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  single = json.loads(single_path.read_text(encoding="utf-8"))
  paired = json.loads(paired_path.read_text(encoding="utf-8"))

  model_name = meta.get("model", "Qwen/Qwen3-8B")
  # revision は meta に無い場合がある
  model_revision = meta.get("model_revision") or meta.get("revision") or ""
  pooling = "mean"
  if "pooling" in meta:
    pooling = meta["pooling"]
  elif "mean-pooled" in str(meta.get("note", "")):
    pooling = "mean"

  seed = single.get("seed", "")
  rows: list[dict] = []

  for layer in single["layers"]:
    rows.append(
      {
        "model_name": model_name,
        "model_revision": model_revision,
        "layer": layer["layer"],
        "layer_label": layer.get("label", ""),
        "probe_type": "single_document_mean_difference_projection",
        "pooling_method": pooling,
        "micro_accuracy": layer["micro_accuracy"],
        "macro_accuracy": layer["macro_accuracy"],
        "num_examples": single["n_pairs"],
        "num_projects": single["n_projects"],
        "seed": seed,
        "best_layer_posthoc": int(layer["layer"] == single["best_layer"]["layer"]),
      }
    )

  for layer in paired["layers"]:
    rows.append(
      {
        "model_name": model_name,
        "model_revision": model_revision,
        "layer": layer["layer"],
        "layer_label": "",
        "probe_type": "paired_difference_sign_mean_diff_direction_lopo",
        "pooling_method": pooling,
        "micro_accuracy": layer["micro"],
        "macro_accuracy": layer["macro"],
        "num_examples": meta["n_pairs"],
        "num_projects": len(meta.get("projects", [])),
        "seed": seed,
        "best_layer_posthoc": int(
          layer["layer"] == paired.get("best", {}).get("layer", -1)
        ),
      }
    )

  write_csv(
    args.out,
    rows,
    [
      "model_name",
      "model_revision",
      "layer",
      "layer_label",
      "probe_type",
      "pooling_method",
      "micro_accuracy",
      "macro_accuracy",
      "num_examples",
      "num_projects",
      "seed",
      "best_layer_posthoc",
    ],
  )
  print(f"wrote {args.out} ({len(rows)} rows)")
  print(
    "NOTE: all layers were searched; best layer is post-hoc selected "
    "(upward selection bias). Only mean pooling was used for this full run; "
    "no claim that mean pooling is optimal."
  )

  # figures
  import matplotlib.pyplot as plt

  single_layers = [r for r in rows if r["probe_type"].startswith("single")]
  paired_layers = [r for r in rows if r["probe_type"].startswith("paired")]
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.plot(
    [r["layer"] for r in single_layers],
    [r["micro_accuracy"] for r in single_layers],
    label="single-document (micro)",
    marker="o",
    markersize=3,
  )
  ax.plot(
    [r["layer"] for r in paired_layers],
    [r["micro_accuracy"] for r in paired_layers],
    label="paired-difference sign (micro)",
    marker="s",
    markersize=3,
  )
  best_s = single["best_layer"]["layer"]
  best_p = paired.get("best", {}).get("layer")
  ax.axvline(best_s, color="C0", linestyle="--", alpha=0.5, label=f"best single L{best_s}")
  if best_p is not None:
    ax.axvline(best_p, color="C1", linestyle="--", alpha=0.5, label=f"best paired L{best_p}")
  ax.set_xlabel("layer (0=embed)")
  ax.set_ylabel("micro accuracy")
  ax.set_title(
    f"{model_name} probe (pooling={pooling}; best layers post-hoc)"
  )
  ax.set_ylim(0.45, 0.75)
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  args.figure_pdf.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.figure_pdf)
  fig.savefig(args.figure_svg)
  plt.close(fig)
  print(f"wrote {args.figure_pdf}")
  print(f"wrote {args.figure_svg}")

  # reading/norms smoke-only note
  for mode in ("reading", "norms"):
    smoke = args.steering_dir.parent / f"Qwen__Qwen3-8B--{mode}" / "probe_report.json"
    if smoke.is_file():
      rep = json.loads(smoke.read_text(encoding="utf-8"))
      print(
        f"NOTE: {mode} probe exists but n_pairs={rep.get('n_pairs')} "
        f"(not full {meta['n_pairs']}); not merged into primary CSV."
      )


if __name__ == "__main__":
  main()
