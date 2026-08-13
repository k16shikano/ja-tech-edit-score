#!/usr/bin/env python3
"""Hard Eval 項目単位の McNemar 検定と対応付き bootstrap。

再標本化の単位は hard evaluation の項目。候補対を独立標本として扱わない。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n <= 0:
    return (float("nan"), float("nan"))
  p = k / n
  denom = 1.0 + z * z / n
  centre = p + z * z / (2.0 * n)
  margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
  return ((centre - margin) / denom, (centre + margin) / denom)


def load_item_csv(path: Path) -> list[dict]:
  with path.open(encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def model_key(row: dict) -> str:
  return f"{row['model_name']}/{row['model_config']}"


def index_by_eval_model(rows: list[dict]) -> dict[tuple[str, str], dict[str, int]]:
  """(eval_set, model_key) -> {item_id: top1_correct}"""
  out: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
  for row in rows:
    key = (row["evaluation_set"], model_key(row))
    item_id = row["item_id"]
    if item_id in out[key]:
      raise SystemExit(f"duplicate item {item_id} for {key}")
    out[key][item_id] = int(row["top1_correct"])
  return out


def index_pairwise(rows: list[dict]) -> dict[tuple[str, str], dict[str, tuple[int, int]]]:
  out: dict[tuple[str, str], dict[str, tuple[int, int]]] = defaultdict(dict)
  for row in rows:
    key = (row["evaluation_set"], model_key(row))
    out[key][row["item_id"]] = (int(row["pairwise_correct"]), int(row["pairwise_total"]))
  return out


def exact_mcnemar(b: int, c: int) -> float:
  """両側 exact McNemar（不一致のみ、p=0.5）。"""
  n = b + c
  if n == 0:
    return 1.0
  return float(binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)


def paired_bootstrap_top1(
  a: np.ndarray,
  b: np.ndarray,
  *,
  n_boot: int,
  seed: int,
) -> dict[str, float]:
  rng = np.random.default_rng(seed)
  n = a.shape[0]
  obs = float(a.mean() - b.mean())
  idx = rng.integers(0, n, size=(n_boot, n))
  diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
  return {
    "observed_accuracy_difference": obs,
    "bootstrap_mean_difference": float(diffs.mean()),
    "ci_95_lower": float(np.quantile(diffs, 0.025)),
    "ci_95_upper": float(np.quantile(diffs, 0.975)),
    "num_bootstrap_samples": float(n_boot),
  }


def paired_bootstrap_pairwise_rate(
  a_agree: np.ndarray,
  a_total: np.ndarray,
  b_agree: np.ndarray,
  b_total: np.ndarray,
  *,
  n_boot: int,
  seed: int,
) -> dict[str, float]:
  """項目を cluster として再標本化し、項目内の候補対をまとめて保持。"""
  rng = np.random.default_rng(seed)
  n = a_agree.shape[0]
  obs_a = float(a_agree.sum() / a_total.sum())
  obs_b = float(b_agree.sum() / b_total.sum())
  obs = obs_a - obs_b
  idx = rng.integers(0, n, size=(n_boot, n))
  diffs = []
  for row in idx:
    da = a_agree[row].sum() / a_total[row].sum()
    db = b_agree[row].sum() / b_total[row].sum()
    diffs.append(da - db)
  diffs_arr = np.asarray(diffs, dtype=np.float64)
  return {
    "observed_accuracy_difference": obs,
    "bootstrap_mean_difference": float(diffs_arr.mean()),
    "ci_95_lower": float(np.quantile(diffs_arr, 0.025)),
    "ci_95_upper": float(np.quantile(diffs_arr, 0.975)),
    "num_bootstrap_samples": float(n_boot),
  }


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
    "--item-csv",
    type=Path,
    default=ROOT / "results/hard_eval_item_correctness.csv",
  )
  parser.add_argument(
    "--mcnemar-out",
    type=Path,
    default=ROOT / "results/mcnemar_results.csv",
  )
  parser.add_argument(
    "--bootstrap-out",
    type=Path,
    default=ROOT / "results/paired_bootstrap_results.csv",
  )
  parser.add_argument(
    "--accuracy-out",
    type=Path,
    default=ROOT / "results/raw/hard_eval_model_accuracy.json",
  )
  parser.add_argument("--n-bootstrap", type=int, default=100_000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--compare",
    action="append",
    default=[],
    help="比較指定 eval_set:model_a/config_a:model_b/config_b （複数可）",
  )
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()

  for out in (args.mcnemar_out, args.bootstrap_out, args.accuracy_out):
    if out.exists() and not args.overwrite:
      raise SystemExit(f"exists (pass --overwrite): {out}")
  if not args.item_csv.is_file():
    raise SystemExit(f"missing: {args.item_csv}")

  rows = load_item_csv(args.item_csv)
  top1 = index_by_eval_model(rows)
  pair = index_pairwise(rows)

  # 既定比較: 各 eval_set 内の主要モデル同士
  default_compares = [
    "v1:pref-sentseq/lr1e-4-ep40:pref-sentseq/lr3e-4-ep40",
    "v1:pref-sentseq/lr3e-4-ep40:pref-bt/default",
    "v1:pref-sentseq/lr3e-4-ep40:pref-ce/default",
    "v1:pref-bt/default:pref-ce/default",
    "v1:pref-sentseq/lr1e-4-ep20:pref-sentseq/lr1e-4-ep40",
    "v2b:pref-sentseq/lr3e-4-ep40:pref-bt/default",
    "v2b:pref-sentseq/lr1e-4-ep40:pref-sentseq/lr3e-4-ep40",
    "v2b:pref-sentseq/lr3e-4-ep40:pref-ce/beyond-para",
    "v2c:pref-sentseq/lr3e-4-ep40:pref-bt/default",
    "v2c:pref-sentseq/lr1e-4-ep40:pref-sentseq/lr3e-4-ep40",
  ]
  compares = args.compare or default_compares

  accuracy_table: dict = {}
  for (eval_set, mkey), items in sorted(top1.items()):
    k = sum(items.values())
    n = len(items)
    lo, hi = wilson_ci(k, n)
    pair_map = pair[(eval_set, mkey)]
    pa = sum(v[0] for v in pair_map.values())
    pt = sum(v[1] for v in pair_map.values())
    accuracy_table[f"{eval_set}/{mkey}"] = {
      "evaluation_set": eval_set,
      "model": mkey,
      "n_items": n,
      "top1_correct": k,
      "top1_accuracy": k / n,
      "wilson_95_lower": lo,
      "wilson_95_upper": hi,
      "pairwise_agree": pa,
      "pairwise_total": pt,
      "pairwise_accuracy": (pa / pt) if pt else None,
    }
    print(
      f"{eval_set}/{mkey}: top1 {k}/{n}={k/n:.3f} "
      f"wilson95=[{lo:.3f},{hi:.3f}] pairwise {pa}/{pt}"
    )

  mcnemar_rows: list[dict] = []
  boot_rows: list[dict] = []

  for spec in compares:
    parts = spec.split(":")
    if len(parts) != 3:
      raise SystemExit(f"bad --compare {spec!r}; want eval:model/config:model/config")
    eval_set, ma, mb = parts
    key_a = (eval_set, ma)
    key_b = (eval_set, mb)
    if key_a not in top1 or key_b not in top1:
      print(f"SKIP missing model for {spec}")
      continue
    ids_a = set(top1[key_a])
    ids_b = set(top1[key_b])
    common = sorted(ids_a & ids_b)
    excluded = len(ids_a | ids_b) - len(common)
    if not common:
      print(f"SKIP empty common items for {spec}")
      continue
    if ids_a != ids_b:
      print(f"NOTE {spec}: excluded non-overlapping items={excluded}")

    ya = np.asarray([top1[key_a][i] for i in common], dtype=np.float64)
    yb = np.asarray([top1[key_b][i] for i in common], dtype=np.float64)
    both_correct = int(((ya == 1) & (yb == 1)).sum())
    a_only = int(((ya == 1) & (yb == 0)).sum())
    b_only = int(((ya == 0) & (yb == 1)).sum())
    both_wrong = int(((ya == 0) & (yb == 0)).sum())
    p_exact = exact_mcnemar(a_only, b_only)
    acc_a = float(ya.mean())
    acc_b = float(yb.mean())
    mcnemar_rows.append(
      {
        "evaluation_set": eval_set,
        "model_a": ma,
        "model_b": mb,
        "n_items": len(common),
        "n_excluded_nonoverlap": excluded,
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "p_value_exact": p_exact,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_difference": acc_a - acc_b,
        "metric": "top1",
      }
    )

    boot = paired_bootstrap_top1(ya, yb, n_boot=args.n_bootstrap, seed=args.seed)
    boot_rows.append(
      {
        "evaluation_set": eval_set,
        "model_a": ma,
        "model_b": mb,
        "n_items": len(common),
        "metric": "top1_accuracy_difference",
        "random_seed": args.seed,
        **boot,
      }
    )

    # pairwise（項目 cluster bootstrap）
    pa = np.asarray([pair[key_a][i][0] for i in common], dtype=np.float64)
    pta = np.asarray([pair[key_a][i][1] for i in common], dtype=np.float64)
    pb = np.asarray([pair[key_b][i][0] for i in common], dtype=np.float64)
    ptb = np.asarray([pair[key_b][i][1] for i in common], dtype=np.float64)
    boot_p = paired_bootstrap_pairwise_rate(
      pa, pta, pb, ptb, n_boot=args.n_bootstrap, seed=args.seed + 1
    )
    boot_rows.append(
      {
        "evaluation_set": eval_set,
        "model_a": ma,
        "model_b": mb,
        "n_items": len(common),
        "metric": "pairwise_agreement_rate_difference_item_cluster",
        "random_seed": args.seed + 1,
        **boot_p,
      }
    )
    print(
      f"McNemar {spec}: b={a_only} c={b_only} p_exact={p_exact:.4g} "
      f"acc_diff={acc_a - acc_b:+.3f}"
    )

  write_csv(
    args.mcnemar_out,
    mcnemar_rows,
    [
      "evaluation_set",
      "model_a",
      "model_b",
      "metric",
      "n_items",
      "n_excluded_nonoverlap",
      "both_correct",
      "a_only_correct",
      "b_only_correct",
      "both_wrong",
      "p_value_exact",
      "accuracy_a",
      "accuracy_b",
      "accuracy_difference",
    ],
  )
  write_csv(
    args.bootstrap_out,
    boot_rows,
    [
      "evaluation_set",
      "model_a",
      "model_b",
      "metric",
      "n_items",
      "observed_accuracy_difference",
      "bootstrap_mean_difference",
      "ci_95_lower",
      "ci_95_upper",
      "num_bootstrap_samples",
      "random_seed",
    ],
  )
  args.accuracy_out.parent.mkdir(parents=True, exist_ok=True)
  args.accuracy_out.write_text(
    json.dumps(accuracy_table, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"wrote {args.mcnemar_out}")
  print(f"wrote {args.bootstrap_out}")
  print(f"wrote {args.accuracy_out}")
  print(
    "NOTE: exploratory comparisons; no multiple-comparison correction applied "
    "as a primary result."
  )


if __name__ == "__main__":
  main()
