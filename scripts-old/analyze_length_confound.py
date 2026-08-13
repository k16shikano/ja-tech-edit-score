#!/usr/bin/env python3
"""Hard Eval における候補長交絡の感度分析（探索的）。

長さ定義は score_hard_eval.py と同じく Python len(text)。
学習データでの長さ向きは pref_split/train のユニーク選好から推定し、
評価集合で向きを選ばない。両方向の length-only も併記する。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

EVAL_INPUTS = {
  "v1": ROOT / "data/hard_eval/bases_v1_labeled.jsonl",
  "v2b": ROOT / "data/hard_eval/bases_v2b_human_fable_copy.jsonl",
  "v2c": ROOT / "data/hard_eval/bases_v2c_human_machine_copy.jsonl",
}


def load_cand_csv(path: Path) -> list[dict]:
  with path.open(encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def train_length_direction(train_path: Path) -> dict:
  longer = shorter = equal = 0
  with train_path.open(encoding="utf-8") as handle:
    for line in handle:
      row = json.loads(line)
      if row.get("meta", {}).get("pair_order", "chosen_first") != "chosen_first":
        continue
      if int(row["label"]) != 1:
        continue
      la = len(row["candidate_a"])
      lb = len(row["candidate_b"])
      if la > lb:
        longer += 1
      elif la < lb:
        shorter += 1
      else:
        equal += 1
  n = longer + shorter + equal
  # 向き: 学習で多い方。同数なら報告のみ（評価で選ばない）
  preferred = "shorter" if shorter > longer else ("longer" if longer > shorter else "tie")
  return {
    "n_unique_pairs": n,
    "edit_longer": longer,
    "edit_shorter": shorter,
    "equal": equal,
    "preferred_from_train_majority": preferred,
  }


def spearman(xs: list[float], ys: list[float]) -> float | None:
  n = len(xs)
  if n < 2:
    return None

  def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    i = 0
    while i < n:
      j = i
      while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
        j += 1
      avg = (i + j) / 2.0 + 1.0
      for k in range(i, j + 1):
        out[order[k]] = avg
      i = j + 1
    return out

  rx, ry = ranks(xs), ranks(ys)
  mx = sum(rx) / n
  my = sum(ry) / n
  num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
  denx = float(np.sqrt(sum((a - mx) ** 2 for a in rx)))
  deny = float(np.sqrt(sum((b - my) ** 2 for b in ry)))
  if denx == 0 or deny == 0:
    return None
  return num / (denx * deny)


def top1_by_length(items: dict[str, list[dict]], *, prefer_longer: bool) -> tuple[int, int]:
  hits = 0
  n = 0
  for _item_id, cands in items.items():
    n += 1
    # 参照 top は reference_rank==1
    ref = [c for c in cands if int(c["reference_rank"]) == 1]
    if len(ref) != 1:
      raise SystemExit(f"bad reference_rank for item {_item_id}")
    if prefer_longer:
      pred = max(cands, key=lambda c: (int(c["char_length"]), c["candidate_role"]))
    else:
      pred = min(cands, key=lambda c: (int(c["char_length"]), c["candidate_role"]))
    if pred["candidate_role"] == ref[0]["candidate_role"]:
      hits += 1
  return hits, n


def residualize_within_item(cands: list[dict]) -> list[float]:
  """項目内中心化長さへの単回帰残差（評価集合自身で係数推定＝探索的）。"""
  y = np.asarray([float(c["model_score"]) for c in cands], dtype=np.float64)
  x = np.asarray([float(c["char_length"]) for c in cands], dtype=np.float64)
  x_c = x - x.mean()
  if np.allclose(x_c, 0):
    return list(y - y.mean())
  # OLS: y ~ 1 + x_c （切片は y.mean、傾きは共分散）
  beta = float(np.dot(x_c, y - y.mean()) / np.dot(x_c, x_c))
  return list(y - beta * x_c)


def top1_from_scores(cands: list[dict], scores: list[float]) -> bool:
  best_i = int(np.argmax(scores))
  ref = [i for i, c in enumerate(cands) if int(c["reference_rank"]) == 1][0]
  return best_i == ref


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
    "--cand-csv",
    type=Path,
    default=ROOT / "results/hard_eval_candidate_metadata.csv",
  )
  parser.add_argument(
    "--train-file",
    type=Path,
    default=ROOT / "data/pref_split/train.jsonl",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=ROOT / "results/length_control_results.csv",
  )
  parser.add_argument(
    "--detail-out",
    type=Path,
    default=ROOT / "results/raw/length_control_detail.json",
  )
  parser.add_argument(
    "--models",
    default="pref-sentseq/lr3e-4-ep40,pref-sentseq/lr1e-4-ep40,pref-bt/default",
    help="comma-separated model_name/model_config",
  )
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()

  for out in (args.out, args.detail_out):
    if out.exists() and not args.overwrite:
      raise SystemExit(f"exists (pass --overwrite): {out}")
  if not args.cand_csv.is_file():
    raise SystemExit(f"missing: {args.cand_csv}")
  if not args.train_file.is_file():
    raise SystemExit(f"missing: {args.train_file}")

  train_dir = train_length_direction(args.train_file)
  print("train length direction:", json.dumps(train_dir, ensure_ascii=False))

  rows = load_cand_csv(args.cand_csv)
  wanted = set(args.models.split(","))
  # group by eval, model, item
  grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
  for row in rows:
    mkey = f"{row['model_name']}/{row['model_config']}"
    if mkey not in wanted:
      continue
    grouped[(row["evaluation_set"], mkey, row["item_id"])].append(row)

  by_eval_model: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(dict)
  for (eval_set, mkey, item_id), cands in grouped.items():
    by_eval_model[(eval_set, mkey)][item_id] = cands

  out_rows: list[dict] = []
  detail: dict = {
    "exploratory": True,
    "char_length_definition": "Python len(text); same as score_hard_eval.py",
    "train_length_direction": train_dir,
    "notes": [
      "length_residualized uses within-item centered length OLS on the same "
      "evaluation set (exploratory; coefficients are not from a held-out set).",
      "Both longer-preferred and shorter-preferred length-only baselines are "
      "reported; train-majority preferred direction is also marked.",
    ],
    "per_item_spearman": {},
  }

  for (eval_set, mkey), items in sorted(by_eval_model.items()):
    # original top1
    orig_hits = 0
    for cands in items.values():
      scores = [float(c["model_score"]) for c in cands]
      if top1_from_scores(cands, scores):
        orig_hits += 1
    n_items = len(items)
    long_hits, _ = top1_by_length(items, prefer_longer=True)
    short_hits, _ = top1_by_length(items, prefer_longer=False)

    # residualized
    resid_hits = 0
    item_corrs: list[float] = []
    all_scores: list[float] = []
    all_lens: list[float] = []
    for item_id, cands in items.items():
      scores = [float(c["model_score"]) for c in cands]
      lens = [float(c["char_length"]) for c in cands]
      corr = spearman(scores, lens)
      if corr is not None:
        item_corrs.append(corr)
        detail["per_item_spearman"].setdefault(f"{eval_set}/{mkey}", {})[item_id] = corr
      all_scores.extend(scores)
      all_lens.extend(lens)
      resid = residualize_within_item(cands)
      if top1_from_scores(cands, resid):
        resid_hits += 1

    pooled = spearman(all_scores, all_lens)
    mean_c = float(np.mean(item_corrs)) if item_corrs else None
    med_c = float(np.median(item_corrs)) if item_corrs else None

    preferred = train_dir["preferred_from_train_majority"]
    train_pref_hits = short_hits if preferred == "shorter" else (
      long_hits if preferred == "longer" else ""
    )

    row = {
      "evaluation_set": eval_set,
      "model_name": mkey.split("/")[0],
      "model_config": mkey.split("/", 1)[1],
      "n_items": n_items,
      "original_model_top1": orig_hits,
      "original_model_top1_accuracy": orig_hits / n_items,
      "length_only_longer_top1": long_hits,
      "length_only_longer_top1_accuracy": long_hits / n_items,
      "length_only_shorter_top1": short_hits,
      "length_only_shorter_top1_accuracy": short_hits / n_items,
      "length_only_train_majority_direction": preferred,
      "length_only_train_majority_top1": train_pref_hits,
      "length_residualized_top1": resid_hits,
      "length_residualized_top1_accuracy": resid_hits / n_items,
      "item_spearman_mean": mean_c if mean_c is not None else "",
      "item_spearman_median": med_c if med_c is not None else "",
      "item_spearman_n": len(item_corrs),
      "pooled_spearman_all_candidates": pooled if pooled is not None else "",
      "analysis_type": "exploratory",
    }
    out_rows.append(row)
    print(
      f"{eval_set}/{mkey}: orig={orig_hits}/{n_items} "
      f"long={long_hits} short={short_hits} resid={resid_hits} "
      f"item_rho_med={med_c} pooled_rho={pooled}"
    )

  write_csv(
    args.out,
    out_rows,
    [
      "evaluation_set",
      "model_name",
      "model_config",
      "n_items",
      "original_model_top1",
      "original_model_top1_accuracy",
      "length_only_longer_top1",
      "length_only_longer_top1_accuracy",
      "length_only_shorter_top1",
      "length_only_shorter_top1_accuracy",
      "length_only_train_majority_direction",
      "length_only_train_majority_top1",
      "length_residualized_top1",
      "length_residualized_top1_accuracy",
      "item_spearman_mean",
      "item_spearman_median",
      "item_spearman_n",
      "pooled_spearman_all_candidates",
      "analysis_type",
    ],
  )
  args.detail_out.parent.mkdir(parents=True, exist_ok=True)
  args.detail_out.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n")
  print(f"wrote {args.out}")
  print(f"wrote {args.detail_out}")


if __name__ == "__main__":
  main()
