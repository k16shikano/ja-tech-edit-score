#!/usr/bin/env python3
"""学習用データ D（800 行）を組み立てる。"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from a1_probe_interval_utils import write_jsonl
from a1_probe_position_server import load_samples, pair_id_of
from pref_static_utils import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_DIR = ROOT / "outputs" / "a1-probe"
DEFAULT_JUDGMENTS = DEFAULT_SAMPLES_DIR / "position_judgments.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "d"

MODE_TO_Y_KIND = {
  "base": "qwen_base",
  "base_norms": "qwen_base_norms",
  "adapter": "qwen_adapter",
}
INPUT_POSITIONS = frozenset({"a", "eq", "b", "c", "d"})
OUTPUT_POSITIONS = frozenset({"a", "b", "c", "d"})


def normalize_position(position: str) -> str:
  if position == "eq":
    return "b"
  if position not in OUTPUT_POSITIONS:
    raise ValueError(f"unknown position: {position!r}")
  return position


def load_generated_rows(samples_dir: Path, judgments_path: Path) -> list[dict]:
  samples = load_samples(samples_dir)
  rows: list[dict] = []
  for judgment in load_jsonl(judgments_path):
    pair_id = str(judgment.get("pair_id") or "").strip()
    item_id = str(judgment.get("item_id") or "").strip()
    mode = str(judgment.get("mode") or "").strip()
    raw_position = str(judgment.get("position") or "").strip()
    if not pair_id or raw_position not in INPUT_POSITIONS:
      continue
    position = normalize_position(raw_position)
    if mode not in MODE_TO_Y_KIND:
      raise ValueError(f"unknown mode in judgment: {pair_id}")
    key = (item_id, mode)
    if key not in samples:
      raise KeyError(f"missing sample for {pair_id}")
    sample = samples[key]
    draft = str(sample.get("draft") or "")
    gold = str(sample.get("gold") or "")
    y = str(sample.get("generated") or "")
    rows.append(
      {
        "row_id": pair_id,
        "item_id": item_id,
        "y_kind": MODE_TO_Y_KIND[mode],
        "draft": draft,
        "y": y,
        "position": position,
        "y_eq_human_edit": y.strip() == gold.strip(),
      }
    )
  return rows


def load_human_rows(samples_dir: Path, item_ids: set[str]) -> list[dict]:
  samples = load_samples(samples_dir)
  rows: list[dict] = []
  for item_id in sorted(item_ids):
    sample = None
    for mode in MODE_TO_Y_KIND:
      key = (item_id, mode)
      if key in samples:
        sample = samples[key]
        break
    if sample is None:
      raise KeyError(f"missing sample for item_id={item_id}")
    gold = str(sample.get("gold") or "")
    rows.append(
      {
        "row_id": f"{item_id}::human",
        "item_id": item_id,
        "y_kind": "human",
        "draft": str(sample.get("draft") or ""),
        "y": gold,
        "position": "d",
        "y_eq_human_edit": True,
      }
    )
  return rows


def split_by_item(
  rows: list[dict],
  *,
  eval_fraction: float,
  seed: int,
) -> tuple[list[dict], list[dict], set[str], dict]:
  item_ids = sorted({r["item_id"] for r in rows})
  rng = random.Random(seed)
  shuffled = list(item_ids)
  rng.shuffle(shuffled)
  n_eval = max(1, int(round(len(shuffled) * eval_fraction)))
  eval_ids = set(shuffled[:n_eval])
  train = [r for r in rows if r["item_id"] not in eval_ids]
  valid = [r for r in rows if r["item_id"] in eval_ids]
  stats = {
    "n_total": len(rows),
    "n_items": len(item_ids),
    "n_train": len(train),
    "n_valid": len(valid),
    "n_valid_items": len(eval_ids),
    "split": f"item_fraction={eval_fraction}",
    "seed": seed,
    "position_counts_total": dict(Counter(r["position"] for r in rows)),
    "y_kind_counts_total": dict(Counter(r["y_kind"] for r in rows)),
    "position_counts_train": dict(Counter(r["position"] for r in train)),
    "position_counts_valid": dict(Counter(r["position"] for r in valid)),
  }
  return train, valid, eval_ids, stats


def build_kfolds(
  rows: list[dict],
  *,
  n_folds: int,
  seed: int,
) -> tuple[list[dict], dict]:
  item_ids = sorted({r["item_id"] for r in rows})
  if len(item_ids) % n_folds != 0:
    raise ValueError(f"cannot split {len(item_ids)} items into {n_folds} folds evenly")
  rng = random.Random(seed)
  shuffled = list(item_ids)
  rng.shuffle(shuffled)
  fold_size = len(shuffled) // n_folds
  assignments: list[dict] = []
  fold_stats: list[dict] = []
  for fold_idx in range(n_folds):
    start = fold_idx * fold_size
    valid_ids = set(shuffled[start : start + fold_size])
    for item_id in sorted(valid_ids):
      assignments.append({"item_id": item_id, "fold": fold_idx})
    valid = [r for r in rows if r["item_id"] in valid_ids]
    train = [r for r in rows if r["item_id"] not in valid_ids]
    fold_stats.append(
      {
        "fold": fold_idx,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_valid_items": len(valid_ids),
        "position_counts_valid": dict(Counter(r["position"] for r in valid)),
      }
    )
  stats = {
    "n_folds": n_folds,
    "n_items": len(item_ids),
    "fold_size": fold_size,
    "seed": seed,
    "folds": fold_stats,
  }
  return assignments, stats


def build_dataset(
  *,
  samples_dir: Path,
  judgments_path: Path,
  out_dir: Path,
  eval_fraction: float,
  seed: int,
) -> dict:
  generated = load_generated_rows(samples_dir, judgments_path)
  item_ids = {r["item_id"] for r in generated}
  human = load_human_rows(samples_dir, item_ids)
  all_rows = generated + human
  if len(generated) != 600:
    raise ValueError(f"expected 600 generated rows, got {len(generated)}")
  if len(human) != 200:
    raise ValueError(f"expected 200 human rows, got {len(human)}")
  if len(all_rows) != 800:
    raise ValueError(f"expected 800 rows, got {len(all_rows)}")

  train, valid, eval_item_ids, stats = split_by_item(
    all_rows,
    eval_fraction=eval_fraction,
    seed=seed,
  )

  out_dir.mkdir(parents=True, exist_ok=True)
  write_jsonl(out_dir / "dataset.jsonl", all_rows)
  write_jsonl(out_dir / "train.jsonl", train)
  write_jsonl(out_dir / "valid.jsonl", valid)
  with (out_dir / "valid_item_ids.jsonl").open("w", encoding="utf-8") as f:
    for item_id in sorted(eval_item_ids):
      f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")
  (out_dir / "split_stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )

  folds_dir = out_dir / "folds"
  folds_dir.mkdir(parents=True, exist_ok=True)
  assignments, fold_stats = build_kfolds(all_rows, n_folds=5, seed=seed)
  write_jsonl(folds_dir / "fold_assignments.jsonl", assignments)
  item_to_fold = {a["item_id"]: a["fold"] for a in assignments}
  for fold_idx in range(5):
    valid_ids = {item_id for item_id, fold in item_to_fold.items() if fold == fold_idx}
    fold_train = [r for r in all_rows if r["item_id"] not in valid_ids]
    fold_valid = [r for r in all_rows if r["item_id"] in valid_ids]
    write_jsonl(folds_dir / f"fold{fold_idx}_train.jsonl", fold_train)
    write_jsonl(folds_dir / f"fold{fold_idx}_valid.jsonl", fold_valid)
  (folds_dir / "fold_stats.json").write_text(
    json.dumps(fold_stats, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )

  return {**stats, "folds": fold_stats}


def main() -> None:
  parser = argparse.ArgumentParser(description="Build teacher dataset D (800 rows).")
  parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
  parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--eval-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  stats = build_dataset(
    samples_dir=args.samples_dir,
    judgments_path=args.judgments,
    out_dir=args.out_dir,
    eval_fraction=args.eval_fraction,
    seed=args.seed,
  )
  print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
