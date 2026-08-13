#!/usr/bin/env python3
"""ブラインド判定 360 件を、生成文人手選好の開発用教師データへ変換する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generated_pref_utils import (
  assign_item_folds,
  build_fold_splits,
  load_jsonl,
  merge_blind_to_teacher_rows,
  summarize_teacher_rows,
  verify_fold_splits,
  write_jsonl,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/blind_eval/pairs.jsonl")
  parser.add_argument("--judgments", default="data/blind_eval/judgments.jsonl")
  parser.add_argument("--items", default="data/blind_eval/items.jsonl")
  parser.add_argument("--out-dir", default="data/generated_pref_experiment")
  parser.add_argument("--n-folds", type=int, default=5)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  pairs = load_jsonl(args.pairs)
  judgments = load_jsonl(args.judgments)
  items = load_jsonl(args.items)
  if not pairs or not judgments or not items:
    raise SystemExit("pairs/judgments/items のいずれかが空です")

  rows = merge_blind_to_teacher_rows(pairs, judgments, items)
  if len(rows) != len(pairs):
    raise SystemExit(f"row count mismatch: merged={len(rows)} pairs={len(pairs)}")

  out_dir = Path(args.out_dir)
  write_jsonl(out_dir / "all.jsonl", rows)

  section_rows = [r for r in rows if r.get("unit") == "section"]
  hunk_rows = [r for r in rows if r.get("unit") == "hunk"]
  write_jsonl(out_dir / "section" / "all.jsonl", section_rows)
  write_jsonl(out_dir / "hunk" / "all.jsonl", hunk_rows)

  item_to_fold = assign_item_folds(items, n_folds=args.n_folds, seed=args.seed)
  all_folds = build_fold_splits(rows, item_to_fold, n_folds=args.n_folds)
  verify_fold_splits(all_folds, n_folds=args.n_folds)

  section_folds = build_fold_splits(section_rows, item_to_fold, n_folds=args.n_folds)
  verify_fold_splits(section_folds, n_folds=args.n_folds)

  for fold_idx, (train_rows, valid_rows) in enumerate(all_folds):
    fold_dir = out_dir / "folds" / f"fold_{fold_idx}"
    write_jsonl(fold_dir / "train.jsonl", train_rows)
    write_jsonl(fold_dir / "valid.jsonl", valid_rows)

  for fold_idx, (train_rows, valid_rows) in enumerate(section_folds):
    fold_dir = out_dir / "folds_section" / f"fold_{fold_idx}"
    write_jsonl(fold_dir / "train.jsonl", train_rows)
    write_jsonl(fold_dir / "valid.jsonl", valid_rows)

  hunk_folds = build_fold_splits(hunk_rows, item_to_fold, n_folds=args.n_folds)
  for fold_idx, (train_rows, valid_rows) in enumerate(hunk_folds):
    fold_dir = out_dir / "folds_hunk" / f"fold_{fold_idx}"
    write_jsonl(fold_dir / "train.jsonl", train_rows)
    write_jsonl(fold_dir / "valid.jsonl", valid_rows)

  report = {
    "pairs": len(pairs),
    "judgments": len(judgments),
    "items": len(items),
    "all": summarize_teacher_rows(rows),
    "section": summarize_teacher_rows(section_rows),
    "hunk": summarize_teacher_rows(hunk_rows),
    "n_folds": args.n_folds,
    "seed": args.seed,
    "fold_valid_items": [
      len({str(r.get("item_id") or "") for r in valid_rows})
      for _, valid_rows in section_folds
    ],
    "note": (
      "主実験は folds_section/ の section 行のみ。"
      "hunk は folds_hunk/ に分離保存し、主モデルへ混ぜない。"
    ),
  }
  report_path = out_dir / "report.json"
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  print(json.dumps(report, ensure_ascii=False, indent=2))
  print(f"wrote {out_dir}/all.jsonl ({len(rows)})")
  print(f"wrote {report_path}")


if __name__ == "__main__":
  main()
