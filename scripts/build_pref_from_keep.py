#!/usr/bin/env python3
"""レビュー済み keep から選好学習用 JSONL を作る。

教師は推敲前後そのもの:
  source_text = 下書き
  candidate_a = 推敲後（良い方）
  candidate_b = 下書き（悪い方）
  label = 1

分割は keep 側の meta.split を維持する。
  train   → 学習
  heldout → 検証（ペア単位層化分割後の heldout を含む）

旧 pref_dataset / pref_split は上書きしない。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def corpus_split(rec: dict) -> str:
  meta = rec.get("meta") or {}
  split = str(meta.get("split") or "").strip()
  if split == "heldout":
    return "valid"
  if split == "train":
    return "train"
  raise SystemExit(f"missing meta.split in {rec.get('id')}")


def to_pref_rows(rec: dict, *, augment_swap: bool) -> list[dict]:
  source = str(rec.get("source_text") or "")
  edited = str(rec.get("edited_text") or "")
  if not source.strip() or not edited.strip():
    return []
  if source == edited:
    return []

  base_id = str(rec.get("id") or "")
  project_id = str(rec.get("project_id") or "")
  labels = list(rec.get("labels") or [])
  if "human_reviewed" not in labels:
    labels.append("human_reviewed")
  unit = str(rec.get("unit") or "")
  if unit and unit not in labels:
    labels.append(unit)

  meta_common = {
    "project_id": project_id,
    "source_reference": str(rec.get("source_reference") or ""),
    "labels": labels,
    "created_at": str(rec.get("created_at") or ""),
    "base_id": base_id,
    "unit": unit,
    "corpus_source": str(rec.get("corpus_source") or ""),
    "quality": str(rec.get("quality") or "keep"),
    "corpus_split": str((rec.get("meta") or {}).get("split") or ""),
  }

  rows = [
    {
      "id": base_id,
      "source_text": source,
      "candidate_a": edited,
      "candidate_b": source,
      "label": 1,
      "meta": {**meta_common, "pair_order": "chosen_first"},
    }
  ]
  if augment_swap:
    rows.append(
      {
        "id": base_id + "-swap",
        "source_text": source,
        "candidate_a": source,
        "candidate_b": edited,
        "label": 0,
        "meta": {**meta_common, "pair_order": "rejected_first"},
      }
    )
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    default="data/revision_corpus/canonical.jsonl",
    help="keep 正本（source_text / edited_text）",
  )
  parser.add_argument(
    "--out-dataset",
    default="data/pref_keep/dataset.jsonl",
    help="swap 込みの選好 JSONL",
  )
  parser.add_argument(
    "--out-split-dir",
    default="data/pref_keep_split",
    help="train.jsonl / valid.jsonl を書くディレクトリ",
  )
  parser.add_argument(
    "--report",
    default="data/pref_keep/build_report.json",
  )
  parser.add_argument(
    "--units",
    default="",
    help="含める unit（カンマ区切り。空ならすべて。例: section または hunk）",
  )
  parser.add_argument("--no-swap", action="store_true")
  args = parser.parse_args()

  in_path = Path(args.input)
  if not in_path.is_file():
    raise SystemExit(f"missing {in_path}")

  unit_filter = {u.strip() for u in args.units.split(",") if u.strip()}
  records = load_jsonl(in_path)
  augment_swap = not args.no_swap

  by_split: dict[str, list[dict]] = {"train": [], "valid": []}
  skipped = Counter()
  unit_counts = Counter()
  project_counts: dict[str, Counter] = {
    "train": Counter(),
    "valid": Counter(),
  }

  for rec in records:
    if str(rec.get("quality") or "keep") != "keep":
      skipped["not_keep"] += 1
      continue
    unit = str(rec.get("unit") or "")
    if unit_filter and unit not in unit_filter:
      skipped["unit_filter"] += 1
      continue
    try:
      split = corpus_split(rec)
    except SystemExit:
      skipped["no_split"] += 1
      continue
    rows = to_pref_rows(rec, augment_swap=augment_swap)
    if not rows:
      skipped["empty_or_identical"] += 1
      continue
    by_split[split].extend(rows)
    unit_counts[unit or "(none)"] += 1
    project_counts[split][str(rec.get("project_id") or "?")] += 1

  if not by_split["train"] or not by_split["valid"]:
    raise SystemExit(
      f"empty split after filter: train={len(by_split['train'])} "
      f"valid={len(by_split['valid'])}"
    )

  out_dataset = Path(args.out_dataset)
  out_split = Path(args.out_split_dir)
  out_dataset.parent.mkdir(parents=True, exist_ok=True)
  out_split.mkdir(parents=True, exist_ok=True)

  all_rows = by_split["train"] + by_split["valid"]
  with out_dataset.open("w", encoding="utf-8") as f:
    for row in all_rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  for name, rows in by_split.items():
    with (out_split / f"{name}.jsonl").open("w", encoding="utf-8") as f:
      for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

  # BT 学習は chosen_first だけ使うので、ユニーク対の数も報告する
  def n_unique(rows: list[dict]) -> int:
    return sum(
      1
      for r in rows
      if (r.get("meta") or {}).get("pair_order", "chosen_first") == "chosen_first"
      and int(r["label"]) == 1
    )

  report = {
    "input": str(in_path),
    "out_dataset": str(out_dataset),
    "out_split_dir": str(out_split),
    "augment_swap": augment_swap,
    "unit_filter": sorted(unit_filter),
    "n_dataset_rows": len(all_rows),
    "n_train_rows": len(by_split["train"]),
    "n_valid_rows": len(by_split["valid"]),
    "n_train_pairs": n_unique(by_split["train"]),
    "n_valid_pairs": n_unique(by_split["valid"]),
    "pairs_by_unit": dict(unit_counts),
    "train_projects": dict(project_counts["train"]),
    "valid_projects": dict(project_counts["valid"]),
    "skipped": dict(skipped),
    "note": (
      "valid は keep の heldout（meta.split）。"
      "ペア単位層化分割後は書籍単位ではない。"
    ),
  }
  report_path = Path(args.report)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )

  print(json.dumps(report, ensure_ascii=False, indent=2))
  print(f"wrote {out_dataset}")
  print(f"wrote {out_split}/train.jsonl ({len(by_split['train'])})")
  print(f"wrote {out_split}/valid.jsonl ({len(by_split['valid'])})")
  print(f"wrote {report_path}")


if __name__ == "__main__":
  main()
