#!/usr/bin/env python3
"""D 800 行からペア台帳を作る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pref_d_interval_utils import load_d_rows
from pref_d_pair_utils import build_pair_ledger, write_json


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, default=Path("data/d/dataset.jsonl"))
  parser.add_argument("--out-ledger", type=Path, default=Path("data/d/pair_ledger.jsonl"))
  parser.add_argument("--out-stats", type=Path, default=Path("data/d/pair_ledger_stats.json"))
  args = parser.parse_args()

  code_root = Path(__file__).resolve().parents[1]
  dataset_path = code_root / args.dataset
  rows = load_d_rows(dataset_path)
  if len(rows) != 800:
    raise SystemExit(f"expected 800 rows in {dataset_path}, got {len(rows)}")

  ledger, summary = build_pair_ledger(rows)
  out_ledger = code_root / args.out_ledger
  out_stats = code_root / args.out_stats
  out_ledger.parent.mkdir(parents=True, exist_ok=True)
  out_ledger.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in ledger) + ("\n" if ledger else ""),
    encoding="utf-8",
  )
  write_json(out_stats, summary)
  print(json.dumps({"ledger": str(out_ledger), **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
  main()
