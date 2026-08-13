#!/usr/bin/env python3
"""節単位採掘データの構成差分統計。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True)
  parser.add_argument("--compare", help="optional hunk-level JSONL for contrast")
  args = parser.parse_args()

  rows = load_rows(Path(args.input))
  n = len(rows)
  if n == 0:
    print("no rows")
    return

  blank_para = 0
  para_changed = 0
  line_changed = 0
  multi_para = 0
  lens_src: list[int] = []
  lens_ed: list[int] = []

  for row in rows:
    src = row.get("source_text") or ""
    ed = row.get("edited_text") or ""
    meta = row.get("meta") or {}
    ps = meta.get("paragraph_count_source")
    pe = meta.get("paragraph_count_edited")
    if ps is None:
      ps = len([p for p in src.split("\n\n") if p.strip()])
    if pe is None:
      pe = len([p for p in ed.split("\n\n") if p.strip()])
    ls = meta.get("line_count_source")
    le = meta.get("line_count_edited")
    if ls is None:
      ls = len([line for line in src.splitlines() if line.strip()])
    if le is None:
      le = len([line for line in ed.splitlines() if line.strip()])

    if "\n\n" in src or "\n\n" in ed:
      blank_para += 1
    if ps > 1 or pe > 1:
      multi_para += 1
    if ps != pe:
      para_changed += 1
    if ls != le:
      line_changed += 1
    lens_src.append(len(src))
    lens_ed.append(len(ed))

  print(f"rows: {n}")
  print(f"with blank-line paragraph boundary: {blank_para}/{n} ({blank_para/n:.1%})")
  print(f"multi-paragraph section (either side): {multi_para}/{n} ({multi_para/n:.1%})")
  print(f"paragraph count changed: {para_changed}/{n} ({para_changed/n:.1%})")
  print(f"line count changed: {line_changed}/{n} ({line_changed/n:.1%})")
  print(
    "source chars: mean",
    int(statistics.mean(lens_src)),
    "median",
    int(statistics.median(lens_src)),
    "p90",
    sorted(lens_src)[int(n * 0.9)],
  )
  print(
    "edited chars: mean",
    int(statistics.mean(lens_ed)),
    "median",
    int(statistics.median(lens_ed)),
    "p90",
    sorted(lens_ed)[int(n * 0.9)],
  )

  if args.compare:
    cmp_rows = load_rows(Path(args.compare))
    cmp_blank = sum(
      1
      for row in cmp_rows
      if "\n\n" in (row.get("source_text") or "") or "\n\n" in (row.get("edited_text") or "")
    )
    print("")
    print(f"compare hunk rows: {len(cmp_rows)}")
    print(f"compare blank-line pairs: {cmp_blank}/{len(cmp_rows)} ({cmp_blank/len(cmp_rows):.1%})")


if __name__ == "__main__":
  main()
