#!/usr/bin/env python3
"""Hard Eval マージン分析 JSON を2本並べて比較表を出す。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
  data = json.loads(path.read_text(encoding="utf-8"))
  position_key = next(
    (k for k in data if k.endswith("_position") and isinstance(data[k], dict)),
    None,
  )
  return {
    "path": str(path),
    "model": data.get("model"),
    "n": data.get("n"),
    "pair_stats": data.get("pair_stats") or {},
    "position_key": position_key or "fable_position",
    "position": data.get(position_key) or {} if position_key else {},
  }


def fmt(v: object) -> str:
  if isinstance(v, float):
    if v != v:
      return "nan"
    return f"{v:.3f}"
  return str(v)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline", required=True, help="現行モデルの margins JSON")
  parser.add_argument("--variant", required=True, help="比較対象モデルの margins JSON")
  parser.add_argument("--out", help="Markdown 出力先")
  args = parser.parse_args()

  baseline = load_summary(Path(args.baseline))
  variant = load_summary(Path(args.variant))

  pair_keys = sorted(set(baseline["pair_stats"]) | set(variant["pair_stats"]))
  lines = [
    "# Hard Eval マージン比較",
    "",
    f"- baseline: `{baseline['path']}` (model={baseline.get('model')}, n={baseline.get('n')})",
    f"- variant: `{variant['path']}` (model={variant.get('model')}, n={variant.get('n')})",
    "",
    "## ペア別点差",
    "",
    "| pair | metric | baseline | variant | delta |",
    "|---|---|---:|---:|---:|",
  ]

  for pair in pair_keys:
    b = baseline["pair_stats"].get(pair, {})
    v = variant["pair_stats"].get(pair, {})
    for metric in ("win_rate", "mean_margin", "median_margin", "min_margin"):
      bv = b.get(metric, float("nan"))
      vv = v.get(metric, float("nan"))
      if isinstance(bv, (int, float)) and isinstance(vv, (int, float)) and bv == bv and vv == vv:
        delta = vv - bv
      else:
        delta = float("nan")
      lines.append(
        f"| {pair} | {metric} | {fmt(bv)} | {fmt(vv)} | {fmt(delta)} |"
      )

  bfp = baseline["position"]
  vfp = variant["position"]
  lines.extend(
    [
      "",
      f"## {baseline['position_key']} (0=copy, 1=human)",
      "",
      "| metric | baseline | variant | delta |",
      "|---|---:|---:|---:|",
    ]
  )
  for metric in ("mean", "median", "between_0_1"):
    bv = bfp.get(metric, float("nan"))
    vv = vfp.get(metric, float("nan"))
    if isinstance(bv, (int, float)) and isinstance(vv, (int, float)) and bv == bv and vv == vv:
      delta = vv - bv
    else:
      delta = float("nan")
    lines.append(f"| {metric} | {fmt(bv)} | {fmt(vv)} | {fmt(delta)} |")

  text = "\n".join(lines) + "\n"
  print(text, end="")
  if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
  main()
