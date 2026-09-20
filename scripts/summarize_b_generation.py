#!/usr/bin/env python3
"""B 生成ブラインド判定の集計（9.1 節）。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from b_generation_common import eligible_sources, load_manifest, write_json

VALID_JUDGMENTS = frozenset({"left", "right", "tie", "unjudgeable"})
BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260907


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def xy_outcome(key: dict, judgment: str | None) -> str:
  if judgment is None:
    return "R"
  if judgment == "unjudgeable":
    return "U"
  if judgment == "tie":
    return "T"
  left_is_xy = key["left"]["condition"] == "xy"
  if judgment == "left":
    return "W" if left_is_xy else "L"
  if judgment == "right":
    return "W" if not left_is_xy else "L"
  raise ValueError(f"unknown judgment {judgment!r}")


def aggregate_counts(rows: list[tuple[str, str]]) -> dict:
  counts = {"W": 0, "L": 0, "T": 0, "U": 0, "R": 0}
  for _, outcome in rows:
    counts[outcome] += 1
  n = sum(counts.values())
  w, l, t, u, r = counts["W"], counts["L"], counts["T"], counts["U"], counts["R"]
  denom = w + l + t
  s = (w + 0.5 * t) / denom if denom else None
  decisive = w / (w + l) if (w + l) else None
  coverage = (w + l + t) / n if n else None
  band_low = (w + 0.5 * t) / n if n else None
  band_high = (w + 0.5 * t + u) / n if n else None
  return {
    "N": n,
    "W": w,
    "L": l,
    "T": t,
    "U": u,
    "R": r,
    "S": s,
    "decisive_win_rate": decisive,
    "coverage": coverage,
    "sensitivity_band": [band_low, band_high],
  }


def bootstrap_s_by_group(
  items: list[tuple[str, str, str]],
  *,
  n_iters: int,
  seed: int,
) -> dict:
  groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
  for group_id, comparison, outcome in items:
    groups[group_id].append((comparison, outcome))
  group_ids = sorted(groups)
  rng = random.Random(seed)
  s_vals: list[float] = []
  invalid = 0
  for _ in range(n_iters):
    sample_ids = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
    rows: list[tuple[str, str]] = []
    for gid in sample_ids:
      rows.extend(groups[gid])
    agg = aggregate_counts(rows)
    if agg["S"] is None:
      invalid += 1
    else:
      s_vals.append(float(agg["S"]))
  if len(s_vals) < 2:
    return {"ci95": None, "n_valid": len(s_vals), "n_invalid": invalid}
  s_vals.sort()
  lo = s_vals[int(0.025 * len(s_vals))]
  hi = s_vals[int(0.975 * len(s_vals))]
  return {"ci95": [lo, hi], "n_valid": len(s_vals), "n_invalid": invalid}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=Path("data/b_generation/manifest.json"))
  parser.add_argument("--pairs", type=Path, default=Path("data/b_generation/blind/pairs.jsonl"))
  parser.add_argument("--keys", type=Path, default=Path("data/b_generation/blind/keys.jsonl"))
  parser.add_argument("--judgments", type=Path, default=Path("data/b_generation/blind/judgments.jsonl"))
  parser.add_argument("--out", type=Path, default=Path("outputs/b_generation/results"))
  args = parser.parse_args()

  code_root = Path(__file__).resolve().parents[1]
  manifest = load_manifest(code_root / args.manifest)
  holdout_n = len(eligible_sources(manifest, split="holdout"))
  keys = load_jsonl(code_root / args.keys)
  judgments_path = code_root / args.judgments
  judgments_rows = load_jsonl(judgments_path) if judgments_path.is_file() else []
  judgment_by_pair: dict[str, str] = {}
  for row in judgments_rows:
    pid = str(row.get("pair_id") or "")
    j = str(row.get("judgment") or "")
    if pid in judgment_by_pair:
      raise SystemExit(f"duplicate judgment for pair_id {pid!r}")
    if j not in VALID_JUDGMENTS:
      raise SystemExit(f"invalid judgment {j!r} for pair_id {pid!r}")
    judgment_by_pair[pid] = j

  key_by_pair = {str(k["pair_id"]): k for k in keys}
  if len(key_by_pair) != len(keys):
    raise SystemExit("duplicate pair_id in keys")

  for pid in judgment_by_pair:
    if pid not in key_by_pair:
      raise SystemExit(f"judgment for unknown pair_id {pid!r}")

  by_comp_seed: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
  bootstrap_items: list[tuple[str, str, str]] = []
  finish_reasons: dict[str, int] = defaultdict(int)

  for key in keys:
    pid = str(key["pair_id"])
    comp = str(key["comparison"])
    seed = int(key["train_seed"])
    outcome = xy_outcome(key, judgment_by_pair.get(pid))
    by_comp_seed[(comp, seed)].append((pid, outcome))
    bootstrap_items.append((str(key["group_id"]), comp, outcome))
    for side in ("left", "right"):
      fr = key.get(f"{side}_finish_reason")
      if fr:
        finish_reasons[str(fr)] += 1

  per_comparison: dict = {}
  for comp in sorted({k[0] for k in by_comp_seed}):
    seed_stats = {}
    s_vals = []
    for seed in sorted({k[1] for k in by_comp_seed if k[0] == comp}):
      rows = by_comp_seed[(comp, seed)]
      if len(rows) != holdout_n:
        raise SystemExit(
          f"pair count mismatch comparison={comp} seed={seed}: got {len(rows)} expected {holdout_n}"
        )
      agg = aggregate_counts(rows)
      if agg["N"] != holdout_n or agg["W"] + agg["L"] + agg["T"] + agg["U"] + agg["R"] != holdout_n:
        raise SystemExit(f"count assert failed for {comp} seed={seed}")
      seed_stats[str(seed)] = agg
      if agg["S"] is not None:
        s_vals.append(float(agg["S"]))
    mean_s = sum(s_vals) / len(s_vals) if len(s_vals) == 3 else None
    if len(s_vals) != 3:
      mean_s = None
    per_comparison[comp] = {
      "by_seed": seed_stats,
      "mean_S_across_seeds": mean_s,
      "bootstrap_by_group": bootstrap_s_by_group(
        [x for x in bootstrap_items if x[1] == comp],
        n_iters=BOOTSTRAP_ITERS,
        seed=BOOTSTRAP_SEED,
      ),
    }

  incomplete = any(
    agg["R"] > 0 for comp in per_comparison.values() for agg in comp["by_seed"].values()
  )
  summary = {
    "holdout_n": holdout_n,
    "n_pairs": len(keys),
    "n_judgments": len(judgment_by_pair),
    "incomplete": incomplete,
    "finish_reason_counts": dict(finish_reasons),
    "comparisons": per_comparison,
    "judgments_path": str(judgments_path),
  }

  out_dir = code_root / args.out
  out_dir.mkdir(parents=True, exist_ok=True)
  write_json(out_dir / "summary.json", summary)

  lines = [
    "# B 条件付き生成 選好集計",
    "",
    f"holdout 原稿数: {holdout_n}",
    f"比較ペア数: {len(keys)}",
    f"判定済み: {len(judgment_by_pair)}",
    "",
  ]
  if incomplete:
    lines.append("未判定（R>0）があるため中間集計です。")
    lines.append("")
  for comp, block in per_comparison.items():
    lines.append(f"## {comp}")
    for seed, agg in block["by_seed"].items():
      s = agg["S"]
      s_txt = "null" if s is None else f"{s:.3f}"
      lines.append(
        f"- seed {seed}: S={s_txt} W={agg['W']} L={agg['L']} T={agg['T']} U={agg['U']} R={agg['R']} coverage={agg['coverage']:.3f}"
      )
    ms = block["mean_S_across_seeds"]
    lines.append(
      f"- 3seed 平均 S: {'null' if ms is None else f'{ms:.3f}'}"
    )
    lines.append("")
  (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps({"wrote": str(out_dir), "incomplete": incomplete}, ensure_ascii=False))


if __name__ == "__main__":
  main()
