#!/usr/bin/env python3
"""Phase 4: 合成パイプライン。TAU_LEVEL の校準と記録。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import load_jsonl, repo_root, write_tau_level_record


def load_predictions(path: Path) -> list[dict[str, Any]]:
  if path.suffix == ".jsonl":
    return load_jsonl(path)
  obj = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(obj, list):
    return obj
  if isinstance(obj, dict) and isinstance(obj.get("items"), list):
    return obj["items"]
  raise SystemExit(f"unsupported predictions format: {path}")


def score_tau_on_d(rows: list[dict[str, Any]], tau: float) -> dict[str, float]:
  by_pos: dict[str, list[bool]] = {"a": [], "b": [], "c": [], "d": []}
  for row in rows:
    pos = str(row.get("position") or row.get("label") or "").lower()
    if pos not in by_pos:
      continue
    p = row.get("p_better")
    if p is None:
      continue
    by_pos[pos].append(float(p) >= tau)
  out: dict[str, float] = {}
  for pos, flags in by_pos.items():
    if flags:
      out[f"pass_rate_{pos}"] = sum(flags) / len(flags)
  if by_pos["a"]:
    out["a_reject_rate"] = 1.0 - out.get("pass_rate_a", 0.0)
  return out


def score_tau_on_b(rows: list[dict[str, Any]], tau: float) -> dict[str, float]:
  human_pass = 0
  human_n = 0
  composer_pass = 0
  composer_n = 0
  for row in rows:
    role = str(row.get("role") or row.get("source") or "").lower()
    p = row.get("p_better")
    if p is None:
      continue
    passed = float(p) >= tau
    if role in ("human", "gold"):
      human_n += 1
      human_pass += int(passed)
    elif role == "composer":
      composer_n += 1
      composer_pass += int(passed)
  out: dict[str, float] = {}
  if human_n:
    out["human_pass_rate"] = human_pass / human_n
  if composer_n:
    out["composer_pass_rate"] = composer_pass / composer_n
  return out


def pick_tau_level(
  d_rows: list[dict[str, Any]],
  b_rows: list[dict[str, Any]],
  *,
  grid_step: float = 0.01,
) -> tuple[float, dict[str, Any]]:
  best_tau = 0.5
  best_score = float("-inf")
  best_metrics: dict[str, Any] = {}
  taus = [round(i * grid_step, 4) for i in range(1, int(1.0 / grid_step))]
  for tau in taus:
    d_m = score_tau_on_d(d_rows, tau)
    b_m = score_tau_on_b(b_rows, tau)
    a_reject = d_m.get("a_reject_rate", 0.0)
    d_pass = d_m.get("pass_rate_d", 0.0)
    human_pass = b_m.get("human_pass_rate", 0.0)
    score = a_reject + d_pass + human_pass
    if score > best_score:
      best_score = score
      best_tau = tau
      best_metrics = {"d_valid": d_m, "b_valid": b_m, "objective": score}
  return best_tau, best_metrics


def calibrate_tau_level(
  *,
  gate_level_checkpoint: Path,
  predictions: Path,
  root: Path,
  den_adapter_checkpoint: Path | None = None,
) -> dict[str, Any]:
  rows = load_predictions(predictions)
  d_rows = [r for r in rows if str(r.get("split") or r.get("set") or "") in ("d_valid", "d")]
  b_rows = [r for r in rows if str(r.get("split") or r.get("set") or "") in ("b_valid", "b", "pref_valid")]
  if not d_rows or not b_rows:
    raise SystemExit(
      f"predictions must include d_valid and b_valid rows (got d={len(d_rows)} b={len(b_rows)})"
    )
  tau, metrics = pick_tau_level(d_rows, b_rows)
  out_path = write_tau_level_record(
    tau_level=tau,
    gate_level_checkpoint=gate_level_checkpoint,
    root=root,
    den_adapter_checkpoint=den_adapter_checkpoint,
    calibration_sets=["d_valid", "b_valid"],
    method="grid_max_a_reject_d_pass_human_pass",
    metrics=metrics,
  )
  return {"tau_level": tau, "wrote": str(out_path), "metrics": metrics}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)

  cal = sub.add_parser("calibrate-tau", help="D valid + B valid から TAU_LEVEL を決め work に記録")
  cal.add_argument("--root", type=Path, default=repo_root())
  cal.add_argument(
    "--gate-level-checkpoint",
    type=Path,
    required=True,
    help="G2 累積リンク MLP のチェックポイント（識別子として tau_level.json に残す）",
  )
  cal.add_argument(
    "--predictions",
    type=Path,
    required=True,
    help="p_better 付きの json/jsonl（d_valid と b_valid の行を含む）",
  )
  cal.add_argument(
    "--den-adapter-checkpoint",
    type=Path,
    default=None,
    help="凍結した密度アダプタ（任意。記録用）",
  )

  args = parser.parse_args()
  if args.cmd == "calibrate-tau":
    result = calibrate_tau_level(
      gate_level_checkpoint=args.gate_level_checkpoint,
      predictions=args.predictions,
      root=args.root,
      den_adapter_checkpoint=args.den_adapter_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
  main()
