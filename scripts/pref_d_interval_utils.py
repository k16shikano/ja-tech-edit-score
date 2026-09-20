#!/usr/bin/env python3
"""学習用データ D の区間損失と評価指標。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from a1_probe_interval_utils import char_edit_ratio
from pref_static_utils import load_jsonl

QWEN_KINDS = frozenset({"qwen_base", "qwen_base_norms", "qwen_adapter"})


@dataclass
class IntervalLossConfig:
  margin_a: float = 0.05
  margin_c_low: float = 0.05
  margin_c_high: float = 0.05
  zone_weight: float = 1.0
  weight_a: float = 3.0
  weight_b: float = 1.0
  weight_c: float = 1.0
  weight_d: float = 1.0


def interval_zone_loss(
  f_values: torch.Tensor,
  positions: list[str],
  *,
  cfg: IntervalLossConfig,
) -> torch.Tensor:
  losses: list[torch.Tensor] = []
  weights = {
    "a": cfg.weight_a,
    "b": cfg.weight_b,
    "c": cfg.weight_c,
    "d": cfg.weight_d,
  }
  for i, pos in enumerate(positions):
    f_i = f_values[i]
    w = weights.get(pos, 1.0)
    if pos == "a":
      losses.append(w * cfg.zone_weight * F.relu(f_i + cfg.margin_a) ** 2)
    elif pos == "b":
      losses.append(w * cfg.zone_weight * f_i ** 2)
    elif pos == "c":
      losses.append(w * cfg.zone_weight * F.relu(cfg.margin_c_low - f_i) ** 2)
      losses.append(w * cfg.zone_weight * F.relu(f_i - (1.0 - cfg.margin_c_high)) ** 2)
    elif pos == "d":
      losses.append(w * cfg.zone_weight * (f_i - 1.0) ** 2)
    else:
      raise ValueError(f"unknown position {pos!r}")
  if not losses:
    return torch.tensor(0.0, device=f_values.device)
  return torch.stack(losses).mean()


def pearson(xs: list[float], ys: list[float]) -> float | None:
  if len(xs) < 2:
    return None
  x = np.array(xs, dtype=np.float64)
  y = np.array(ys, dtype=np.float64)
  if float(x.std()) == 0.0 or float(y.std()) == 0.0:
    return None
  return float(np.corrcoef(x, y)[0, 1])


def zone_hit(row: dict, f: float, *, eps: float = 0.25) -> bool:
  pos = row["position"]
  if pos == "a":
    return f < 0
  if pos == "b":
    return abs(f) <= eps
  if pos == "c":
    return f > eps and f < 1.0 - eps
  if pos == "d":
    return abs(f - 1.0) <= eps
  return False


def primary_a_sign_hit(row: dict, f: float) -> bool | None:
  if row["y_kind"] not in QWEN_KINDS:
    return None
  if row["position"] == "a":
    return f < 0
  if row["position"] == "c":
    return f > 0
  return None


def primary_a_sign_accuracy(rows: list[dict], f_values: list[float]) -> dict[str, float]:
  hits = 0
  total = 0
  a_hits = a_total = c_hits = c_total = 0
  for row, f in zip(rows, f_values):
    hit = primary_a_sign_hit(row, f)
    if hit is None:
      continue
    total += 1
    hits += int(hit)
    if row["position"] == "a":
      a_total += 1
      a_hits += int(hit)
    elif row["position"] == "c":
      c_total += 1
      c_hits += int(hit)
  return {
    "primary_a_sign_acc": hits / total if total else 0.0,
    "primary_a_n": float(total),
    "primary_a_frac_lt_0": a_hits / a_total if a_total else 0.0,
    "primary_a_n_a": float(a_total),
    "primary_c_frac_gt_0": c_hits / c_total if c_total else 0.0,
    "primary_c_n_c": float(c_total),
  }


def subset_rows(rows: list[dict], *, predicate) -> list[dict]:
  return [r for r in rows if predicate(r)]


def h1_within_position(rows: list[dict], f_values: list[float]) -> dict[str, dict]:
  by_pos: dict[str, list[int]] = defaultdict(list)
  for i, row in enumerate(rows):
    if row["y_kind"] not in QWEN_KINDS:
      continue
    if row["position"] not in ("a", "c"):
      continue
    by_pos[row["position"]].append(i)

  out: dict[str, dict] = {}
  for pos, idxs in sorted(by_pos.items()):
    fs = [f_values[i] for i in idxs]
    len_diff = [float(len(rows[i]["y"]) - len(rows[i]["draft"])) for i in idxs]
    edit_ratio = [char_edit_ratio(rows[i]["draft"], rows[i]["y"]) for i in idxs]
    out[pos] = {
      "n": len(idxs),
      "pearson_f_len_diff": pearson(fs, len_diff),
      "pearson_f_edit_ratio": pearson(fs, edit_ratio),
    }
    if pos == "a":
      shorter = [i for i in idxs if len(rows[i]["y"]) < len(rows[i]["draft"])]
      longer = [i for i in idxs if len(rows[i]["y"]) >= len(rows[i]["draft"])]
      out[pos]["frac_f_lt_0_if_shorter"] = (
        sum(f_values[i] < 0 for i in shorter) / len(shorter) if shorter else None
      )
      out[pos]["frac_f_lt_0_if_longer"] = (
        sum(f_values[i] < 0 for i in longer) / len(longer) if longer else None
      )
  return out


def eval_report(rows: list[dict], f_values: list[float], *, eps: float = 0.25) -> dict:
  primary = primary_a_sign_accuracy(rows, f_values)
  h1 = h1_within_position(rows, f_values)

  def acc_for(predicate) -> dict[str, float]:
    sub = [(r, f) for r, f in zip(rows, f_values) if predicate(r)]
    if not sub:
      return {"n": 0.0, "acc": 0.0}
    hits = sum(int(zone_hit(r, f, eps=eps)) for r, f in sub)
    return {"n": float(len(sub)), "acc": hits / len(sub)}

  structural = acc_for(lambda r: r["position"] == "b")
  human_d = acc_for(lambda r: r["y_kind"] == "human")
  qwen_d_match = acc_for(
    lambda r: r["y_kind"] in QWEN_KINDS and r["position"] == "d" and r.get("y_eq_human_edit")
  )
  qwen_d_distinct = acc_for(
    lambda r: r["y_kind"] in QWEN_KINDS
    and r["position"] == "d"
    and not r.get("y_eq_human_edit")
  )

  return {
    **primary,
    "h1": h1,
    "structural_zero": structural,
    "human_d_ceiling": human_d,
    "qwen_d_string_match": qwen_d_match,
    "qwen_d_distinct_expression": qwen_d_distinct,
  }


def load_d_rows(path) -> list[dict]:
  return load_jsonl(str(path))
