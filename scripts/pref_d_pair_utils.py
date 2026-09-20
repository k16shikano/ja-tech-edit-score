#!/usr/bin/env python3
"""D ペア台帳と GPM/BT ペア評価の共通処理。"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F

from pref_static_utils import load_jsonl

POSITION_TO_RANK = {"a": 0, "b": 1, "c": 2, "d": 3}
QWEN_KINDS = frozenset({"qwen_base", "qwen_base_norms", "qwen_adapter"})


def u_margin(m: float) -> float:
  if m > 0:
    return 1.0
  if m == 0:
    return 0.5
  return 0.0


def gpm_score(vectors_a: torch.Tensor, vectors_b: torch.Tensor) -> torch.Tensor:
  if vectors_a.shape != vectors_b.shape:
    raise ValueError(f"shape mismatch: {vectors_a.shape} vs {vectors_b.shape}")
  if vectors_a.shape[-1] % 2 != 0:
    raise ValueError(f"head_dim must be even, got {vectors_a.shape[-1]}")
  k = vectors_a.shape[-1] // 2
  va = vectors_a.view(*vectors_a.shape[:-1], k, 2)
  vb = vectors_b.view(*vectors_b.shape[:-1], k, 2)
  return (va[..., 0] * vb[..., 1] - va[..., 1] * vb[..., 0]).sum(dim=-1)


def build_pair_ledger(rows: list[dict]) -> tuple[list[dict], dict]:
  by_item: dict[str, list[dict]] = defaultdict(list)
  for row in rows:
    by_item[str(row["item_id"])].append(row)

  ledger: list[dict] = []
  stats = defaultdict(int)
  for item_id, item_rows in sorted(by_item.items()):
    indexed = []
    for row in item_rows:
      draft = str(row["draft"])
      y = str(row["y"])
      if y == draft:
        stats["unchanged_candidate"] += 1
        continue
      pos = str(row["position"])
      if pos not in POSITION_TO_RANK:
        raise ValueError(f"unknown position {pos!r} for item {item_id}")
      indexed.append(
        {
          "row_id": str(row["row_id"]),
          "item_id": item_id,
          "draft": draft,
          "y": y,
          "position": pos,
          "label_rank": POSITION_TO_RANK[pos],
          "y_kind": str(row["y_kind"]),
        }
      )

    for left, right in combinations(indexed, 2):
      if left["label_rank"] == right["label_rank"]:
        stats["same_label_excluded"] += 1
        continue
      if left["label_rank"] < right["label_rank"]:
        loser, winner = left, right
      else:
        loser, winner = right, left
      ledger.append(
        {
          "item_id": item_id,
          "draft": left["draft"],
          "winner_row_id": winner["row_id"],
          "loser_row_id": loser["row_id"],
          "winner_y": winner["y"],
          "loser_y": loser["y"],
          "winner_position": winner["position"],
          "loser_position": loser["position"],
          "winner_rank": winner["label_rank"],
          "loser_rank": loser["label_rank"],
          "winner_y_kind": winner["y_kind"],
          "loser_y_kind": loser["y_kind"],
        }
      )
      stats["directed_pairs"] += 1

  summary = {
    "n_items": len(by_item),
    "n_source_rows": len(rows),
    "n_directed_pairs": len(ledger),
    "counts": dict(stats),
  }
  return ledger, summary


def load_pair_ledger(path: Path) -> list[dict]:
  return load_jsonl(str(path))


def filter_ledger_by_items(ledger: list[dict], item_ids: set[str]) -> list[dict]:
  return [row for row in ledger if str(row["item_id"]) in item_ids]


def group_ledger_by_item(ledger: list[dict]) -> dict[str, list[dict]]:
  out: dict[str, list[dict]] = defaultdict(list)
  for row in ledger:
    out[str(row["item_id"])].append(row)
  return out


def pairwise_bt_loss(
  delta_w: torch.Tensor,
  delta_l: torch.Tensor,
) -> torch.Tensor:
  return F.softplus(-(delta_w - delta_l))


def pairwise_gpm_loss(
  vec_w: torch.Tensor,
  vec_l: torch.Tensor,
) -> torch.Tensor:
  return F.softplus(-gpm_score(vec_w, vec_l))


def item_mean_pair_loss(
  item_ids: list[str],
  pairs_by_item: dict[str, list[dict]],
  pair_losses: dict[tuple[str, str, str], torch.Tensor],
) -> torch.Tensor:
  item_losses: list[torch.Tensor] = []
  for item_id in item_ids:
    pairs = pairs_by_item.get(item_id, [])
    if not pairs:
      continue
    losses = []
    for pair in pairs:
      key = (item_id, pair["winner_row_id"], pair["loser_row_id"])
      losses.append(pair_losses[key])
    item_losses.append(torch.stack(losses).mean())
  if not item_losses:
    return torch.tensor(0.0, device=next(iter(pair_losses.values())).device)
  return torch.stack(item_losses).mean()


def rank_pair_metrics_scalar(
  items: dict[str, list[dict]],
  scores_by_row_id: dict[str, float | None],
) -> dict:
  return _rank_pair_metrics(items, scores_by_row_id, mode="scalar")


def rank_pair_metrics_gpm(
  items: dict[str, list[dict]],
  vectors_by_row_id: dict[str, list[float] | None],
) -> dict:
  return _rank_pair_metrics(items, vectors_by_row_id, mode="gpm")


def _rank_pair_metrics(
  items: dict[str, list[dict]],
  values_by_row_id: dict[str, object | None],
  *,
  mode: str,
) -> dict:
  per_item: list[dict] = []
  macro_sum = 0.0
  macro_n = 0
  micro_num = 0.0
  micro_den = 0
  counts = {"correct": 0, "reversed": 0, "tie": 0}
  stats = defaultdict(int)

  for item_id, rows in sorted(items.items()):
    indexed = []
    for row in rows:
      pos = row["position"]
      y = row["y"]
      draft = row["draft"]
      if y == draft:
        stats["unchanged_candidate"] += 1
        continue
      indexed.append(
        {
          "row": row,
          "label_rank": POSITION_TO_RANK[pos],
          "value": values_by_row_id.get(row["row_id"]),
        }
      )

    planned_pairs = 0
    evaluated_pairs = 0
    item_us = []
    for a, b in combinations(indexed, 2):
      if a["label_rank"] == b["label_rank"]:
        stats["same_label_excluded"] += 1
        continue
      planned_pairs += 1
      if a["value"] is None or b["value"] is None:
        if a["value"] is None:
          stats["length_or_error"] += 1
        if b["value"] is None:
          stats["length_or_error"] += 1
        continue
      if mode == "scalar":
        m = (a["label_rank"] - b["label_rank"]) * (float(a["value"]) - float(b["value"]))
      elif mode == "gpm":
        va = torch.tensor(a["value"], dtype=torch.float64)
        vb = torch.tensor(b["value"], dtype=torch.float64)
        m = float((a["label_rank"] - b["label_rank"]) * gpm_score(va, vb).item())
      else:
        raise ValueError(f"unknown mode {mode!r}")
      u = u_margin(m)
      item_us.append(u)
      evaluated_pairs += 1
      micro_num += u
      micro_den += 1
      if u == 1.0:
        counts["correct"] += 1
      elif u == 0.0:
        counts["reversed"] += 1
      else:
        counts["tie"] += 1

    item_macro = sum(item_us) / len(item_us) if item_us else None
    if item_us:
      macro_sum += item_macro
      macro_n += 1
    per_item.append(
      {
        "item_id": item_id,
        "planned_pairs": planned_pairs,
        "evaluated_pairs": evaluated_pairs,
        "A_item": item_macro,
      }
    )

  return {
    "A_macro": (macro_sum / macro_n) if macro_n else None,
    "A_micro": (micro_num / micro_den) if micro_den else None,
    "items_with_pairs": macro_n,
    "planned_pairs_total": sum(x["planned_pairs"] for x in per_item),
    "evaluated_pairs_total": micro_den,
    "pair_counts": counts,
    "exclusion_stats": dict(stats),
    "per_item": per_item,
  }


def group_rows_by_item(rows: list[dict]) -> dict[str, list[dict]]:
  out: dict[str, list[dict]] = defaultdict(list)
  for row in rows:
    out[str(row["item_id"])].append(row)
  return out


def qwen_only_items(items: dict[str, list[dict]]) -> dict[str, list[dict]]:
  filtered: dict[str, list[dict]] = defaultdict(list)
  for item_id, rows in items.items():
    for row in rows:
      if row["y_kind"] in QWEN_KINDS:
        filtered[item_id].append(row)
  return filtered


def finite_scalar(value: float | None) -> bool:
  return value is not None and math.isfinite(value)


def write_json(path: Path, obj: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
