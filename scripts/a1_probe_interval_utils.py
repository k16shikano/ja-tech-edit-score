#!/usr/bin/env python3
"""A1 probe 600 対を区間教師 f(x,y) 用の行に載せる。"""
from __future__ import annotations

import json
import random
from pathlib import Path

from a1_probe_position_server import load_samples, pair_id_of
from pref_static_utils import load_jsonl

ZONE_POSITIONS = frozenset({"a", "eq", "b", "c", "d"})


def load_interval_rows(
  samples_dir: Path,
  judgments_path: Path,
) -> list[dict]:
  samples = load_samples(samples_dir)
  rows: list[dict] = []
  for judgment in load_jsonl(judgments_path):
    pair_id = str(judgment.get("pair_id") or "").strip()
    item_id = str(judgment.get("item_id") or "").strip()
    mode = str(judgment.get("mode") or "").strip()
    position = str(judgment.get("position") or "").strip()
    if not pair_id or position not in ZONE_POSITIONS:
      continue
    key = (item_id, mode)
    if key not in samples:
      raise KeyError(f"missing sample for {pair_id}")
    sample = samples[key]
    draft = str(sample.get("draft") or "")
    gold = str(sample.get("gold") or "")
    y = str(sample.get("generated") or "")
    rows.append(
      {
        "pair_id": pair_id,
        "item_id": item_id,
        "mode": mode,
        "position": position,
        "draft": draft,
        "y": y,
        "gold": gold,
        "judgment_source": str(judgment.get("source") or ""),
      }
    )
  return rows


def split_interval_rows(
  rows: list[dict],
  *,
  eval_pair_ids: set[str] | None = None,
  eval_fraction: float = 0.2,
  seed: int = 0,
) -> tuple[list[dict], list[dict], dict]:
  if eval_pair_ids is not None:
    train = [r for r in rows if r["pair_id"] not in eval_pair_ids]
    eval_rows = [r for r in rows if r["pair_id"] in eval_pair_ids]
    stats = {
      "n_total": len(rows),
      "n_train": len(train),
      "n_eval": len(eval_rows),
      "split": "eval_pair_ids",
    }
    return train, eval_rows, stats

  pair_ids = sorted({r["pair_id"] for r in rows})
  rng = random.Random(seed)
  shuffled = list(pair_ids)
  rng.shuffle(shuffled)
  n_eval = max(1, int(round(len(shuffled) * eval_fraction)))
  eval_ids = set(shuffled[:n_eval])
  train = [r for r in rows if r["pair_id"] not in eval_ids]
  eval_rows = [r for r in rows if r["pair_id"] in eval_ids]
  stats = {
    "n_total": len(rows),
    "n_train": len(train),
    "n_eval": len(eval_rows),
    "split": f"random_fraction={eval_fraction}",
    "seed": seed,
  }
  return train, eval_rows, stats


def collect_interval_texts(rows: list[dict]) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for row in rows:
    for key in ("draft", "y", "gold"):
      value = str(row[key])
      if value not in seen:
        seen.add(value)
        texts.append(value)
  return texts


def write_jsonl(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_eval_pair_ids(path: Path, pair_ids: set[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for pid in sorted(pair_ids):
      f.write(json.dumps({"pair_id": pid}, ensure_ascii=False) + "\n")


def load_eval_pair_ids(path: Path) -> set[str]:
  return {str(r["pair_id"]) for r in load_jsonl(path) if r.get("pair_id")}


def char_edit_ratio(a: str, b: str) -> float:
  if a == b:
    return 0.0
  la, lb = len(a), len(b)
  if la == 0 and lb == 0:
    return 0.0
  # 簡易: 1 - 2*common/ (la+lb) ではなく正規化 Levenshtein 近似として長さ差比率
  return abs(la - lb) / max(la, lb, 1)
