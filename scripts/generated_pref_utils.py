#!/usr/bin/env python3
"""生成文の人手選好を教師にした評価器実験の共通処理。"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from pref_static_utils import load_jsonl
from train_pref_sentseq import (
  PreparedSentSeqData,
  SentSeqEncoder,
  SentSeqRewardModel,
  encode_texts_to_doc_vectors,
)

SCHEMA_GENERATED_PREF = "generated_pref_v1"
SELECTION_BIASED_COMPARE_TYPES = frozenset(
  {
    "4_draft_vs_adapter_selected",
    "5_gold_vs_adapter_selected",
  }
)


def is_generated_pref_row(row: dict) -> bool:
  return row.get("schema") == SCHEMA_GENERATED_PREF or "preference" in row


def is_legacy_pref_row(row: dict) -> bool:
  return "label" in row and "preference" not in row


def preference_to_target(preference: str) -> float:
  pref = str(preference).strip().lower()
  if pref == "a":
    return 1.0
  if pref == "b":
    return 0.0
  if pref == "tie":
    return 0.5
  raise ValueError(f"unknown preference: {preference!r}")


def preference_loss(delta: torch.Tensor, preference: str) -> torch.Tensor:
  """delta = r(A) - r(B)。tie は sigmoid(delta)=0.5 を目標に BCE。"""
  target = torch.full_like(delta, preference_to_target(preference))
  return nn.functional.binary_cross_entropy_with_logits(delta, target)


def feature_dim_for(d_model: int, *, use_length_features: bool) -> int:
  numeric = 5 if use_length_features else 1
  return d_model * 4 + numeric


def score_from_doc_vectors_with_length_toggle(
  model: SentSeqRewardModel,
  source_vec: torch.Tensor,
  candidate_vec: torch.Tensor,
  *,
  len_source: torch.Tensor,
  len_candidate: torch.Tensor,
  use_length_features: bool,
) -> torch.Tensor:
  diff = source_vec - candidate_vec
  abs_diff = diff.abs()
  cos = (source_vec * candidate_vec).sum(dim=-1, keepdim=True)
  if use_length_features:
    len_s = len_source.unsqueeze(-1)
    len_c = len_candidate.unsqueeze(-1)
    numeric = torch.cat(
      [
        cos,
        len_s,
        len_c,
        len_c - len_s,
        (len_c - len_s).abs(),
      ],
      dim=-1,
    )
  else:
    numeric = cos
  x = torch.cat([source_vec, candidate_vec, diff, abs_diff, numeric], dim=-1)
  return model.head(x)


def build_sentseq_model(
  *,
  embed_dim: int,
  d_model: int,
  use_length_features: bool,
  **encoder_kwargs,
) -> SentSeqRewardModel:
  feature_dim = feature_dim_for(d_model, use_length_features=use_length_features)
  encoder = SentSeqEncoder(embed_dim=embed_dim, d_model=d_model, **encoder_kwargs)
  return SentSeqRewardModel(encoder, feature_dim=feature_dim)


def index_by_key(rows: Iterable[dict], key: str) -> dict[str, dict]:
  out: dict[str, dict] = {}
  for row in rows:
    value = str(row.get(key) or "")
    if value:
      out[value] = row
  return out


def merge_blind_to_teacher_rows(
  pairs: list[dict],
  judgments: list[dict],
  items: list[dict],
) -> list[dict]:
  """pairs + judgments + items を結合し、360 件の選好教師行を返す。"""
  item_index = index_by_key(items, "id")
  judgment_index = index_by_key(judgments, "pair_id")
  rows: list[dict] = []
  missing_judgment = 0
  missing_item = 0

  for pair in pairs:
    pair_id = str(pair.get("pair_id") or "")
    judgment = judgment_index.get(pair_id)
    if judgment is None:
      missing_judgment += 1
      continue
    item_id = str(pair.get("item_id") or "")
    item = item_index.get(item_id)
    if item is None:
      missing_item += 1
      continue
    compare_type = str(pair.get("compare_type") or "")
    rows.append(
      {
        "schema": SCHEMA_GENERATED_PREF,
        "pair_id": pair_id,
        "item_id": item_id,
        "project_id": str(item.get("project_id") or ""),
        "unit": str(item.get("unit") or ""),
        "compare_type": compare_type,
        "source_text": str(pair.get("context_draft") or ""),
        "candidate_a": str(pair.get("a_text") or ""),
        "candidate_b": str(pair.get("b_text") or ""),
        "a_source": str(pair.get("a_source") or judgment.get("a_source") or ""),
        "b_source": str(pair.get("b_source") or judgment.get("b_source") or ""),
        "preference": str(judgment.get("choice") or "").lower(),
        "a_broken": bool(judgment.get("a_broken")),
        "b_broken": bool(judgment.get("b_broken")),
        "a_noedit": bool(judgment.get("a_noedit")),
        "b_noedit": bool(judgment.get("b_noedit")),
        "selection_biased": compare_type in SELECTION_BIASED_COMPARE_TYPES,
        "swapped": bool(pair.get("swapped")),
      }
    )

  if missing_judgment or missing_item:
    raise SystemExit(
      f"missing joins: judgments={missing_judgment} items={missing_item} "
      f"pairs={len(pairs)}"
    )
  return rows


def summarize_teacher_rows(rows: list[dict]) -> dict:
  return {
    "total": len(rows),
    "preference": dict(Counter(str(r.get("preference") or "") for r in rows)),
    "unit": dict(Counter(str(r.get("unit") or "") for r in rows)),
    "compare_type": dict(Counter(str(r.get("compare_type") or "") for r in rows)),
    "broken_any": sum(1 for r in rows if r.get("a_broken") or r.get("b_broken")),
    "selection_biased": sum(1 for r in rows if r.get("selection_biased")),
  }


def assign_item_folds(
  items: list[dict],
  *,
  n_folds: int = 5,
  seed: int = 42,
) -> dict[str, int]:
  """item_id 単位の決定的 stratified fold 割当（unit 比率を保つ）。"""
  rng = random.Random(seed)
  by_unit: dict[str, list[str]] = defaultdict(list)
  for item in items:
    item_id = str(item.get("id") or "")
    if not item_id:
      continue
    unit = str(item.get("unit") or "unknown")
    by_unit[unit].append(item_id)

  item_to_fold: dict[str, int] = {}
  for unit, ids in sorted(by_unit.items()):
    shuffled = list(ids)
    rng.shuffle(shuffled)
    for idx, item_id in enumerate(shuffled):
      item_to_fold[item_id] = idx % n_folds
  return item_to_fold


def build_fold_splits(
  rows: list[dict],
  item_to_fold: dict[str, int],
  *,
  n_folds: int = 5,
) -> list[tuple[list[dict], list[dict]]]:
  folds: list[tuple[list[dict], list[dict]]] = []
  for fold_idx in range(n_folds):
    valid_items = {item_id for item_id, f in item_to_fold.items() if f == fold_idx}
    train_rows = [r for r in rows if str(r.get("item_id") or "") not in valid_items]
    valid_rows = [r for r in rows if str(r.get("item_id") or "") in valid_items]
    folds.append((train_rows, valid_rows))
  return folds


def verify_fold_splits(
  folds: list[tuple[list[dict], list[dict]]],
  *,
  n_folds: int,
) -> None:
  seen_valid_items: set[str] = set()
  for fold_idx, (train_rows, valid_rows) in enumerate(folds):
    train_items = {str(r.get("item_id") or "") for r in train_rows}
    valid_items = {str(r.get("item_id") or "") for r in valid_rows}
    overlap = train_items & valid_items
    if overlap:
      raise SystemExit(
        f"fold {fold_idx}: train/valid item overlap={len(overlap)} "
        f"example={next(iter(overlap))}"
      )
    dup = seen_valid_items & valid_items
    if dup:
      raise SystemExit(
        f"fold {fold_idx}: valid items already used in another fold "
        f"example={next(iter(dup))}"
      )
    seen_valid_items |= valid_items


def filter_rows_by_unit(
  rows: list[dict],
  *,
  unit: str,
) -> tuple[list[dict], dict]:
  kept = [r for r in rows if str(r.get("unit") or "") == unit]
  excluded = len(rows) - len(kept)
  return kept, {"excluded": excluded, "kept": len(kept), "unit": unit}


def load_generated_pref_rows(
  path: str | Path,
  *,
  unit: str | None = "section",
) -> tuple[list[dict], dict]:
  rows = load_jsonl(str(path))
  generated = [r for r in rows if is_generated_pref_row(r)]
  if unit is None:
    return generated, {"excluded": 0, "kept": len(generated), "unit": None}
  return filter_rows_by_unit(generated, unit=unit)


def load_legacy_anchor_rows(
  path: str | Path,
  *,
  unit: str = "section",
) -> list[dict]:
  """keep section train など legacy 行（label/chosen_first）だけを返す。"""
  rows = load_jsonl(str(path))
  out: list[dict] = []
  for row in rows:
    if not is_legacy_pref_row(row):
      continue
    meta = row.get("meta") or {}
    row_unit = str(meta.get("unit") or "")
    if row_unit != unit:
      continue
    if meta.get("pair_order", "chosen_first") != "chosen_first":
      continue
    if int(row.get("label", 0)) != 1:
      continue
    out.append(row)
  return out


def augment_generated_side_swap(rows: list[dict]) -> list[dict]:
  """非 tie 行について左右入替の増強行を足す。preference も a↔b に反転。"""
  augmented = list(rows)
  for row in rows:
    pref = str(row.get("preference") or "").lower()
    if pref not in {"a", "b"}:
      continue
    swapped = dict(row)
    swapped["pair_id"] = str(row.get("pair_id") or "") + ":swap"
    swapped["candidate_a"] = row["candidate_b"]
    swapped["candidate_b"] = row["candidate_a"]
    swapped["a_source"] = row.get("b_source")
    swapped["b_source"] = row.get("a_source")
    swapped["a_broken"] = row.get("b_broken")
    swapped["b_broken"] = row.get("a_broken")
    swapped["a_noedit"] = row.get("b_noedit")
    swapped["b_noedit"] = row.get("a_noedit")
    swapped["preference"] = "b" if pref == "a" else "a"
    swapped["swapped"] = not bool(row.get("swapped"))
    augmented.append(swapped)
  return augmented


def legacy_row_to_training(row: dict) -> dict:
  """legacy 行を生成実験と同じ preference 表現へ。"""
  return {
    "schema": "legacy_anchor_v1",
    "pair_id": str(row.get("id") or ""),
    "source_text": row["source_text"],
    "candidate_a": row["candidate_a"],
    "candidate_b": row["candidate_b"],
    "preference": "a",
    "unit": str((row.get("meta") or {}).get("unit") or "section"),
  }


@torch.no_grad()
def score_pair_deltas(
  model: SentSeqRewardModel,
  rows: list[dict],
  prepared: PreparedSentSeqData,
  *,
  device: torch.device,
  batch_size: int,
  use_length_features: bool,
) -> np.ndarray:
  model.eval()
  deltas: list[float] = []
  for start in range(0, len(rows), batch_size):
    batch = rows[start : start + batch_size]
    sources = [r["source_text"] for r in batch]
    cand_a = [r["candidate_a"] for r in batch]
    cand_b = [r["candidate_b"] for r in batch]
    v_src = encode_texts_to_doc_vectors(model, sources, prepared, device=device)
    v_a = encode_texts_to_doc_vectors(model, cand_a, prepared, device=device)
    v_b = encode_texts_to_doc_vectors(model, cand_b, prepared, device=device)
    len_s = torch.tensor([float(len(t)) for t in sources], dtype=torch.float32, device=device)
    len_a = torch.tensor([float(len(t)) for t in cand_a], dtype=torch.float32, device=device)
    len_b = torch.tensor([float(len(t)) for t in cand_b], dtype=torch.float32, device=device)
    s_a = score_from_doc_vectors_with_length_toggle(
      model, v_src, v_a, len_source=len_s, len_candidate=len_a, use_length_features=use_length_features
    )
    s_b = score_from_doc_vectors_with_length_toggle(
      model, v_src, v_b, len_source=len_s, len_candidate=len_b, use_length_features=use_length_features
    )
    deltas.extend((s_a - s_b).float().cpu().tolist())
  return np.asarray(deltas, dtype=np.float64)


def fit_tie_threshold(train_deltas: np.ndarray, train_preferences: list[str]) -> float:
  """train の non-tie マージン分布から tie 判定用 |delta| 閾値を決める。"""
  abs_deltas = np.abs(train_deltas)
  non_tie = np.array([p in {"a", "b"} for p in train_preferences], dtype=bool)
  tie = np.array([p == "tie" for p in train_preferences], dtype=bool)
  if not np.any(non_tie):
    return 0.0
  # non-tie の下位 15 パーセンタイルを tie 域とする（train のみで決定）
  base = float(np.quantile(abs_deltas[non_tie], 0.15))
  if np.any(tie):
    tie_med = float(np.median(abs_deltas[tie]))
    return max(base, tie_med)
  return base


def predict_preferences(deltas: np.ndarray, tie_threshold: float) -> list[str]:
  out: list[str] = []
  for delta in deltas:
    if abs(delta) < tie_threshold:
      out.append("tie")
    elif delta > 0:
      out.append("a")
    else:
      out.append("b")
  return out


def evaluate_generated_preferences(
  deltas: np.ndarray,
  preferences: list[str],
  *,
  tie_threshold: float,
) -> dict[str, float | int | dict[str, int | float]]:
  prefs = [str(p).lower() for p in preferences]
  preds = predict_preferences(deltas, tie_threshold)
  non_tie_idx = [i for i, p in enumerate(prefs) if p in {"a", "b"}]
  non_tie_correct = sum(1 for i in non_tie_idx if preds[i] == prefs[i])
  tie_idx = [i for i, p in enumerate(prefs) if p == "tie"]
  tie_margins = [abs(float(deltas[i])) for i in tie_idx]
  confusion: dict[str, int] = Counter()
  for gold, pred in zip(prefs, preds, strict=True):
    confusion[f"{gold}->{pred}"] += 1

  three_way_correct = sum(1 for gold, pred in zip(prefs, preds, strict=True) if gold == pred)
  three_way_accuracy = three_way_correct / max(len(prefs), 1)

  recalls: dict[str, float] = {}
  for cls in ("a", "b", "tie"):
    gold_idx = [i for i, p in enumerate(prefs) if p == cls]
    if not gold_idx:
      continue
    hit = sum(1 for i in gold_idx if preds[i] == cls)
    recalls[cls] = hit / len(gold_idx)
  three_way_macro_recall = sum(recalls.values()) / max(len(recalls), 1)

  return {
    "n": len(prefs),
    "non_tie_n": len(non_tie_idx),
    "non_tie_accuracy": non_tie_correct / max(len(non_tie_idx), 1),
    "three_way_accuracy": three_way_accuracy,
    "three_way_macro_recall": three_way_macro_recall,
    "recall_a": recalls.get("a", 0.0),
    "recall_b": recalls.get("b", 0.0),
    "recall_tie": recalls.get("tie", 0.0),
    "tie_n": len(tie_idx),
    "tie_margin_mean": float(np.mean(tie_margins)) if tie_margins else 0.0,
    "tie_margin_median": float(np.median(tie_margins)) if tie_margins else 0.0,
    "tie_threshold": tie_threshold,
    "confusion": dict(confusion),
  }


def write_jsonl(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
