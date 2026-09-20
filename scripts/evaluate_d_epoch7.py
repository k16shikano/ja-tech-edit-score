#!/usr/bin/env python3
"""D epoch7 判定器の直接評価（2.2節）。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import merge_rows
from b_generation_common import write_json
from pref_d_cross_encoder import load_cross_encoder, predict_f_delta
from pref_d_interval_utils import eval_report, load_d_rows
from pref_static_utils import load_jsonl


POSITION_TO_RANK = {"a": 0, "b": 1, "c": 2, "d": 3}
QWEN_KINDS = frozenset({"qwen_base", "qwen_base_norms", "qwen_adapter"})


def pair_token_length(tokenizer, draft: str, candidate: str, *, max_length: int) -> int:
  encoded = tokenizer(
    [draft],
    [candidate],
    padding=False,
    truncation=False,
    add_special_tokens=True,
  )
  return len(encoded["input_ids"][0])


def score_rows(model, tokenizer, rows: list[dict], *, max_length: int, device: torch.device) -> list[float | None]:
  scores: list[float | None] = []
  for start in range(0, len(rows), 16):
    batch = rows[start : start + 16]
    lengths = [
      pair_token_length(tokenizer, r["draft"], r["y"], max_length=max_length) for r in batch
    ]
    if any(length > max_length for length in lengths):
      scores.extend([None] * len(batch))
      continue
    vals = predict_f_delta(
      model,
      tokenizer,
      [r["draft"] for r in batch],
      [r["y"] for r in batch],
      max_length=max_length,
      device=device,
      batch_size=len(batch),
    )
    for v in vals:
      if not math.isfinite(v):
        scores.append(None)
      else:
        scores.append(float(v))
  return scores


def u_margin(m: float) -> float:
  if m > 0:
    return 1.0
  if m == 0:
    return 0.5
  return 0.0


def rank_pair_metrics(items: dict[str, list[dict]], scores_by_row_id: dict[str, float | None]) -> dict:
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
      if row["y_kind"] == "human":
        cand_kind = "human"
      elif row["y_kind"] in QWEN_KINDS:
        cand_kind = "qwen"
      else:
        cand_kind = row["y_kind"]
      indexed.append(
        {
          "row": row,
          "j": len(indexed),
          "label_rank": POSITION_TO_RANK[pos],
          "score": scores_by_row_id.get(row["row_id"]),
          "y_kind": cand_kind,
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
      if a["score"] is None or b["score"] is None:
        if a["score"] is None:
          stats["length_or_error"] += 1
        if b["score"] is None:
          stats["length_or_error"] += 1
        continue
      m = (a["label_rank"] - b["label_rank"]) * (a["score"] - b["score"])
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


def group_by_item(rows: list[dict]) -> dict[str, list[dict]]:
  out: dict[str, list[dict]] = defaultdict(list)
  for row in rows:
    out[str(row["item_id"])].append(row)
  return out


def qwen_only_metrics(items: dict[str, list[dict]], scores_by_row_id: dict[str, float | None]) -> dict:
  filtered: dict[str, list[dict]] = defaultdict(list)
  for item_id, rows in items.items():
    for row in rows:
      if row["y_kind"] in QWEN_KINDS:
        filtered[item_id].append(row)
  return rank_pair_metrics(filtered, scores_by_row_id)


def c_length_and_agreement(
  model,
  tokenizer,
  *,
  pairs_path: Path,
  judgments_path: Path,
  max_length: int,
  device: torch.device,
) -> dict:
  pairs = load_jsonl(str(pairs_path))
  judgments = load_jsonl(str(judgments_path))
  merged = merge_rows(pairs, judgments)

  left_over = right_over = either_over = both_ok = 0
  length_rows = []
  for row in merged:
    draft = row["context_draft"]
    left = row["a_text"]
    right = row["b_text"]
    left_len = pair_token_length(tokenizer, draft, left, max_length=max_length)
    right_len = pair_token_length(tokenizer, draft, right, max_length=max_length)
    left_is_over = left_len > max_length
    right_is_over = right_len > max_length
    if left_is_over:
      left_over += 1
    if right_is_over:
      right_over += 1
    if left_is_over or right_is_over:
      either_over += 1
    else:
      both_ok += 1
    length_rows.append(
      {
        "pair_id": row["pair_id"],
        "left_tokens": left_len,
        "right_tokens": right_len,
        "left_over": left_is_over,
        "right_over": right_is_over,
      }
    )

  evaluable = []
  for row in merged:
    draft = row["context_draft"]
    left = row["a_text"]
    right = row["b_text"]
    if pair_token_length(tokenizer, draft, left, max_length=max_length) > max_length:
      continue
    if pair_token_length(tokenizer, draft, right, max_length=max_length) > max_length:
      continue
    left_score = predict_f_delta(model, tokenizer, [draft], [left], max_length=max_length, device=device)[0]
    right_score = predict_f_delta(model, tokenizer, [draft], [right], max_length=max_length, device=device)[0]
    if not math.isfinite(left_score) or not math.isfinite(right_score):
      continue
    human_choice = row.get("choice")
    if human_choice == "tie":
      continue
    model_pref = "a" if left_score > right_score else ("b" if right_score > left_score else "tie")
    if left_score == right_score:
      model_pref = "tie"
    evaluable.append(
      {
        "pair_id": row["pair_id"],
        "human_choice": human_choice,
        "model_pref": model_pref,
        "left_score": left_score,
        "right_score": right_score,
      }
    )

  agreed = sum(
    1
    for r in evaluable
    if r["model_pref"] == r["human_choice"]
    or (r["model_pref"] == "tie" and r["human_choice"] == "tie")
  )
  ties = sum(1 for r in evaluable if r["model_pref"] == r["human_choice"] == "tie")
  decisive = [r for r in evaluable if r["human_choice"] != "tie" and r["model_pref"] != "tie"]
  decisive_agreed = sum(1 for r in decisive if r["model_pref"] == r["human_choice"])

  n_human_non_tie = sum(1 for row in merged if row.get("choice") != "tie")
  return {
    "length_report": {
      "total_pairs": len(merged),
      "left_over": left_over,
      "right_over": right_over,
      "either_over": either_over,
      "both_within_limit": both_ok,
      "rows": length_rows,
    },
    "rank_agreement": {
      "evaluable_n": len(evaluable),
      "coverage_vs_56": len(evaluable) / 56 if n_human_non_tie else None,
      "coverage_vs_60": len(evaluable) / 60,
      "exact_agreement": agreed / len(evaluable) if evaluable else None,
      "decisive_agreement": decisive_agreed / len(decisive) if decisive else None,
      "tie_matches": ties,
      "items": evaluable,
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lock", type=Path, default=Path("outputs/d_epoch7/d_epoch7.lock.json"))
  parser.add_argument("--d-train", type=Path, default=Path("data/d/train.jsonl"))
  parser.add_argument("--d-valid", type=Path, default=Path("data/d/valid.jsonl"))
  parser.add_argument("--c-pairs", type=Path, default=Path("data/blind_eval/pairs_gold_vs_adapter_selected.jsonl"))
  parser.add_argument("--c-judgments", type=Path, default=Path("data/blind_eval/judgments_gold_vs_adapter_selected.jsonl"))
  parser.add_argument("--out", type=Path, default=Path("outputs/d_epoch7/evaluation"))
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()

  code_root = Path(__file__).resolve().parents[1]
  lock = json.loads((code_root / args.lock).read_text(encoding="utf-8"))
  ckpt = Path(lock["checkpoint"]["path"])
  device = torch.device(args.device)

  model, tokenizer, cfg, meta = load_cross_encoder(ckpt, device=device)
  max_length = int(lock["input"]["max_length"])

  train_rows = load_d_rows(code_root / args.d_train)
  valid_rows = load_d_rows(code_root / args.d_valid)
  train_items = {str(r["item_id"]) for r in train_rows}
  valid_items = {str(r["item_id"]) for r in valid_rows}
  overlap = train_items & valid_items

  all_rows = train_rows + valid_rows
  scores = score_rows(model, tokenizer, all_rows, max_length=max_length, device=device)
  scores_by_row_id = {r["row_id"]: s for r, s in zip(all_rows, scores)}

  valid_f = [s for s in scores[len(train_rows) :] if s is not None]
  valid_report = eval_report(valid_rows, valid_f)

  items_all = group_by_item(all_rows)
  items_valid = group_by_item(valid_rows)
  pair_all = rank_pair_metrics(items_all, scores_by_row_id)
  pair_valid = rank_pair_metrics(items_valid, scores_by_row_id)
  pair_valid_qwen = qwen_only_metrics(items_valid, scores_by_row_id)

  c_report = c_length_and_agreement(
    model,
    tokenizer,
    pairs_path=code_root / args.c_pairs,
    judgments_path=code_root / args.c_judgments,
    max_length=max_length,
    device=device,
  )

  out_dir = code_root / args.out
  out_dir.mkdir(parents=True, exist_ok=True)
  summary = {
    "lock": str(code_root / args.lock),
    "checkpoint": str(ckpt),
    "best_epoch": meta.get("best_epoch"),
    "split_check": {
      "train_items": len(train_items),
      "valid_items": len(valid_items),
      "overlap_items": len(overlap),
      "overlap_ids": sorted(overlap) if overlap else [],
    },
    "valid_interval_report": valid_report,
    "pair_rank_valid": pair_valid,
    "pair_rank_valid_qwen_only": pair_valid_qwen,
    "pair_rank_all": pair_all,
    "c_eval": c_report,
  }
  write_json(out_dir / "summary.json", summary)
  write_json(out_dir / "c_length_report.json", c_report["length_report"])
  write_json(out_dir / "d_valid_pair_rank.json", pair_valid)
  print(json.dumps({"wrote": str(out_dir), "A_macro_valid": pair_valid["A_macro"]}, ensure_ascii=False))


if __name__ == "__main__":
  main()
