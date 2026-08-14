#!/usr/bin/env python3
"""難試験 v2b / v2c を setwise runtime で採点する。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from setwise_model import (
  STRICT_COMPARE_EPS,
  format_best_model,
  strict_pair_outcome,
  strict_top1_hit,
)


def load_items(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      obj["_line"] = line_no
      rows.append(obj)
  return rows


def validate_item(item: dict) -> str | None:
  if item.get("status") != "labeled":
    return None
  cands = item.get("candidates") or []
  ids = [c.get("id") for c in cands]
  if len(ids) < 2:
    return f"{item.get('id')}: need >=2 candidates"
  if len(ids) != len(set(ids)):
    return f"{item.get('id')}: duplicate candidate id"
  human = item.get("human") or {}
  best = human.get("best_id")
  if not best:
    return f"{item.get('id')}: human.best_id required when labeled"
  if best not in ids:
    return f"{item.get('id')}: best_id {best!r} not in candidates"
  rank = human.get("rank")
  if rank is not None and set(rank) != set(ids):
    return f"{item.get('id')}: rank must be a permutation of candidate ids"
  if not str(item.get("base_text") or "").strip():
    return f"{item.get('id')}: empty base_text"
  return None


def spearman(xs: list[float], ys: list[float]) -> float | None:
  n = len(xs)
  if n < 2:
    return None

  def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    i = 0
    while i < n:
      j = i
      while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
        j += 1
      avg = (i + j) / 2.0 + 1.0
      for k in range(i, j + 1):
        out[order[k]] = avg
      i = j + 1
    return out

  rx, ry = ranks(xs), ranks(ys)
  mx = sum(rx) / n
  my = sum(ry) / n
  num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
  denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
  deny = math.sqrt(sum((b - my) ** 2 for b in ry))
  if denx == 0 or deny == 0:
    return None
  return num / (denx * deny)


def pairwise_agreement(human_rank: list[str], scores: dict[str, float]) -> tuple[int, int]:
  agree = 0
  total = 0
  for i, better in enumerate(human_rank):
    for worse in human_rank[i + 1 :]:
      total += 1
      if scores[better] > scores[worse] + STRICT_COMPARE_EPS:
        agree += 1
  return agree, total


def pair_direction(scores: dict[str, float], better: str, worse: str) -> bool:
  return scores[better] > scores[worse] + STRICT_COMPARE_EPS


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="labeled hard-eval jsonl")
  parser.add_argument("--model", required=True, help="pref-setwise-section-triples directory")
  parser.add_argument("--report", default="outputs/hard_eval_setwise_report.json")
  args = parser.parse_args()

  from pref_setwise_runtime import load_setwise_model, rank_candidates_setwise

  items = load_items(Path(args.input))
  errors: list[str] = []
  labeled: list[dict] = []
  for item in items:
    err = validate_item(item)
    if err:
      errors.append(err)
    elif item.get("status") == "labeled":
      labeled.append(item)
  if errors:
    raise SystemExit("validation failed:\n  " + "\n  ".join(errors))
  if not labeled:
    raise SystemExit("no labeled items")

  loaded = load_setwise_model(Path(args.model))
  per_item: list[dict] = []
  top1_hits = 0
  top1_ties = 0
  pair_agree = 0
  pair_total = 0
  direction_counts = {
    "human_over_draft": [0, 0],
    "human_over_mid": [0, 0],
    "mid_over_draft": [0, 0],
  }
  direction_detail = {
    "human_over_draft": {"win": 0, "tie": 0, "loss": 0},
    "human_over_mid": {"win": 0, "tie": 0, "loss": 0},
    "mid_over_draft": {"win": 0, "tie": 0, "loss": 0},
  }
  length_corrs: list[float] = []

  for item in labeled:
    base = item["base_text"]
    cands = item["candidates"]
    ids = [c["id"] for c in cands]
    texts = [c["text"] for c in cands]
    result = rank_candidates_setwise(loaded, base, texts)
    score_map = {cid: float(result["logits"][idx]) for idx, cid in enumerate(ids)}
    ranked_ids = [ids[i] for i in result["ranked_indices"]]
    best_human = item["human"]["best_id"]
    hit = strict_top1_hit(score_map, best_human, ids)
    top1_tie = bool(result.get("top1_tie"))

    if hit:
      top1_hits += 1
    if top1_tie:
      top1_ties += 1

    row: dict = {
      "id": item["id"],
      "best_human": best_human,
      "best_model": format_best_model(score_map, ids),
      "top1_hit": hit,
      "top1_tie": top1_tie,
      "scores": score_map,
      "model_rank": ranked_ids,
      "logits": result["logits"],
    }
    human_rank = item["human"].get("rank") or []
    if human_rank:
      a, t = pairwise_agreement(human_rank, score_map)
      pair_agree += a
      pair_total += t
      row["pairwise_agree"] = a
      row["pairwise_total"] = t
      row["human_rank"] = human_rank
      if len(human_rank) >= 3:
        human_id, mid_id, draft_id = human_rank[0], human_rank[1], human_rank[2]
        for key, better, worse in (
          ("human_over_draft", human_id, draft_id),
          ("human_over_mid", human_id, mid_id),
          ("mid_over_draft", mid_id, draft_id),
        ):
          outcome = strict_pair_outcome(score_map[better], score_map[worse])
          direction_detail[key][outcome] += 1
          direction_counts[key][1] += 1
          if outcome == "win":
            direction_counts[key][0] += 1
          row[key] = outcome == "win"
          row[f"{key}_outcome"] = outcome

    lengths = [float(len(t)) for t in texts]
    corr = spearman(result["logits"], lengths)
    if corr is not None:
      length_corrs.append(corr)
      row["score_length_spearman"] = corr
    per_item.append(row)

  n = len(labeled)
  summary = {
    "scorer": "setwise",
    "model": str(args.model),
    "input": args.input,
    "strict_compare_eps": STRICT_COMPARE_EPS,
    "n_labeled": n,
    "top1_accuracy": top1_hits / n,
    "top1_hits": top1_hits,
    "top1_ties": top1_ties,
    "pairwise_accuracy": (pair_agree / pair_total) if pair_total else None,
    "pairwise_agree": pair_agree,
    "pairwise_total": pair_total,
    "mean_score_length_spearman": (
      sum(length_corrs) / len(length_corrs) if length_corrs else None
    ),
  }
  for key, (hits, total) in direction_counts.items():
    summary[f"{key}_rate"] = (hits / total) if total else None
    summary[f"{key}_hits"] = hits
    summary[f"{key}_total"] = total
    detail = direction_detail[key]
    summary[f"{key}_wins"] = detail["win"]
    summary[f"{key}_ties"] = detail["tie"]
    summary[f"{key}_losses"] = detail["loss"]

  report = {"summary": summary, "items": per_item}
  report_path = Path(args.report)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )

  md_path = report_path.with_suffix(".md")
  lines = [
    "# Hard Eval setwise report",
    "",
    f"- scorer: `{summary['scorer']}`",
    f"- model: `{summary['model']}`",
    f"- n: {summary['n_labeled']}",
    f"- top1_accuracy: {summary['top1_accuracy']:.3f} ({summary['top1_hits']}/{summary['n_labeled']})",
    f"- top1_ties: {summary['top1_ties']}",
  ]
  if summary["pairwise_accuracy"] is not None:
    lines.append(
      f"- pairwise_accuracy: {summary['pairwise_accuracy']:.3f} "
      f"({summary['pairwise_agree']}/{summary['pairwise_total']})"
    )
  for key in ("human_over_draft", "human_over_mid", "mid_over_draft"):
    rate = summary.get(f"{key}_rate")
    hits = summary.get(f"{key}_hits")
    total = summary.get(f"{key}_total")
    wins = summary.get(f"{key}_wins")
    ties = summary.get(f"{key}_ties")
    losses = summary.get(f"{key}_losses")
    if rate is not None:
      lines.append(
        f"- {key}: win/tie/loss {wins}/{ties}/{losses} "
        f"(rate {rate:.3f}, hits {hits}/{total})"
      )
  if summary["mean_score_length_spearman"] is not None:
    lines.append(
      f"- mean score↔length Spearman: {summary['mean_score_length_spearman']:.3f}"
    )
  lines.extend(["", "| id | human best | model best | hit |", "|---|---|---|:---:|"])
  for row in per_item:
    lines.append(
      f"| {row['id']} | {row['best_human']} | {row['best_model']} | "
      f"{'yes' if row['top1_hit'] else 'no'} |"
    )
  lines.append("")
  md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  print(json.dumps(summary, ensure_ascii=False, indent=2))
  print(f"wrote {report_path}")
  print(f"wrote {md_path}")


if __name__ == "__main__":
  main()
