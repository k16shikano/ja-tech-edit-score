#!/usr/bin/env python3
"""難試験（Hard Eval）の採点。

ラベル付き JSONL の各項目について s(base_text, candidate) を出し、
人間の best_id / rank との一致を集計する。

scorer:
  bt … outputs/pref-bt 系（CPU）
  ce … outputs/pref-ce 系（GPU 可、CPU でも可）
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


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
  if rank is not None:
    if set(rank) != set(ids):
      return f"{item.get('id')}: rank must be a permutation of candidate ids"
  if not str(item.get("base_text") or "").strip():
    return f"{item.get('id')}: empty base_text"
  return None


def spearman(xs: list[float], ys: list[float]) -> float | None:
  n = len(xs)
  if n < 2:
    return None
  # rank with average ties
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


def score_fn_bt(model_dir: Path):
  from pref_bt_runtime import load_bt_model, score_candidates_bt

  loaded = load_bt_model(model_dir)

  def score(base: str, texts: list[str]) -> list[float]:
    return score_candidates_bt(loaded, base, texts)

  return score


def score_fn_ce(model_dir: Path):
  from pref_ce_runtime import load_ce_model, score_candidates_ce

  loaded = load_ce_model(model_dir)

  def score(base: str, texts: list[str]) -> list[float]:
    return score_candidates_ce(loaded, base, texts)

  return score


def pairwise_agreement(human_rank: list[str], scores: dict[str, float]) -> tuple[int, int]:
  """Returns (agree, total) over pairs implied by human_rank (best-first)."""
  agree = 0
  total = 0
  for i, better in enumerate(human_rank):
    for worse in human_rank[i + 1 :]:
      total += 1
      if scores[better] > scores[worse]:
        agree += 1
  return agree, total


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="labeled hard-eval jsonl")
  parser.add_argument("--scorer", choices=["bt", "ce"], required=True)
  parser.add_argument("--model", required=True, help="pref-bt or pref-ce directory")
  parser.add_argument("--report", default="outputs/hard_eval_report.json")
  parser.add_argument("--include-pending", action="store_true", help="score pending too (no metrics)")
  args = parser.parse_args()

  items = load_items(Path(args.input))
  errors = []
  labeled: list[dict] = []
  for item in items:
    status = item.get("status")
    if status == "labeled" or (args.include_pending and status == "pending"):
      err = validate_item(item) if status == "labeled" else None
      if err:
        errors.append(err)
      elif status == "labeled":
        labeled.append(item)
  if errors:
    raise SystemExit("validation failed:\n  " + "\n  ".join(errors))
  if not labeled:
    raise SystemExit("no labeled items")

  model_dir = Path(args.model)
  if args.scorer == "bt":
    score = score_fn_bt(model_dir)
  else:
    score = score_fn_ce(model_dir)

  per_item: list[dict] = []
  top1_hits = 0
  pair_agree = 0
  pair_total = 0
  length_corrs: list[float] = []

  for item in labeled:
    base = item["base_text"]
    cands = item["candidates"]
    ids = [c["id"] for c in cands]
    texts = [c["text"] for c in cands]
    scores_list = score(base, texts)
    score_map = {i: float(s) for i, s in zip(ids, scores_list, strict=True)}
    ranked = sorted(ids, key=lambda i: score_map[i], reverse=True)
    best_model = ranked[0]
    best_human = item["human"]["best_id"]
    hit = best_model == best_human
    if hit:
      top1_hits += 1

    row: dict = {
      "id": item["id"],
      "best_human": best_human,
      "best_model": best_model,
      "top1_hit": hit,
      "scores": score_map,
      "model_rank": ranked,
    }
    human_rank = item["human"].get("rank")
    if human_rank:
      a, t = pairwise_agreement(human_rank, score_map)
      pair_agree += a
      pair_total += t
      row["pairwise_agree"] = a
      row["pairwise_total"] = t
      row["human_rank"] = human_rank

    lengths = [float(len(t)) for t in texts]
    corr = spearman(scores_list, lengths)
    if corr is not None:
      length_corrs.append(corr)
      row["score_length_spearman"] = corr
    per_item.append(row)

  n = len(labeled)
  summary = {
    "scorer": args.scorer,
    "model": str(model_dir),
    "input": args.input,
    "n_labeled": n,
    "top1_accuracy": top1_hits / n,
    "top1_hits": top1_hits,
    "pairwise_accuracy": (pair_agree / pair_total) if pair_total else None,
    "pairwise_agree": pair_agree,
    "pairwise_total": pair_total,
    "mean_score_length_spearman": (
      sum(length_corrs) / len(length_corrs) if length_corrs else None
    ),
  }

  report = {"summary": summary, "items": per_item}
  report_path = Path(args.report)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )

  md_path = report_path.with_suffix(".md")
  lines = [
    "# Hard Eval report",
    "",
    f"- scorer: `{summary['scorer']}`",
    f"- model: `{summary['model']}`",
    f"- n: {summary['n_labeled']}",
    f"- top1_accuracy: {summary['top1_accuracy']:.3f} ({summary['top1_hits']}/{summary['n_labeled']})",
  ]
  if summary["pairwise_accuracy"] is not None:
    lines.append(
      f"- pairwise_accuracy: {summary['pairwise_accuracy']:.3f} "
      f"({summary['pairwise_agree']}/{summary['pairwise_total']})"
    )
  if summary["mean_score_length_spearman"] is not None:
    lines.append(
      f"- mean score↔length Spearman: {summary['mean_score_length_spearman']:.3f}"
    )
  lines.append("")
  lines.append("| id | human best | model best | hit |")
  lines.append("|---|---|---|:---:|")
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
