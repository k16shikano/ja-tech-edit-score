#!/usr/bin/env python3
"""スカラー仮説検査の人手判定を、件ごとの推移性で集計する。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from pref_static_utils import load_jsonl


def _winner_source(row: dict) -> str | None:
  choice = str(row.get("choice") or "")
  if choice == "tie":
    return None
  if choice == "a":
    return str(row.get("a_source") or "")
  if choice == "b":
    return str(row.get("b_source") or "")
  return None


def _loser_source(row: dict) -> str | None:
  choice = str(row.get("choice") or "")
  if choice == "tie":
    return None
  if choice == "a":
    return str(row.get("b_source") or "")
  if choice == "b":
    return str(row.get("a_source") or "")
  return None


def has_strict_cycle(nodes: list[str], wins: list[tuple[str, str]]) -> bool:
  adj: dict[str, list[str]] = {n: [] for n in nodes}
  for winner, loser in wins:
    if winner in adj and loser in adj:
      adj[winner].append(loser)
  color = {n: 0 for n in nodes}

  def dfs(u: str) -> bool:
    color[u] = 1
    for v in adj[u]:
      if color[v] == 1:
        return True
      if color[v] == 0 and dfs(v):
        return True
    color[u] = 2
    return False

  return any(dfs(n) for n in nodes if color[n] == 0)


def classify_item(rows: list[dict]) -> dict:
  nodes: set[str] = set()
  for row in rows:
    nodes.add(str(row.get("a_source") or ""))
    nodes.add(str(row.get("b_source") or ""))
  node_list = sorted(n for n in nodes if n)
  expected_pairs = len(node_list) * (len(node_list) - 1) // 2
  wins: list[tuple[str, str]] = []
  n_tie = 0
  n_incomparable = 0
  reasons: dict[str, int] = defaultdict(int)
  judged_pairs: set[frozenset[str]] = set()
  for row in rows:
    pair = frozenset(
      {str(row.get("a_source") or ""), str(row.get("b_source") or "")}
    )
    judged_pairs.add(pair)
    choice = str(row.get("choice") or "")
    if choice == "incomparable":
      n_incomparable += 1
      reason = str(row.get("incomparable_reason") or "other")
      reasons[reason] += 1
      continue
    winner = _winner_source(row)
    loser = _loser_source(row)
    if winner is None:
      n_tie += 1
      continue
    wins.append((winner, loser or ""))

  incomplete = len(judged_pairs) < expected_pairs
  incomparable = n_incomparable > 0
  cyclic = (not incomplete) and (not incomparable) and has_strict_cycle(node_list, wins)
  draft_wins = sum(1 for w, l in wins if w == "draft")
  draft_losses = sum(1 for w, l in wins if l == "draft")
  return {
    "item_id": rows[0].get("item_id") if rows else "",
    "n_judged": len(rows),
    "n_expected": expected_pairs,
    "n_tie": n_tie,
    "n_incomparable": n_incomparable,
    "incomparable_reasons": dict(reasons),
    "incomplete": incomplete,
    "incomparable": incomparable,
    "cyclic": cyclic,
    "transitive": (not incomplete) and (not cyclic) and (not incomparable),
    "draft_strict_wins": draft_wins,
    "draft_strict_losses": draft_losses,
  }


def analyze(pairs: list[dict], judgments: list[dict]) -> dict:
  by_item: dict[str, list[dict]] = defaultdict(list)
  pair_by_id = {str(p.get("pair_id") or ""): p for p in pairs}
  for row in judgments:
    pid = str(row.get("pair_id") or "")
    p = pair_by_id.get(pid)
    if not p:
      continue
    merged = dict(p)
    merged.update(row)
    by_item[str(merged.get("item_id") or "")].append(merged)

  item_rows = [classify_item(rs) for _, rs in sorted(by_item.items())]
  n_complete = sum(1 for r in item_rows if not r["incomplete"])
  n_cyclic = sum(1 for r in item_rows if r["cyclic"])
  n_transitive = sum(1 for r in item_rows if r["transitive"])
  n_incomparable_items = sum(1 for r in item_rows if r["incomparable"])
  n_incomparable_pairs = sum(int(r["n_incomparable"]) for r in item_rows)
  reason_counts: dict[str, int] = defaultdict(int)
  for r in item_rows:
    for k, n in (r.get("incomparable_reasons") or {}).items():
      reason_counts[str(k)] += int(n)
  return {
    "n_pairs": len(pairs),
    "n_judgments": len(judgments),
    "n_items_judged": len(item_rows),
    "n_complete_items": n_complete,
    "n_transitive": n_transitive,
    "n_cyclic": n_cyclic,
    "n_incomparable_items": n_incomparable_items,
    "n_incomparable_pairs": n_incomparable_pairs,
    "incomparable_reasons": dict(reason_counts),
    "items": item_rows,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--pairs",
    default="data/blind_eval/pairs_scalar_transitivity.jsonl",
  )
  parser.add_argument(
    "--judgments",
    default="data/blind_eval/judgments_scalar_transitivity.jsonl",
  )
  parser.add_argument(
    "--out",
    default="data/blind_eval/scalar_transitivity_analysis.json",
  )
  args = parser.parse_args()

  pairs = load_jsonl(Path(args.pairs))
  judgments = load_jsonl(Path(args.judgments))
  if not pairs:
    raise SystemExit(f"no pairs in {args.pairs}")
  report = analyze(pairs, judgments)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  public = {k: v for k, v in report.items() if k != "items"}
  out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(
    f"items_judged={public['n_items_judged']} complete={public['n_complete_items']} "
    f"transitive={public['n_transitive']} cyclic={public['n_cyclic']} "
    f"incomparable_items={public['n_incomparable_items']} "
    f"incomparable_pairs={public['n_incomparable_pairs']} "
    f"pairs={public['n_pairs']} judgments={public['n_judgments']}",
    flush=True,
  )


if __name__ == "__main__":
  main()
