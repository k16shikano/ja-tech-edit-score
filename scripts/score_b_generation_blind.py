#!/usr/bin/env python3
"""正式評価器による B ブラインド比較の代理判定（人手の代わりには使わない補助）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from pref_scorer import load_scorer


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def judgment_from_scores(score_left: float, score_right: float) -> str:
  if score_left > score_right:
    return "left"
  if score_right > score_left:
    return "right"
  return "tie"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", type=Path, default=Path("data/b_generation/blind/pairs.jsonl"))
  parser.add_argument("--keys", type=Path, default=Path("data/b_generation/blind/keys.jsonl"))
  parser.add_argument(
    "--scorer",
    type=Path,
    default=Path("outputs/pref-sentseq-section-triples"),
  )
  parser.add_argument("--out", type=Path, default=Path("data/b_generation/blind/judgments_scorer_proxy.jsonl"))
  args = parser.parse_args()

  code_root = Path(__file__).resolve().parents[1]
  pairs = {str(p["pair_id"]): p for p in load_jsonl(code_root / args.pairs)}
  keys = load_jsonl(code_root / args.keys)
  scorer_dir = code_root / args.scorer
  if not scorer_dir.is_dir():
    raise SystemExit(f"missing scorer {scorer_dir}")

  loaded = load_scorer(scorer_dir, calibrate_draft_zero=True)
  out_path = code_root / args.out
  out_path.parent.mkdir(parents=True, exist_ok=True)

  rows: list[dict] = []
  for key in keys:
    pid = str(key["pair_id"])
    pair = pairs[pid]
    draft = str(pair["context"]["draft"])
    left_text = str(pair["left"]["text"])
    right_text = str(pair["right"]["text"])
    if not left_text.strip() or not right_text.strip():
      judgment = "unjudgeable"
      note = "empty candidate"
      score_left = score_right = None
    else:
      scores = loaded.score(draft, [left_text, right_text])
      score_left, score_right = float(scores[0]), float(scores[1])
      judgment = judgment_from_scores(score_left, score_right)
      note = "pref-sentseq-section-triples delta"
    rows.append(
      {
        "schema_version": "b-blind-judgment-v1",
        "pair_id": pid,
        "judgment": judgment,
        "source": "scorer_proxy",
        "scorer_kind": loaded.kind,
        "score_left": score_left,
        "score_right": score_right,
        "note": note,
      }
    )

  with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(json.dumps({"wrote": str(out_path), "n": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
  main()
