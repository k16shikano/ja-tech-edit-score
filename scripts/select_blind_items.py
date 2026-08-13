#!/usr/bin/env python3
"""検証ペアから人手ブラインド判定用の 60 件を選ぶ。

節ペアを優先し、下書きが 100 文字未満のものは除く。シード固定の乱択。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def parse_chat(obj: dict) -> dict | None:
  msgs = obj.get("messages") or []
  if len(msgs) < 2:
    return None
  user = str(msgs[0].get("content") or "")
  draft = user.split("\n\n", 1)[-1] if "\n\n" in user else user
  revised = str(msgs[1].get("content") or "")
  meta = obj.get("meta") or {}
  return {
    "id": str(meta.get("id") or ""),
    "project_id": str(meta.get("project_id") or ""),
    "unit": str(meta.get("unit") or ""),
    "draft": draft,
    "gold": revised,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--heldout", default="data/edit_sft_all/heldout.jsonl")
  parser.add_argument("--out", default="data/blind_eval/items.jsonl")
  parser.add_argument("--n", type=int, default=60)
  parser.add_argument("--min-chars", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--prefer-section",
    action="store_true",
    default=True,
    help="節ペアを先に埋める（既定オン）",
  )
  parser.add_argument("--no-prefer-section", action="store_false", dest="prefer_section")
  args = parser.parse_args()

  heldout_path = Path(args.heldout)
  if not heldout_path.is_file():
    raise SystemExit(f"missing {heldout_path}")

  candidates: list[dict] = []
  for obj in load_jsonl(heldout_path):
    row = parse_chat(obj)
    if not row or not row["id"]:
      continue
    if len(row["draft"]) < args.min_chars:
      continue
    candidates.append(row)

  rng = random.Random(args.seed)
  selected: list[dict] = []
  if args.prefer_section:
    sections = [c for c in candidates if c.get("unit") == "section"]
    hunks = [c for c in candidates if c.get("unit") != "section"]
    rng.shuffle(sections)
    rng.shuffle(hunks)
    selected = sections[: args.n]
    if len(selected) < args.n:
      selected.extend(hunks[: args.n - len(selected)])
  else:
    rng.shuffle(candidates)
    selected = candidates[: args.n]

  if len(selected) < args.n:
    raise SystemExit(
      f"only {len(selected)} candidates (need {args.n}); "
      f"pool={len(candidates)} min_chars={args.min_chars}"
    )

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for row in selected:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  n_sec = sum(1 for r in selected if r.get("unit") == "section")
  print(
    f"wrote {out} n={len(selected)} section={n_sec} "
    f"hunk={len(selected) - n_sec} seed={args.seed}"
  )


if __name__ == "__main__":
  main()
