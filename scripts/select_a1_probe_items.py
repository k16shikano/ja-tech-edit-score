#!/usr/bin/env python3
"""A1 学習側から、三群生成の対象件を乱択する。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def row_id(obj: dict) -> str:
  return str(obj.get("id") or (obj.get("meta") or {}).get("id") or "").strip()


def load_train_ids(path: Path) -> list[str]:
  ids: list[str] = []
  seen: set[str] = set()
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rid = row_id(json.loads(line))
      if not rid or rid in seen:
        continue
      seen.add(rid)
      ids.append(rid)
  return ids


def pick_ids(ids: list[str], *, n: int, seed: int) -> list[str]:
  ordered = sorted(ids)
  rng = random.Random(seed)
  if n >= len(ordered):
    rng.shuffle(ordered)
    return ordered
  return sorted(rng.sample(ordered, n))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--train",
    default="data/edit_sft_hunk_nopara/train.jsonl",
  )
  parser.add_argument("--out", default="data/a1_probe/ids.jsonl")
  parser.add_argument("--protocol-out", default="data/a1_probe/protocol.json")
  parser.add_argument("--n-items", type=int, default=200)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  train_path = Path(args.train)
  if not train_path.is_file():
    raise SystemExit(f"missing {train_path}")
  ids = load_train_ids(train_path)
  if not ids:
    raise SystemExit(f"no ids in {train_path}")
  picked = pick_ids(ids, n=args.n_items, seed=args.seed)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for rid in picked:
      f.write(json.dumps({"id": rid}, ensure_ascii=False) + "\n")

  protocol = {
    "n_train": len(ids),
    "n_items": len(picked),
    "seed": args.seed,
    "train_file": str(train_path),
    "ids_file": str(out),
    "modes": ["base", "base_norms", "adapter"],
    "norms": "japanese-tech-writing",
    "adapter_norms": False,
  }
  Path(args.protocol_out).write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"wrote {len(picked)} ids -> {out}")


if __name__ == "__main__":
  main()
