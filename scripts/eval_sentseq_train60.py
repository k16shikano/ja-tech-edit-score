#!/usr/bin/env python3
"""keep 節 train と三つ組み train の共通節から 60 件を取り、文列型で点を付ける。

工程 6 の 60 件（人間の推敲対選抜生成）とは別の取り出しである。
学習側の節だけを使う。検証 50 節は入らない。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from eval_pref_multigranular import _sentseq_score_fn, _strictly_greater
from pref_static_utils import load_jsonl
from setwise_triple_utils import reconstruct_triples_from_pref_rows

KEEP_TRAIN = "data/pref_keep_split_section/train.jsonl"
PREF_TRAIN = "data/section_middle/pref_train.jsonl"
KEEP_MODEL = "outputs/pref-multigranular/pref-sentseq-keep-pairsplit"
TRIPLES_MODEL = "outputs/pref-sentseq-section-triples"


def keep_train_item_ids(rows: list[dict]) -> set[str]:
  out: set[str] = set()
  for row in rows:
    meta = row.get("meta") or {}
    if str(meta.get("pair_order") or "chosen_first") != "chosen_first":
      continue
    item_id = str(meta.get("base_id") or "").strip()
    if item_id:
      out.add(item_id)
  return out


def sample_train_triples(
  *,
  keep_rows: list[dict],
  pref_rows: list[dict],
  n: int,
  seed: int,
) -> list:
  keep_ids = keep_train_item_ids(keep_rows)
  triples = [
    t for t in reconstruct_triples_from_pref_rows(pref_rows) if t.item_id in keep_ids
  ]
  triples.sort(key=lambda t: t.item_id)
  if n > len(triples):
    raise ValueError(f"need {n} triples, have {len(triples)} in keep∩pref train")
  rng = random.Random(seed)
  return rng.sample(triples, n)


def score_triples(score_fn, triples: list) -> list[dict]:
  rows: list[dict] = []
  for triple in triples:
    scores = score_fn(triple.source_text, [triple.human, triple.composer, triple.draft])
    human, composer, draft = (float(v) for v in scores)
    rows.append(
      {
        "item_id": triple.item_id,
        "scores": {"human": human, "composer": composer, "draft": draft},
        "human_over_draft": _strictly_greater(human, draft),
        "human_over_composer": _strictly_greater(human, composer),
      }
    )
  return rows


def summarize(rows: list[dict]) -> dict:
  n = len(rows)
  over_draft = sum(1 for r in rows if r["human_over_draft"])
  over_composer = sum(1 for r in rows if r["human_over_composer"])
  return {
    "n": n,
    "human_over_draft": over_draft,
    "human_over_composer": over_composer,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--keep-train", default=KEEP_TRAIN)
  parser.add_argument("--pref-train", default=PREF_TRAIN)
  parser.add_argument("--keep-model", default=KEEP_MODEL)
  parser.add_argument("--triples-model", default=TRIPLES_MODEL)
  parser.add_argument("--n", type=int, default=60)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--output",
    default="outputs/sentseq_train60.json",
  )
  args = parser.parse_args()

  keep_rows = load_jsonl(args.keep_train)
  pref_rows = load_jsonl(args.pref_train)
  triples = sample_train_triples(
    keep_rows=keep_rows,
    pref_rows=pref_rows,
    n=args.n,
    seed=args.seed,
  )

  models = [
    ("keep_pairsplit", Path(args.keep_model)),
    ("section_triples", Path(args.triples_model)),
  ]
  runs = []
  for name, model_dir in models:
    if not (model_dir / "model.pt").is_file():
      raise SystemExit(f"missing {model_dir / 'model.pt'}")
    print(f"scoring {name} n={len(triples)}", flush=True)
    rows = score_triples(_sentseq_score_fn(model_dir), triples)
    summary = summarize(rows)
    print(
      f"{name} human>draft {summary['human_over_draft']}/{summary['n']} "
      f"human>composer {summary['human_over_composer']}/{summary['n']}",
      flush=True,
    )
    runs.append({"name": name, "model": str(model_dir), **summary, "items": rows})

  out = {
    "keep_train": args.keep_train,
    "pref_train": args.pref_train,
    "n": args.n,
    "seed": args.seed,
    "item_ids": [t.item_id for t in triples],
    "runs": runs,
  }
  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
  main()
