#!/usr/bin/env python3
"""B の学習側三つ組について、比較できないを許す付け直し用の総当たり対を作る。

一件の候補は下書き、人間の推敲、Composer の推敲。各 item ちょうど 3 対。
劣化と付けた学習側 1 件も含める。A2 の検証 50 件は出さない。
"""
from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from section_middle_utils import (
  CHOICE_DEGRADED,
  corpus_split,
  judgments_by_id,
  load_jsonl,
  revisions_by_id,
  strip_code_fence,
)
from setwise_triple_utils import SetwiseTriple, reconstruct_triples_from_pref_rows

COMPARE_TYPE = "b_rejudge"
SRC_DRAFT = "draft"
SRC_HUMAN = "human"
SRC_COMPOSER = "composer"
QUESTION = "A と B のどちらが、自分の推敲に近いか"

COMPARE_BY_SOURCES = {
  frozenset({SRC_HUMAN, SRC_DRAFT}): "gold_vs_draft",
  frozenset({SRC_HUMAN, SRC_COMPOSER}): "gold_vs_composer",
  frozenset({SRC_COMPOSER, SRC_DRAFT}): "composer_vs_draft",
}


def item_ids_from_pref(rows: list[dict]) -> set[str]:
  out: set[str] = set()
  for row in rows:
    meta = row.get("meta") or {}
    iid = str(meta.get("item_id") or "").strip()
    if iid:
      out.add(iid)
  return out


def _triple_from_parts(
  *,
  item_id: str,
  draft: str,
  human: str,
  composer: str,
  meta: dict | None = None,
) -> SetwiseTriple | None:
  draft = draft.strip()
  human = human.strip()
  composer = composer.strip()
  if not draft or not human or not composer:
    return None
  if human == composer or human == draft or composer == draft:
    return None
  return SetwiseTriple(
    item_id=item_id,
    source_text=draft,
    draft=draft,
    human=human,
    composer=composer,
    meta=meta or {},
  )


def collect_train_triples(
  *,
  train_file: Path,
  valid_file: Path,
  items_file: Path,
  revisions_file: Path,
  judgments_file: Path,
  include_degraded: bool = True,
) -> tuple[list[SetwiseTriple], dict]:
  train_rows = load_jsonl(train_file)
  valid_ids = item_ids_from_pref(load_jsonl(valid_file))
  triples = reconstruct_triples_from_pref_rows(train_rows)
  triple_by_id = {t.item_id: t for t in triples}

  stats = {
    "n_pref_train_items": len(triple_by_id),
    "n_valid_excluded": len(valid_ids),
    "n_degraded_added": 0,
    "degraded_item_ids": [],
  }

  if include_degraded:
    items_by_id = {str(i.get("id") or ""): i for i in load_jsonl(items_file) if i.get("id")}
    revisions = revisions_by_id(load_jsonl(revisions_file))
    judgments = judgments_by_id(load_jsonl(judgments_file))
    for item_id, jud in sorted(judgments.items()):
      if str(jud.get("choice") or "") != CHOICE_DEGRADED:
        continue
      item = items_by_id.get(item_id)
      if not item:
        continue
      if corpus_split(item) != "train":
        continue
      rev = revisions.get(item_id)
      if not rev:
        continue
      triple = _triple_from_parts(
        item_id=item_id,
        draft=str(item.get("source_text") or ""),
        human=str(item.get("edited_text") or ""),
        composer=str(rev.get("text") or ""),
        meta=dict(item.get("meta") or {}),
      )
      if triple is None:
        continue
      if item_id in triple_by_id:
        continue
      triple_by_id[item_id] = triple
      stats["n_degraded_added"] += 1
      stats["degraded_item_ids"].append(item_id)

  out = [triple_by_id[iid] for iid in sorted(triple_by_id) if iid not in valid_ids]
  stats["n_items"] = len(out)
  stats["n_overlap_with_valid"] = len(set(triple_by_id) & valid_ids)
  return out, stats


def build_pairs(triples: list[SetwiseTriple], *, seed: int) -> list[dict]:
  pair_rng = random.Random(seed + 1)
  shuffle_rng = random.Random(seed + 2)
  pairs: list[dict] = []
  for triple in sorted(triples, key=lambda t: t.item_id):
    cands = [
      (SRC_DRAFT, triple.draft),
      (SRC_HUMAN, triple.human),
      (SRC_COMPOSER, triple.composer),
    ]
    for (src_a, text_a), (src_b, text_b) in combinations(cands, 2):
      swap = pair_rng.random() < 0.5
      left_src, right_src = src_a, src_b
      left_text, right_text = text_a, text_b
      if swap:
        left_src, right_src = src_b, src_a
        left_text, right_text = text_b, text_a
      compare_type = COMPARE_BY_SOURCES[frozenset({src_a, src_b})]
      pairs.append(
        {
          "pair_id": f"{triple.item_id}::{compare_type}",
          "item_id": triple.item_id,
          "compare_type": compare_type,
          "context_draft": triple.draft,
          "a_text": left_text,
          "b_text": right_text,
          "a_source": left_src,
          "b_source": right_src,
          "swapped": swap,
          "question": QUESTION,
        }
      )
  shuffle_rng.shuffle(pairs)
  for i, row in enumerate(pairs):
    row["order"] = i
  return pairs


def write_protocol(path: Path, *, seed: int, stats: dict) -> None:
  protocol = {
    "n_items": stats.get("n_items"),
    "n_pairs_per_item": 3,
    "n_pairs": stats.get("n_pairs"),
    "seed": seed,
    "source_train": "data/section_middle/pref_train.jsonl",
    "source_valid_excluded": "data/section_middle/pref_valid.jsonl",
    "source_items": "data/revision_corpus/keep_section.jsonl",
    "source_revisions": "data/section_middle/revisions.jsonl",
    "source_judgments": "data/section_middle/judgments.jsonl",
    "include_degraded_train": True,
    "compare": "draft, human, composer の総当たり 3 対",
    "judgment": {
      "view": "左右の役割を見せない。下書きは文脈として出す。人間の推敲も候補として出す。",
      "question": QUESTION,
      "options": ["a", "b", "tie", "incomparable"],
      "incomparable_reasons": [
        "both_worse",
        "noedit_vs_worse",
        "broken_vs_worse",
        "veto",
        "other",
      ],
    },
    "pairs_file": "data/section_middle/pairs_rejudge.jsonl",
    "judgments_file": "data/section_middle/judgments_rejudge.jsonl",
    "stats": stats,
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-file", default="data/section_middle/pref_train.jsonl")
  parser.add_argument("--valid-file", default="data/section_middle/pref_valid.jsonl")
  parser.add_argument("--items", default="data/revision_corpus/keep_section.jsonl")
  parser.add_argument("--revisions", default="data/section_middle/revisions.jsonl")
  parser.add_argument("--judgments", default="data/section_middle/judgments.jsonl")
  parser.add_argument("--out", default="data/section_middle/pairs_rejudge.jsonl")
  parser.add_argument(
    "--protocol-out",
    default="data/section_middle/b_rejudge_protocol.json",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--no-degraded",
    action="store_true",
    help="劣化 1 件を含めない（テスト用）",
  )
  args = parser.parse_args()

  triples, stats = collect_train_triples(
    train_file=Path(args.train_file),
    valid_file=Path(args.valid_file),
    items_file=Path(args.items),
    revisions_file=Path(args.revisions),
    judgments_file=Path(args.judgments),
    include_degraded=not args.no_degraded,
  )
  if not triples:
    raise SystemExit("no train triples")
  if stats["n_overlap_with_valid"]:
    raise SystemExit(
      f"train items overlap with valid: {stats['n_overlap_with_valid']}"
    )

  pairs = build_pairs(triples, seed=args.seed)
  stats["n_pairs"] = len(pairs)
  stats["n_swapped"] = sum(1 for row in pairs if row["swapped"])

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for row in pairs:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  write_protocol(Path(args.protocol_out), seed=args.seed, stats=stats)
  print(
    f"wrote {out} items={stats['n_items']} pairs={len(pairs)} "
    f"degraded_added={stats['n_degraded_added']} "
    f"valid_overlap={stats['n_overlap_with_valid']} seed={args.seed}",
    flush=True,
  )


if __name__ == "__main__":
  main()
