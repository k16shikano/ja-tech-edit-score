#!/usr/bin/env python3
"""三つ組み preference 行から setwise 学習例を復元する。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

REQUIRED_PAIR_KINDS = ("gold_vs_draft", "gold_vs_gen", "gen_vs_draft")
ROLE_HUMAN = "human"
ROLE_COMPOSER = "composer"
ROLE_DRAFT = "draft"
CANONICAL_ROLES = (ROLE_HUMAN, ROLE_COMPOSER, ROLE_DRAFT)


@dataclass(frozen=True)
class SetwiseTriple:
  item_id: str
  source_text: str
  draft: str
  human: str
  composer: str
  meta: dict

  def texts_by_role(self) -> dict[str, str]:
    return {
      ROLE_DRAFT: self.draft,
      ROLE_HUMAN: self.human,
      ROLE_COMPOSER: self.composer,
    }

  def canonical_candidates(self) -> list[str]:
    return [self.human, self.composer, self.draft]

  def canonical_rank_indices(self) -> list[int]:
    """best-first indices into canonical_candidates()."""
    return [0, 1, 2]


class TripleReconstructionError(ValueError):
  pass


def _item_id(row: dict) -> str:
  meta = row.get("meta") or {}
  item_id = str(meta.get("item_id") or "").strip()
  if not item_id:
    raise TripleReconstructionError(f"missing meta.item_id in row {row.get('id')!r}")
  return item_id


def _expect_chosen_a(row: dict, *, context: str) -> None:
  if int(row.get("label", 0)) != 1:
    raise TripleReconstructionError(f"{context}: label must be 1")
  if str(row.get("preference") or "") != "a":
    raise TripleReconstructionError(f"{context}: preference must be 'a'")


def _expect_distinct_roles(
  *,
  human: str,
  composer: str,
  draft: str,
  context: str,
) -> None:
  if human == composer or human == draft or composer == draft:
    raise TripleReconstructionError(
      f"{context}: human, composer, and draft texts must be distinct"
    )


def reconstruct_triples_from_pref_rows(rows: Iterable[dict]) -> list[SetwiseTriple]:
  """section_triple_v1 の 3 行を item ごとに 1 組へ復元する。"""
  grouped: dict[str, dict[str, dict]] = defaultdict(dict)
  meta_by_item: dict[str, dict] = {}
  source_by_item: dict[str, str] = {}

  for row in rows:
    schema = str(row.get("schema") or "")
    if schema and schema != "section_triple_v1":
      raise TripleReconstructionError(
        f"unsupported schema {schema!r} for row {row.get('id')!r}"
      )
    item_id = _item_id(row)
    pair_kind = str(row.get("pair_kind") or "").strip()
    if not pair_kind:
      raise TripleReconstructionError(f"missing pair_kind for item {item_id}")
    if pair_kind in grouped[item_id]:
      raise TripleReconstructionError(f"duplicate pair_kind {pair_kind!r} for item {item_id}")
    grouped[item_id][pair_kind] = row
    source = str(row.get("source_text") or "")
    if item_id in source_by_item and source_by_item[item_id] != source:
      raise TripleReconstructionError(f"inconsistent source_text for item {item_id}")
    source_by_item[item_id] = source
    meta_by_item[item_id] = dict(row.get("meta") or {})

  triples: list[SetwiseTriple] = []
  for item_id, pairs in grouped.items():
    missing = [k for k in REQUIRED_PAIR_KINDS if k not in pairs]
    if missing:
      raise TripleReconstructionError(
        f"missing pair_kind {missing} for item {item_id}"
      )
    extra = sorted(set(pairs) - set(REQUIRED_PAIR_KINDS))
    if extra:
      raise TripleReconstructionError(
        f"unexpected pair_kind {extra} for item {item_id}"
      )

    gvd = pairs["gold_vs_draft"]
    gvg = pairs["gold_vs_gen"]
    gnd = pairs["gen_vs_draft"]
    ctx = f"item {item_id}"
    for row in (gvd, gvg, gnd):
      _expect_chosen_a(row, context=ctx)

    draft = str(gvd["candidate_b"])
    human = str(gvd["candidate_a"])
    composer = str(gnd["candidate_a"])
    if str(gvg["candidate_a"]) != human:
      raise TripleReconstructionError(f"contradictory human role for {ctx}")
    if str(gvg["candidate_b"]) != composer:
      raise TripleReconstructionError(f"contradictory composer role for {ctx}")
    if str(gnd["candidate_b"]) != draft:
      raise TripleReconstructionError(f"contradictory draft role for {ctx}")
    if draft != source_by_item[item_id]:
      raise TripleReconstructionError(
        f"draft must equal source_text for {ctx}"
      )
    _expect_distinct_roles(human=human, composer=composer, draft=draft, context=ctx)

    triples.append(
      SetwiseTriple(
        item_id=item_id,
        source_text=source_by_item[item_id],
        draft=draft,
        human=human,
        composer=composer,
        meta=meta_by_item[item_id],
      )
    )
  triples.sort(key=lambda t: t.item_id)
  return triples
