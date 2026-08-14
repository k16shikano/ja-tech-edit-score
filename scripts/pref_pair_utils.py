#!/usr/bin/env python3
"""節三つ組みと段落内ペアから、一対学習用の例を作る。"""
from __future__ import annotations

from dataclasses import dataclass

from setwise_triple_utils import (
  ROLE_COMPOSER,
  ROLE_DRAFT,
  ROLE_HUMAN,
  SetwiseTriple,
)


@dataclass(frozen=True)
class PairExample:
  item_id: str
  source_text: str
  candidates: tuple[str, ...]
  roles: tuple[str, ...]
  unit: str

  @property
  def human_index(self) -> int:
    return self.roles.index(ROLE_HUMAN)

  @property
  def composer_index(self) -> int | None:
    if ROLE_COMPOSER not in self.roles:
      return None
    return self.roles.index(ROLE_COMPOSER)

  @property
  def draft_index(self) -> int | None:
    if ROLE_DRAFT not in self.roles:
      return None
    return self.roles.index(ROLE_DRAFT)


def examples_from_triples(
  triples: list[SetwiseTriple],
  *,
  include_composer: bool,
) -> list[PairExample]:
  out: list[PairExample] = []
  for triple in triples:
    if include_composer:
      candidates = (triple.human, triple.composer, triple.draft)
      roles = (ROLE_HUMAN, ROLE_COMPOSER, ROLE_DRAFT)
    else:
      candidates = (triple.human, triple.draft)
      roles = (ROLE_HUMAN, ROLE_DRAFT)
    out.append(
      PairExample(
        item_id=triple.item_id,
        source_text=triple.source_text,
        candidates=candidates,
        roles=roles,
        unit="section",
      )
    )
  return out


def examples_from_hunk_rows(rows: list[dict]) -> list[PairExample]:
  out: list[PairExample] = []
  for row in rows:
    meta = row.get("meta") or {}
    if str(meta.get("pair_order") or "chosen_first") != "chosen_first":
      continue
    if int(row.get("label", 0)) != 1:
      continue
    chosen = str(row.get("candidate_a") or "")
    rejected = str(row.get("candidate_b") or "")
    source = str(row.get("source_text") or "")
    if not source.strip() or not chosen.strip() or not rejected.strip():
      continue
    if chosen == rejected:
      continue
    out.append(
      PairExample(
        item_id=str(row.get("id") or ""),
        source_text=source,
        candidates=(chosen, rejected),
        roles=(ROLE_HUMAN, ROLE_DRAFT),
        unit="hunk",
      )
    )
  return out


def unique_texts_from_examples(examples: list[PairExample]) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for ex in examples:
    for text in (ex.source_text, *ex.candidates):
      if text not in seen:
        seen.add(text)
        texts.append(text)
  return texts
