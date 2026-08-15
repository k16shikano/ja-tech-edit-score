from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from eval_sentseq_train60 import keep_train_item_ids, sample_train_triples, summarize


def test_keep_train_item_ids_skips_swap() -> None:
  rows = [
    {"meta": {"base_id": "a", "pair_order": "chosen_first"}},
    {"meta": {"base_id": "a", "pair_order": "rejected_first"}},
    {"meta": {"base_id": "b", "pair_order": "chosen_first"}},
  ]
  assert keep_train_item_ids(rows) == {"a", "b"}


def test_summarize_counts() -> None:
  rows = [
    {"human_over_draft": True, "human_over_composer": True},
    {"human_over_draft": True, "human_over_composer": False},
    {"human_over_draft": False, "human_over_composer": False},
  ]
  s = summarize(rows)
  assert s == {"n": 3, "human_over_draft": 2, "human_over_composer": 1}


def test_sample_train_triples_reproducible_and_in_keep() -> None:
  keep = ROOT / "data/pref_keep_split_section/train.jsonl"
  pref = ROOT / "data/section_middle/pref_train.jsonl"
  from pref_static_utils import load_jsonl

  keep_rows = load_jsonl(str(keep))
  pref_rows = load_jsonl(str(pref))
  a = sample_train_triples(keep_rows=keep_rows, pref_rows=pref_rows, n=60, seed=0)
  b = sample_train_triples(keep_rows=keep_rows, pref_rows=pref_rows, n=60, seed=0)
  assert [t.item_id for t in a] == [t.item_id for t in b]
  keep_ids = keep_train_item_ids(keep_rows)
  assert all(t.item_id in keep_ids for t in a)
  assert len({t.item_id for t in a}) == 60
