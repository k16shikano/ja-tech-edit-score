from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from train_pref_gpm import (
  count_hunk_draft_over_gold,
  count_section_cycles,
  count_section_draft_over_gen,
  count_section_draft_over_gold,
  count_section_gen_over_gold,
  load_hunk_preference_rows,
  load_triple_preference_rows,
  pairwise_preference_score,
  section_has_preference_cycle,
  validate_head_dim,
)


def test_pairwise_preference_score_skew_symmetry() -> None:
  torch.manual_seed(0)
  v_a = torch.randn(8, 4)
  v_b = torch.randn(8, 4)
  s_ab = pairwise_preference_score(v_a, v_b)
  s_ba = pairwise_preference_score(v_b, v_a)
  assert torch.allclose(s_ab, -s_ba, atol=1e-5)
  s_aa = pairwise_preference_score(v_a, v_a)
  assert torch.allclose(s_aa, torch.zeros_like(s_aa), atol=1e-5)


def test_validate_head_dim_rejects_odd() -> None:
  with pytest.raises(ValueError, match="even"):
    validate_head_dim(3)


def test_load_triple_preference_rows_kinds_and_drop() -> None:
  rows = [
    {
      "pair_kind": "gold_vs_draft",
      "label": 1,
      "candidate_a": "g",
      "candidate_b": "d",
    },
    {
      "pair_kind": "gold_vs_gen",
      "label": 1,
      "candidate_a": "g",
      "candidate_b": "c",
    },
    {
      "pair_kind": "gen_vs_draft",
      "label": 1,
      "candidate_a": "c",
      "candidate_b": "d",
    },
    {"pair_kind": "other", "label": 1},
  ]
  loaded = load_triple_preference_rows(rows)
  assert [r["pair_kind"] for r in loaded] == [
    "gold_vs_draft",
    "gold_vs_gen",
    "gen_vs_draft",
  ]
  dropped = load_triple_preference_rows(rows, drop_gen_over_draft=True)
  assert [r["pair_kind"] for r in dropped] == ["gold_vs_draft", "gold_vs_gen"]


def test_load_hunk_preference_rows_drops_swap() -> None:
  rows = [
    {
      "meta": {"pair_order": "chosen_first"},
      "label": 1,
      "source_text": "s",
      "candidate_a": "a",
      "candidate_b": "b",
    },
    {
      "meta": {"pair_order": "rejected_first"},
      "label": 0,
      "source_text": "s",
      "candidate_a": "b",
      "candidate_b": "a",
    },
    {
      "meta": {"pair_order": "chosen_first"},
      "label": 0,
      "source_text": "s",
      "candidate_a": "a",
      "candidate_b": "b",
    },
  ]
  loaded = load_hunk_preference_rows(rows)
  assert len(loaded) == 1
  assert loaded[0]["candidate_a"] == "a"


def test_section_has_preference_cycle() -> None:
  assert section_has_preference_cycle(1.0, 1.0, 1.0) is False
  assert section_has_preference_cycle(-1.0, 1.0, 1.0) is True
  assert section_has_preference_cycle(1.0, -1.0, -1.0) is True


def test_count_draft_over_gold_and_gen_over_gold() -> None:
  scores = [1.0, -0.5, 0.0, 2.0]
  assert count_hunk_draft_over_gold(scores) == 2
  assert count_section_draft_over_gold(scores) == 2
  assert count_section_gen_over_gold([-1.0, 0.0, 0.5]) == 2


def test_count_section_draft_over_gen() -> None:
  assert count_section_draft_over_gen([1.0, 0.0, -1.0]) == 2


def test_count_section_cycles() -> None:
  cyclic = count_section_cycles([-1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0])
  assert cyclic == 2
  transitive = count_section_cycles([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
  assert transitive == 0
