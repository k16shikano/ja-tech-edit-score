from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from eval_pref_multigranular_blind60 import agreement_for_pairs


def _row(
  pair_id: str,
  *,
  choice: str,
  score_a: float,
  score_b: float,
  a_source: str = "gold",
  b_source: str = "adapter_selected",
) -> dict:
  return {
    "pair_id": pair_id,
    "choice": choice,
    "a_source": a_source,
    "b_source": b_source,
    "score_a": score_a,
    "score_b": score_b,
  }


def test_agreement_all_agree() -> None:
  rows = [
    _row("p1", choice="a", score_a=2.0, score_b=1.0),
    _row("p2", choice="b", score_a=0.5, score_b=1.5),
    _row("p3", choice="a", score_a=3.0, score_b=0.0),
  ]
  stats = agreement_for_pairs(rows)
  assert stats["n_total"] == 3
  assert stats["human_tie"] == 0
  assert stats["human_pref_n"] == 3
  assert stats["scorer_tie"] == 0
  assert stats["comparable_n"] == 3
  assert stats["agree_n"] == 3
  assert stats["agree_rate"] == 1.0
  assert stats["gold_higher_n"] == 2
  assert all(item["agree"] is True for item in stats["items"])


def test_agreement_scorer_tie_excluded_from_denominator() -> None:
  rows = [
    _row("p1", choice="a", score_a=1.0, score_b=1.0),
    _row("p2", choice="b", score_a=0.0, score_b=2.0),
  ]
  stats = agreement_for_pairs(rows)
  assert stats["human_pref_n"] == 2
  assert stats["scorer_tie"] == 1
  assert stats["comparable_n"] == 1
  assert stats["agree_n"] == 1
  assert stats["items"][0]["scorer_pref"] is None
  assert stats["items"][0]["agree"] is None
  assert stats["items"][1]["agree"] is True


def test_agreement_one_disagree() -> None:
  rows = [
    _row("p1", choice="a", score_a=1.0, score_b=2.0),
    _row("p2", choice="b", score_a=0.0, score_b=2.0),
  ]
  stats = agreement_for_pairs(rows)
  assert stats["comparable_n"] == 2
  assert stats["agree_n"] == 1
  assert stats["agree_rate"] == 0.5
  assert stats["items"][0]["agree"] is False
  assert stats["items"][1]["agree"] is True
