from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from generated_pref_utils import (  # noqa: E402
  SELECTION_BIASED_COMPARE_TYPES,
  assign_item_folds,
  build_fold_splits,
  merge_blind_to_teacher_rows,
  verify_fold_splits,
  write_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
  write_jsonl(path, rows)


def _synthetic_blind_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
  """短い人工文だけの 12 item / 72 比較（tie 含む）。"""
  items = []
  for i in range(12):
    unit = "section" if i < 10 else "hunk"
    items.append(
      {
        "id": f"item-{i}",
        "project_id": "proj",
        "unit": unit,
        "draft": f"下書き{i}。",
        "gold": f"推敲{i}。",
      }
    )

  pairs = []
  judgments = []
  compare_types = [
    "1_draft_vs_base_greedy",
    "2_draft_vs_adapter_greedy",
    "3_base_vs_adapter_greedy",
    "4_draft_vs_adapter_selected",
    "5_gold_vs_adapter_selected",
    "6_gold_vs_draft",
  ]
  choice_cycle = ["a", "b", "tie", "a", "b", "tie"]
  for item in items:
    for j, ctype in enumerate(compare_types):
      pair_id = f"{item['id']}::{ctype}"
      broken = ctype.startswith("3_")
      pairs.append(
        {
          "pair_id": pair_id,
          "item_id": item["id"],
          "compare_type": ctype,
          "context_draft": item["draft"],
          "a_text": f"A-{item['id']}-{ctype}",
          "b_text": f"B-{item['id']}-{ctype}",
          "a_source": "draft",
          "b_source": "adapter_greedy",
          "swapped": False,
        }
      )
      judgments.append(
        {
          "pair_id": pair_id,
          "item_id": item["id"],
          "compare_type": ctype,
          "choice": choice_cycle[j % len(choice_cycle)],
          "a_broken": broken,
          "b_broken": False,
          "a_noedit": False,
          "b_noedit": False,
        }
      )

  items_path = tmp_path / "items.jsonl"
  pairs_path = tmp_path / "pairs.jsonl"
  judgments_path = tmp_path / "judgments.jsonl"
  _write_jsonl(items_path, items)
  _write_jsonl(pairs_path, pairs)
  _write_jsonl(judgments_path, judgments)
  return pairs_path, judgments_path, items_path


def test_merge_keeps_all_rows_including_tie_and_broken(tmp_path: Path) -> None:
  pairs_path, judgments_path, items_path = _synthetic_blind_fixture(tmp_path)
  pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
  judgments = [json.loads(line) for line in judgments_path.read_text().splitlines()]
  items = [json.loads(line) for line in items_path.read_text().splitlines()]

  rows = merge_blind_to_teacher_rows(pairs, judgments, items)
  assert len(rows) == 72
  ties = [r for r in rows if r["preference"] == "tie"]
  assert len(ties) == 24
  broken = [r for r in rows if r["a_broken"] or r["b_broken"]]
  assert broken, "broken 行が落ちている"
  biased = [r for r in rows if r["selection_biased"]]
  assert len(biased) == 24
  assert all(r["compare_type"] in SELECTION_BIASED_COMPARE_TYPES for r in biased)


def test_fold_splits_no_item_overlap(tmp_path: Path) -> None:
  pairs_path, judgments_path, items_path = _synthetic_blind_fixture(tmp_path)
  pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
  judgments = [json.loads(line) for line in judgments_path.read_text().splitlines()]
  items = [json.loads(line) for line in items_path.read_text().splitlines()]
  rows = merge_blind_to_teacher_rows(pairs, judgments, items)
  item_to_fold = assign_item_folds(items, n_folds=5, seed=42)
  folds = build_fold_splits(rows, item_to_fold, n_folds=5)
  verify_fold_splits(folds, n_folds=5)

  seen_valid: set[str] = set()
  for train_rows, valid_rows in folds:
    train_items = {r["item_id"] for r in train_rows}
    valid_items = {r["item_id"] for r in valid_rows}
    assert not (train_items & valid_items)
    assert not (seen_valid & valid_items)
    seen_valid |= valid_items
  assert len(seen_valid) == len(items)


def test_real_blind_data_counts_if_present() -> None:
  pairs_path = ROOT / "data" / "blind_eval" / "pairs.jsonl"
  if not pairs_path.is_file():
    pytest.skip("blind eval data not available")
  judgments_path = ROOT / "data" / "blind_eval" / "judgments.jsonl"
  items_path = ROOT / "data" / "blind_eval" / "items.jsonl"
  pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
  judgments = [json.loads(line) for line in judgments_path.read_text().splitlines()]
  items = [json.loads(line) for line in items_path.read_text().splitlines()]
  rows = merge_blind_to_teacher_rows(pairs, judgments, items)
  assert len(rows) == 360
  assert sum(1 for r in rows if r["preference"] == "tie") == 68
  assert sum(1 for r in rows if r["selection_biased"]) == 120
