from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from analyze_scalar_transitivity import analyze, classify_item
from build_scalar_transitivity_pairs import COMPARE_TYPE, SRC_DRAFT, build_pairs, unelected_rows


def _item(i: int) -> dict:
  return {"id": f"id-{i:02d}", "draft": f"下書き{i}。", "gold": f"人間{i}。"}


def _samples(item_id: str, n: int = 8) -> list[dict]:
  rows = []
  for k in range(n):
    rows.append(
      {
        "id": item_id,
        "sample_index": k,
        "generated": f"選抜文{item_id}。" if k == 0 else f"未選抜{item_id}-{k}。",
      }
    )
  return rows


def test_unelected_drops_selected_text() -> None:
  draft = "下書き。"
  samples = _samples("id-00")
  remaining = unelected_rows(draft, samples, "選抜文id-00。")
  assert all("選抜文" not in str(r["generated"]) for r in remaining)
  assert len(remaining) == 7


def test_build_pairs_ten_items_three_pairs_excludes_selected() -> None:
  items = [_item(i) for i in range(12)]
  samples_by_id = {it["id"]: _samples(it["id"]) for it in items}
  pairs, stats = build_pairs(
    items,
    samples_by_id,
    n_items=10,
    n_unselected=2,
    seed=42,
    primary_model=None,
  )
  assert stats["n_selected_items"] == 10
  assert len(pairs) == 30
  assert {p["compare_type"] for p in pairs} == {COMPARE_TYPE}
  by_item: dict[str, list[dict]] = {}
  for p in pairs:
    by_item.setdefault(p["item_id"], []).append(p)
    sources = {p["a_source"], p["b_source"]}
    assert "gold" not in sources
    assert "adapter_selected" not in sources
    assert "adapter_unselected:0" not in sources
    assert "選抜文" not in p["a_text"]
    assert "選抜文" not in p["b_text"]
    assert "人間" not in p["a_text"]
    assert "人間" not in p["b_text"]
    assert p["context_draft"].startswith("下書き")
    assert p["question"]
  assert len(by_item) == 10
  for rs in by_item.values():
    assert len(rs) == 3
    n_with_draft = sum(
      1 for p in rs if SRC_DRAFT in {p["a_source"], p["b_source"]}
    )
    assert n_with_draft == 2
  assert [p["order"] for p in sorted(pairs, key=lambda r: r["order"])] == list(range(30))


def test_analyze_detects_cycle_and_transitive() -> None:
  pairs = [
    {
      "pair_id": "i1::p1",
      "item_id": "i1",
      "a_source": "draft",
      "b_source": "adapter_unselected:1",
    },
    {
      "pair_id": "i1::p2",
      "item_id": "i1",
      "a_source": "adapter_unselected:1",
      "b_source": "adapter_unselected:2",
    },
    {
      "pair_id": "i1::p3",
      "item_id": "i1",
      "a_source": "adapter_unselected:2",
      "b_source": "draft",
    },
  ]
  cyclic = [
    {"pair_id": "i1::p1", "choice": "a"},
    {"pair_id": "i1::p2", "choice": "a"},
    {"pair_id": "i1::p3", "choice": "a"},
  ]
  report = analyze(pairs, cyclic)
  assert report["n_cyclic"] == 1
  assert report["n_transitive"] == 0

  transitive = [
    {"pair_id": "i1::p1", "choice": "b"},
    {"pair_id": "i1::p2", "choice": "a"},
    {"pair_id": "i1::p3", "choice": "b"},
  ]
  report2 = analyze(pairs, transitive)
  assert report2["n_cyclic"] == 0
  assert report2["n_transitive"] == 1


def test_classify_tie_is_not_cycle() -> None:
  rows = [
    {
      "item_id": "i1",
      "a_source": "draft",
      "b_source": "u1",
      "choice": "tie",
    },
    {
      "item_id": "i1",
      "a_source": "u1",
      "b_source": "u2",
      "choice": "a",
    },
    {
      "item_id": "i1",
      "a_source": "u2",
      "b_source": "draft",
      "choice": "b",
    },
  ]
  out = classify_item(rows)
  assert out["cyclic"] is False
  assert out["transitive"] is True


def test_classify_incomparable_is_not_tie_or_transitive() -> None:
  rows = [
    {
      "item_id": "i1",
      "a_source": "draft",
      "b_source": "u1",
      "choice": "incomparable",
      "incomparable_reason": "both_worse",
    },
    {
      "item_id": "i1",
      "a_source": "u1",
      "b_source": "u2",
      "choice": "a",
    },
    {
      "item_id": "i1",
      "a_source": "u2",
      "b_source": "draft",
      "choice": "b",
    },
  ]
  out = classify_item(rows)
  assert out["incomparable"] is True
  assert out["n_incomparable"] == 1
  assert out["n_tie"] == 0
  assert out["cyclic"] is False
  assert out["transitive"] is False


def test_make_scalar_transitivity_judge_uses_new_files() -> None:
  result = subprocess.run(
    ["make", "-n", "scalar-transitivity-judge"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "pairs_scalar_transitivity.jsonl" in out
  assert "judgments_scalar_transitivity.jsonl" in out
  assert "8325" in out
  assert "0.0.0.0" in out
  assert "pairs.jsonl" not in out.replace("pairs_scalar_transitivity.jsonl", "")
  assert "自分の推敲に近いか" in out
  assert "比較できない" in out
