from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from build_b_rejudge_pairs import (
  COMPARE_BY_SOURCES,
  SRC_COMPOSER,
  SRC_DRAFT,
  SRC_HUMAN,
  build_pairs,
  collect_train_triples,
)
from section_middle_utils import CHOICE_DEGRADED, CHOICE_OK, write_jsonl
from setwise_triple_utils import SetwiseTriple


def _triple(i: int) -> SetwiseTriple:
  draft = f"下書き{i}。"
  human = f"人間{i}。"
  composer = f"生成{i}。"
  return SetwiseTriple(
    item_id=f"id-{i:02d}",
    source_text=draft,
    draft=draft,
    human=human,
    composer=composer,
    meta={},
  )


def test_build_pairs_three_per_item_all_source_combos() -> None:
  triples = [_triple(i) for i in range(5)]
  pairs = build_pairs(triples, seed=42)
  assert len(pairs) == 15
  by_item: dict[str, list[dict]] = {}
  for p in pairs:
    by_item.setdefault(p["item_id"], []).append(p)
    assert p["a_source"] != p["b_source"]
    assert frozenset({p["a_source"], p["b_source"]}) in COMPARE_BY_SOURCES
    assert p["question"]
  assert len(by_item) == 5
  for rs in by_item.values():
    assert len(rs) == 3
    combos = {frozenset({r["a_source"], r["b_source"]}) for r in rs}
    assert combos == {
      frozenset({SRC_DRAFT, SRC_HUMAN}),
      frozenset({SRC_DRAFT, SRC_COMPOSER}),
      frozenset({SRC_HUMAN, SRC_COMPOSER}),
    }
  n_swap = sum(1 for p in pairs if p["swapped"])
  assert 1 <= n_swap <= 14
  human_pairs = [
    p
    for p in pairs
    if p["compare_type"] in ("gold_vs_draft", "gold_vs_composer")
  ]
  assert human_pairs
  for p in human_pairs:
    texts = {p["a_source"]: p["a_text"], p["b_source"]: p["b_text"]}
    assert "人間" in texts[SRC_HUMAN]


def test_collect_train_includes_degraded_excludes_valid(tmp_path: Path) -> None:
  items = [
    {
      "id": "train-ok",
      "source_text": "下書き甲。",
      "edited_text": "人間甲。",
      "meta": {"split": "train"},
    },
    {
      "id": "held-ok",
      "source_text": "下書き乙。",
      "edited_text": "人間乙。",
      "meta": {"split": "heldout"},
    },
    {
      "id": "train-bad",
      "source_text": "下書き丙。",
      "edited_text": "人間丙。",
      "meta": {"split": "train"},
    },
  ]
  items_path = tmp_path / "items.jsonl"
  write_jsonl(items_path, items)

  revisions_path = tmp_path / "revisions.jsonl"
  write_jsonl(
    revisions_path,
    [
      {"id": "train-ok", "text": "生成甲。"},
      {"id": "held-ok", "text": "生成乙。"},
      {"id": "train-bad", "text": "生成丙。"},
    ],
  )

  judgments_path = tmp_path / "judgments.jsonl"
  write_jsonl(
    judgments_path,
    [
      {"item_id": "train-ok", "choice": CHOICE_OK},
      {"item_id": "held-ok", "choice": CHOICE_OK},
      {"item_id": "train-bad", "choice": CHOICE_DEGRADED},
    ],
  )

  train_rows = []
  for item in items:
    if item["id"] == "train-ok":
      from section_middle_utils import build_triple_pref_rows

      train_rows.extend(build_triple_pref_rows(item, "生成甲。"))
  train_path = tmp_path / "pref_train.jsonl"
  write_jsonl(train_path, train_rows)

  valid_rows = []
  for item in items:
    if item["id"] == "held-ok":
      from section_middle_utils import build_triple_pref_rows

      valid_rows.extend(build_triple_pref_rows(item, "生成乙。"))
  valid_path = tmp_path / "pref_valid.jsonl"
  write_jsonl(valid_path, valid_rows)

  triples, stats = collect_train_triples(
    train_file=train_path,
    valid_file=valid_path,
    items_file=items_path,
    revisions_file=revisions_path,
    judgments_file=judgments_path,
  )
  ids = {t.item_id for t in triples}
  assert ids == {"train-ok", "train-bad"}
  assert "held-ok" not in ids
  assert stats["n_degraded_added"] == 1
  assert stats["n_overlap_with_valid"] == 0


def test_make_b_rejudge_targets() -> None:
  pairs = subprocess.run(
    ["make", "-n", "b-rejudge-pairs"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  assert "pairs_rejudge.jsonl" in pairs.stdout
  assert "build_b_rejudge_pairs.py" in pairs.stdout

  judge = subprocess.run(
    ["make", "-n", "b-rejudge-judge"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = judge.stdout
  assert "pairs_rejudge.jsonl" in out
  assert "judgments_rejudge.jsonl" in out
  assert "8326" in out
  assert "0.0.0.0" in out
  assert "自分の推敲に近いか" in out
  assert "比較できない" in out

  analyze = subprocess.run(
    ["make", "-n", "analyze-b-rejudge"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  assert "analyze_b_rejudge.py" in analyze.stdout


def test_real_pref_train_counts(tmp_path: Path) -> None:
  train_path = ROOT / "data/section_middle/pref_train.jsonl"
  valid_path = ROOT / "data/section_middle/pref_valid.jsonl"
  if not train_path.is_file() or not valid_path.is_file():
    return
  triples, stats = collect_train_triples(
    train_file=train_path,
    valid_file=valid_path,
    items_file=ROOT / "data/revision_corpus/keep_section.jsonl",
    revisions_file=ROOT / "data/section_middle/revisions.jsonl",
    judgments_file=ROOT / "data/section_middle/judgments.jsonl",
  )
  valid_ids = {
    json.loads(line)["meta"]["item_id"]
    for line in valid_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  }
  assert len({t.item_id for t in triples}) == stats["n_items"]
  assert len(triples) == stats["n_pref_train_items"] + stats["n_degraded_added"]
  assert not ({t.item_id for t in triples} & valid_ids)
  pairs = build_pairs(triples, seed=42)
  assert len(pairs) == len(triples) * 3
