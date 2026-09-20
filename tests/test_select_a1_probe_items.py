from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from select_a1_probe_items import load_train_ids, pick_ids, row_id
from stage_japanese_tech_writing import strip_frontmatter


def test_pick_ids_reproducible() -> None:
  ids = [f"id-{i:03d}" for i in range(50)]
  a = pick_ids(ids, n=10, seed=42)
  b = pick_ids(ids, n=10, seed=42)
  c = pick_ids(ids, n=10, seed=0)
  assert a == b
  assert len(a) == 10
  assert a != c


def test_pick_ids_takes_all_when_n_exceeds() -> None:
  ids = ["b", "a"]
  out = pick_ids(ids, n=10, seed=42)
  assert sorted(out) == ["a", "b"]


def test_load_train_ids(tmp_path: Path) -> None:
  p = tmp_path / "train.jsonl"
  rows = [
    {"messages": [], "meta": {"id": "keep-hunk_nopara:x-1"}},
    {"messages": [], "meta": {"id": "keep-hunk_nopara:x-2"}},
  ]
  p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
  assert load_train_ids(p) == [
    "keep-hunk_nopara:x-1",
    "keep-hunk_nopara:x-2",
  ]
  assert row_id(rows[0]) == "keep-hunk_nopara:x-1"


def test_strip_frontmatter() -> None:
  raw = "---\nname: x\n---\n\n# 本文\n"
  assert strip_frontmatter(raw) == "# 本文\n"
