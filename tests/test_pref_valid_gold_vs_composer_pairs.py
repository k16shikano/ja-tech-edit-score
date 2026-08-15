from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from build_pref_valid_gold_vs_composer_pairs import COMPARE_TYPE, build_pairs
from setwise_triple_utils import SetwiseTriple


def _triples(n: int) -> list[SetwiseTriple]:
  out: list[SetwiseTriple] = []
  for i in range(n):
    draft = f"下書き{i}。"
    human = f"人間{i}。"
    composer = f"生成{i}。"
    out.append(
      SetwiseTriple(
        item_id=f"id-{i:02d}",
        source_text=draft,
        draft=draft,
        human=human,
        composer=composer,
        meta={},
      )
    )
  return out


def test_build_pairs_one_per_item_hides_roles() -> None:
  triples = _triples(10)
  pairs = build_pairs(triples, seed=42)
  assert len(pairs) == 10
  assert {p["item_id"] for p in pairs} == {t.item_id for t in triples}
  assert [p["order"] for p in pairs] == list(range(10))
  n_swap = 0
  by_id = {t.item_id: t for t in triples}
  for p in pairs:
    assert p["compare_type"] == COMPARE_TYPE
    assert p["context_draft"] == by_id[p["item_id"]].draft
    assert {p["a_source"], p["b_source"]} == {"gold", "composer"}
    gold = by_id[p["item_id"]].human
    composer = by_id[p["item_id"]].composer
    texts = {p["a_source"]: p["a_text"], p["b_source"]: p["b_text"]}
    assert texts["gold"] == gold
    assert texts["composer"] == composer
    if p["swapped"]:
      n_swap += 1
      assert p["a_source"] == "composer"
    else:
      assert p["a_source"] == "gold"
  assert 1 <= n_swap <= 9


def test_protocol_points_at_pref_valid_not_adapter() -> None:
  protocol = json.loads(
    (ROOT / "data" / "blind_eval" / "valid50_gold_vs_composer_protocol.json").read_text(
      encoding="utf-8"
    )
  )
  assert protocol["source"] == "data/section_middle/pref_valid.jsonl"
  assert protocol["n_items"] == 50
  assert "adapter" not in json.dumps(protocol)


def test_make_pref_valid_blind_judge_uses_new_files_not_c() -> None:
  result = subprocess.run(
    ["make", "-n", "pref-valid-blind-judge"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "pairs_pref_valid_gold_vs_composer.jsonl" in out
  assert "judgments_pref_valid_gold_vs_composer.jsonl" in out
  assert "blind_judge_server.py" in out
  assert "--host" in out
  assert "0.0.0.0" in out
  assert "data/blind_eval/pairs.jsonl" not in out
  assert "adapter_selected" not in out
