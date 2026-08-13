from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from export_edit_sft import INSTRUCTION
from section_middle_utils import (
  CHOICE_DEGRADED,
  CHOICE_OK,
  build_pref_from_judgments,
  build_revision_prompt,
  build_triple_pref_rows,
  strip_code_fence,
)


def test_prompt_is_eval_instruction() -> None:
  draft = "見出しです。\n\n本文。"
  prompt = build_revision_prompt(draft)
  assert prompt.startswith(INSTRUCTION)
  assert prompt.endswith(draft)
  assert "中間" not in prompt
  assert "ほどほど" not in prompt


def test_strip_code_fence() -> None:
  assert strip_code_fence("```\n推敲。\n```") == "推敲。"
  assert strip_code_fence("推敲。") == "推敲。"


def test_triple_has_three_same_source_pairs() -> None:
  item = {
    "id": "s1",
    "source_text": "下書き。",
    "edited_text": "人間の推敲。",
    "unit": "section",
    "project_id": "p",
    "meta": {"split": "train"},
  }
  rows = build_triple_pref_rows(item, "生成。")
  assert [r["pair_kind"] for r in rows] == [
    "gold_vs_draft",
    "gold_vs_gen",
    "gen_vs_draft",
  ]
  assert all(r["source_text"] == "下書き。" for r in rows)
  assert all(r["label"] == 1 for r in rows)
  by_kind = {r["pair_kind"]: r for r in rows}
  assert by_kind["gold_vs_draft"]["candidate_a"] == "人間の推敲。"
  assert by_kind["gold_vs_draft"]["candidate_b"] == "下書き。"
  assert by_kind["gold_vs_gen"]["candidate_b"] == "生成。"
  assert by_kind["gen_vs_draft"]["candidate_a"] == "生成。"


def test_identical_gen_is_not_a_middle() -> None:
  item = {
    "id": "s1",
    "source_text": "下書き。",
    "edited_text": "人間の推敲。",
    "meta": {"split": "train"},
  }
  assert build_triple_pref_rows(item, "下書き。") == []


def test_degraded_is_excluded_ok_goes_to_split() -> None:
  items = [
    {
      "id": "train-ok",
      "source_text": "下書き甲。",
      "edited_text": "人間甲。",
      "unit": "section",
      "meta": {"split": "train"},
    },
    {
      "id": "held-ok",
      "source_text": "下書き乙。",
      "edited_text": "人間乙。",
      "unit": "section",
      "meta": {"split": "heldout"},
    },
    {
      "id": "train-bad",
      "source_text": "下書き丙。",
      "edited_text": "人間丙。",
      "unit": "section",
      "meta": {"split": "train"},
    },
  ]
  revisions = {
    "train-ok": {"id": "train-ok", "text": "生成甲。"},
    "held-ok": {"id": "held-ok", "text": "生成乙。"},
    "train-bad": {"id": "train-bad", "text": "生成丙。"},
  }
  judgments = {
    "train-ok": {"item_id": "train-ok", "choice": CHOICE_OK},
    "held-ok": {"item_id": "held-ok", "choice": CHOICE_OK},
    "train-bad": {"item_id": "train-bad", "choice": CHOICE_DEGRADED},
  }
  train, valid, stats = build_pref_from_judgments(items, revisions, judgments)
  assert stats["ok"] == 2
  assert stats["degraded"] == 1
  assert stats["train_items"] == 1
  assert stats["valid_items"] == 1
  assert stats["train_rows"] == 3
  assert stats["valid_rows"] == 3
  assert all(r["meta"]["item_id"] == "train-ok" for r in train)
  assert all(r["meta"]["item_id"] == "held-ok" for r in valid)


def test_degrade_server_next_and_judge(tmp_path: Path) -> None:
  from fastapi.testclient import TestClient

  from middle_degrade_server import create_app

  items = tmp_path / "items.jsonl"
  revs = tmp_path / "revs.jsonl"
  jud = tmp_path / "jud.jsonl"
  items.write_text(
    '{"id":"x","source_text":"下書き。","edited_text":"推敲。","meta":{"split":"train"}}\n',
    encoding="utf-8",
  )
  revs.write_text('{"id":"x","text":"生成。"}\n', encoding="utf-8")
  jud.write_text("", encoding="utf-8")
  client = TestClient(
    create_app(items_path=items, revisions_path=revs, judgments_path=jud)
  )
  assert client.get("/api/status").json()["remaining"] == 1
  nxt = client.get("/api/next").json()
  assert nxt["item_id"] == "x"
  assert nxt["generated_text"] == "生成。"
  posted = client.post("/api/judge", json={"item_id": "x", "choice": "ok"})
  assert posted.json()["ok"] is True
  assert client.get("/api/next").json()["done"] is True
