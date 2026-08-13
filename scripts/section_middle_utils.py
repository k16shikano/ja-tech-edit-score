#!/usr/bin/env python3
"""節ペアに Composer 推敲を足した三つ組み用の共通処理。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from export_edit_sft import INSTRUCTION

PROMPT_TAG = "edit-sft-instruction"
SCHEMA = "section_triple_v1"
CHOICE_OK = "ok"
CHOICE_DEGRADED = "degraded"
FENCE_RE = re.compile(r"^```(?:\w+)?\n(.*)\n```\s*$", re.DOTALL)


def load_jsonl(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      rows.append(json.loads(line))
  return rows


def append_jsonl(path: Path, row: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def corpus_split(rec: dict) -> str:
  split = str((rec.get("meta") or {}).get("split") or "").strip()
  if split == "heldout":
    return "valid"
  if split == "train":
    return "train"
  raise ValueError(f"missing meta.split in {rec.get('id')}")


def build_revision_prompt(draft: str) -> str:
  return f"{INSTRUCTION}\n\n{draft}"


def strip_code_fence(text: str) -> str:
  cleaned = (text or "").strip()
  match = FENCE_RE.match(cleaned)
  if match:
    return match.group(1).strip()
  return cleaned


def revisions_by_id(rows: list[dict]) -> dict[str, dict]:
  out: dict[str, dict] = {}
  for row in rows:
    rid = str(row.get("id") or "")
    if rid:
      out[rid] = row
  return out


def judgments_by_id(rows: list[dict]) -> dict[str, dict]:
  out: dict[str, dict] = {}
  for row in rows:
    rid = str(row.get("item_id") or row.get("id") or "")
    if rid:
      out[rid] = row
  return out


def _pref_row(
  *,
  item: dict,
  pair_kind: str,
  candidate_a: str,
  candidate_b: str,
) -> dict:
  base_id = str(item.get("id") or "")
  split = corpus_split(item)
  return {
    "id": f"{base_id}__{pair_kind}",
    "source_text": str(item.get("source_text") or ""),
    "candidate_a": candidate_a,
    "candidate_b": candidate_b,
    "label": 1,
    "preference": "a",
    "schema": SCHEMA,
    "pair_kind": pair_kind,
    "meta": {
      "item_id": base_id,
      "unit": str(item.get("unit") or "section"),
      "project_id": str(item.get("project_id") or ""),
      "corpus_split": str((item.get("meta") or {}).get("split") or ""),
      "split": split,
      "prompt_tag": PROMPT_TAG,
    },
  }


def build_triple_pref_rows(item: dict, generated_text: str) -> list[dict]:
  draft = str(item.get("source_text") or "").strip()
  gold = str(item.get("edited_text") or "").strip()
  gen = strip_code_fence(generated_text).strip()
  if not draft or not gold or not gen:
    return []
  if gen == draft or gen == gold:
    return []
  return [
    _pref_row(item=item, pair_kind="gold_vs_draft", candidate_a=gold, candidate_b=draft),
    _pref_row(item=item, pair_kind="gold_vs_gen", candidate_a=gold, candidate_b=gen),
    _pref_row(item=item, pair_kind="gen_vs_draft", candidate_a=gen, candidate_b=draft),
  ]


def build_pref_from_judgments(
  items: list[dict],
  revisions: dict[str, dict],
  judgments: dict[str, dict],
) -> tuple[list[dict], list[dict], dict]:
  train: list[dict] = []
  valid: list[dict] = []
  stats = {
    "items": len(items),
    "ok": 0,
    "degraded": 0,
    "skipped_no_revision": 0,
    "skipped_no_judgment": 0,
    "skipped_empty_triple": 0,
    "train_items": 0,
    "valid_items": 0,
    "train_rows": 0,
    "valid_rows": 0,
  }
  for item in items:
    item_id = str(item.get("id") or "")
    rev = revisions.get(item_id)
    if not rev or not str(rev.get("text") or "").strip():
      stats["skipped_no_revision"] += 1
      continue
    jud = judgments.get(item_id)
    if not jud:
      stats["skipped_no_judgment"] += 1
      continue
    choice = str(jud.get("choice") or "")
    if choice == CHOICE_DEGRADED:
      stats["degraded"] += 1
      continue
    if choice != CHOICE_OK:
      stats["skipped_no_judgment"] += 1
      continue
    rows = build_triple_pref_rows(item, str(rev.get("text") or ""))
    if not rows:
      stats["skipped_empty_triple"] += 1
      continue
    stats["ok"] += 1
    split = corpus_split(item)
    if split == "valid":
      valid.extend(rows)
      stats["valid_items"] += 1
      stats["valid_rows"] += len(rows)
    else:
      train.extend(rows)
      stats["train_items"] += 1
      stats["train_rows"] += len(rows)
  return train, valid, stats
