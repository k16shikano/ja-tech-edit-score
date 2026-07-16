#!/usr/bin/env python3
"""activation steering / 編集モデル共用のユーティリティ。"""
from __future__ import annotations

import json
import re
from pathlib import Path

REF_MARKER = "\n\n[参照]\n"


def strip_reference_block(text: str) -> str:
  if REF_MARKER in text:
    return text.split(REF_MARKER, 1)[0].rstrip()
  # input 先頭直後にだけ参照が付くケースも落とす
  return re.sub(r"\n\n\[参照\]\n[\s\S]*\Z", "", text).rstrip()


def model_slug(model_id: str) -> str:
  return model_id.strip().replace("/", "__").replace(" ", "_")


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_revision_pairs_from_dpo(rows: list[dict]) -> list[dict]:
  """dpo_curated 行から draft/revised ペアを作る。

  rejected_mode=source を前提に、rejected=下書き、chosen=推敲後とする。
  """
  out: list[dict] = []
  for rec in rows:
    meta = rec.get("meta") or {}
    draft = strip_reference_block(str(rec.get("rejected") or ""))
    revised = strip_reference_block(str(rec.get("chosen") or ""))
    if not draft or not revised:
      continue
    if draft == revised:
      continue
    out.append(
      {
        "id": rec.get("id", ""),
        "project_id": str(meta.get("project_id") or "(unknown)"),
        "draft": draft,
        "revised": revised,
      }
    )
  return out


def load_revision_pairs(path: Path) -> list[dict]:
  rows = load_jsonl(path)
  if not rows:
    return []
  # すでに export 済み形式
  if "draft" in rows[0] and "revised" in rows[0]:
    return rows
  return iter_revision_pairs_from_dpo(rows)
