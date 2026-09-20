#!/usr/bin/env python3
"""評価データ E の共通読み込み・スキーマ。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))

from ead.common import load_jsonl, repo_root, write_json, write_jsonl
from ead.e_prompt import PROMPT_TAG, build_e_user_content
from section_middle_utils import CHOICE_DEGRADED

E_KINDS = ("E1", "E2", "E3")
E1_EXPECTED = 378
E2_E3_SCOPE = "c_blind60"
E2_E3_EXPECTED = 60
DEFAULT_E_ADAPTER = "outputs/Qwen__Qwen3-8B-pairsplit-v2/adapter"
DEFAULT_E_GENERATOR = "Qwen/Qwen3-8B"
DEGRADED_CHOICE = CHOICE_DEGRADED


def e_data_dir(root: Path | None = None) -> Path:
  return (root or repo_root()) / "outputs" / "ead" / "data"


def keep_section_path(root: Path | None = None) -> Path:
  return (root or repo_root()) / "data" / "revision_corpus" / "keep_section.jsonl"


def heldout_section_path(root: Path | None = None) -> Path:
  return (root or repo_root()) / "data" / "edit_sft_section" / "heldout.jsonl"


def blind_eval_items_path(root: Path | None = None) -> Path:
  return (root or repo_root()) / "data" / "blind_eval" / "items.jsonl"


def adapter_greedy_path(root: Path | None = None) -> Path:
  return (root or repo_root()) / "outputs" / "edit-sft-eval-v3" / "adapter_greedy.jsonl"


def edit_sft_section_paths(root: Path | None = None) -> list[Path]:
  base = (root or repo_root()) / "data" / "edit_sft_section"
  return [base / "train.jsonl", base / "heldout.jsonl"]


def revisions_path(root: Path | None = None) -> Path:
  return (root or repo_root()) / "data" / "section_middle" / "revisions.jsonl"


def judgments_path(root: Path | None = None) -> Path:
  return (root or repo_root()) / "data" / "section_middle" / "judgments.jsonl"


def manifest_path(data_dir: Path | None = None, root: Path | None = None) -> Path:
  return (data_dir or e_data_dir(root)) / "manifest.json"


def load_e2_e3_draft_ids(root: Path | None = None) -> list[str]:
  """E2/E3 共通の下書き id（評価用データ C の 60 件）。"""
  ids = sorted(load_blind_eval_items_by_id(root))
  if len(ids) != E2_E3_EXPECTED:
    raise ValueError(f"expected {E2_E3_EXPECTED} blind_eval items, got {len(ids)}")
  return ids


def load_blind_eval_items_by_id(root: Path | None = None) -> dict[str, dict]:
  rows = load_jsonl(blind_eval_items_path(root))
  out = {str(r["id"]): r for r in rows}
  if len(out) != len(rows):
    raise ValueError(f"duplicate id in {blind_eval_items_path(root)}")
  return out


def load_manifest(data_dir: Path, root: Path | None = None) -> dict[str, Any]:
  path = manifest_path(data_dir, root)
  if not path.is_file():
    return expected_manifest(root)
  import json

  return json.loads(path.read_text(encoding="utf-8"))


def expected_manifest(root: Path | None = None) -> dict[str, Any]:
  root = root or repo_root()
  e2_e3_ids = load_e2_e3_draft_ids(root)
  n = len(e2_e3_ids)
  return {
    "schema": "ead_e_v1",
    "prompt_tag": PROMPT_TAG,
    "e2_e3_draft_scope": E2_E3_SCOPE,
    "e2_e3_draft_source": str(blind_eval_items_path(root).relative_to(root)),
    "e2_e3_item_ids": e2_e3_ids,
    "expected": {"E1": E1_EXPECTED, "E2": n, "E3": n},
  }


def load_keep_section_by_id(root: Path | None = None) -> dict[str, dict]:
  rows = load_jsonl(keep_section_path(root))
  return {str(r["id"]): r for r in rows}


def load_judgments_by_id(root: Path | None = None) -> dict[str, str]:
  out: dict[str, str] = {}
  for row in load_jsonl(judgments_path(root)):
    item_id = str(row.get("item_id") or "")
    if item_id:
      out[item_id] = str(row.get("choice") or "")
  return out


def load_a2_section_samples(
  root: Path | None = None,
  *,
  item_ids: set[str] | None = None,
  paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
  samples: list[dict[str, Any]] = []
  for path in paths or edit_sft_section_paths(root):
    for row in load_jsonl(path):
      user = str(row["messages"][0]["content"])
      if "\n\n" not in user:
        raise ValueError(f"bad user content (no draft separator): {path}")
      draft = user.split("\n\n", 1)[1]
      meta = row.get("meta") or {}
      item_id = str(meta.get("id") or "")
      if not item_id:
        raise ValueError(f"missing meta.id in {path}")
      if item_ids is not None and item_id not in item_ids:
        continue
      expected = build_e_user_content(draft)
      if user != expected:
        raise ValueError(f"user content != build_e_user_content for {item_id}")
      samples.append(
        {
          "item_id": item_id,
          "project_id": str(meta.get("project_id") or ""),
          "draft": draft,
          "user_content": user,
        }
      )
  samples.sort(key=lambda r: r["item_id"])
  return samples


def load_e2_e3_samples(root: Path | None = None) -> list[dict[str, Any]]:
  items = load_blind_eval_items_by_id(root)
  samples: list[dict[str, Any]] = []
  for item_id in load_e2_e3_draft_ids(root):
    item = items[item_id]
    draft = str(item.get("draft") or "")
    if not draft.strip():
      raise ValueError(f"empty draft for {item_id}")
    user_content = build_e_user_content(draft)
    samples.append(
      {
        "item_id": item_id,
        "project_id": str(item.get("project_id") or ""),
        "draft": draft,
        "user_content": user_content,
      }
    )
  return samples


def e_row_path(data_dir: Path, e_kind: str) -> Path:
  return data_dir / f"{e_kind.lower()}.jsonl"


def merge_e_rows(data_dir: Path, root: Path | None = None) -> Path:
  merged: list[dict[str, Any]] = []
  for kind in E_KINDS:
    path = e_row_path(data_dir, kind)
    if path.is_file():
      merged.extend(load_jsonl(path))
  merged.sort(key=lambda r: (str(r.get("e_kind") or ""), str(r.get("item_id") or "")))
  out = data_dir / "e.jsonl"
  write_jsonl(out, merged)
  return out


def write_manifest(data_dir: Path, root: Path | None = None, *, extra: dict | None = None) -> dict[str, Any]:
  root = root or repo_root()
  manifest = expected_manifest(root)
  counts = {}
  for kind in E_KINDS:
    path = e_row_path(data_dir, kind)
    counts[kind] = len(load_jsonl(path)) if path.is_file() else 0
  manifest["counts"] = counts
  manifest["sources"] = {
    "keep_section": str(keep_section_path(root).relative_to(root)),
    "revisions": str(revisions_path(root).relative_to(root)),
    "judgments": str(judgments_path(root).relative_to(root)),
    "blind_eval_items": str(blind_eval_items_path(root).relative_to(root)),
    "adapter_greedy": str(adapter_greedy_path(root).relative_to(root)),
  }
  if extra:
    manifest.update(extra)
  write_json(manifest_path(data_dir, root), manifest)
  return manifest
