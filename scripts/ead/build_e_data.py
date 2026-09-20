#!/usr/bin/env python3
"""E1 取り込みと E1/E2/E3 の統合 jsonl・manifest 生成。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import load_jsonl, repo_root, write_jsonl
from section_middle_utils import CHOICE_OK

from ead.e_data import (
  DEGRADED_CHOICE,
  DEFAULT_E_ADAPTER,
  DEFAULT_E_GENERATOR,
  adapter_greedy_path,
  e_data_dir,
  e_row_path,
  load_blind_eval_items_by_id,
  load_e2_e3_draft_ids,
  load_judgments_by_id,
  load_keep_section_by_id,
  merge_e_rows,
  revisions_path,
  write_manifest,
)
from ead.e_prompt import PROMPT_TAG


def build_e1_rows(root: Path) -> list[dict[str, Any]]:
  keep = load_keep_section_by_id(root)
  judgments = load_judgments_by_id(root)
  revisions = {str(r["id"]): r for r in load_jsonl(revisions_path(root))}
  rows: list[dict[str, Any]] = []
  for item_id, item in sorted(keep.items()):
    choice = judgments.get(item_id, "")
    if choice == DEGRADED_CHOICE:
      continue
    if choice != CHOICE_OK:
      raise SystemExit(f"unexpected judgment for {item_id}: {choice!r}")
    rev = revisions.get(item_id)
    if rev is None:
      raise SystemExit(f"missing revision for {item_id}")
    generated = str(rev.get("text") or "").strip()
    if not generated:
      raise SystemExit(f"empty revision text for {item_id}")
    draft = str(item.get("source_text") or "")
    if not draft.strip():
      raise SystemExit(f"empty draft for {item_id}")
    rows.append(
      {
        "item_id": item_id,
        "project_id": str(item.get("project_id") or rev.get("project_id") or ""),
        "draft": draft,
        "generated": generated,
        "e_kind": "E1",
        "prompt_tag": str(rev.get("prompt_tag") or PROMPT_TAG),
        "generator": str(rev.get("generator") or ""),
        "generation": "api",
        "adapter": None,
      }
    )
  return rows


def build_e2_from_c(root: Path, *, adapter: str, adapter_greedy: Path) -> list[dict[str, Any]]:
  items = load_blind_eval_items_by_id(root)
  draft_ids = load_e2_e3_draft_ids(root)
  by_id = {str(r["id"]): r for r in load_jsonl(adapter_greedy)}
  rows: list[dict[str, Any]] = []
  for item_id in draft_ids:
    gen = by_id.get(item_id)
    if gen is None:
      raise SystemExit(f"missing adapter_greedy row for {item_id}")
    if str(gen.get("mode") or "") != "adapter":
      raise SystemExit(f"adapter_greedy {item_id}: mode={gen.get('mode')!r}")
    if str(gen.get("generation") or "") != "greedy":
      raise SystemExit(f"adapter_greedy {item_id}: generation={gen.get('generation')!r}")
    item = items[item_id]
    draft = str(item.get("draft") or "")
    gen_draft = str(gen.get("draft") or "")
    if draft != gen_draft:
      raise SystemExit(f"draft mismatch for {item_id} (blind_eval vs adapter_greedy)")
    generated = str(gen.get("generated") or "").strip()
    if not generated:
      raise SystemExit(f"empty generated for {item_id}")
    rows.append(
      {
        "item_id": item_id,
        "project_id": str(item.get("project_id") or gen.get("project_id") or ""),
        "draft": draft,
        "generated": generated,
        "e_kind": "E2",
        "prompt_tag": PROMPT_TAG,
        "generator": DEFAULT_E_GENERATOR,
        "generation": "greedy",
        "adapter": adapter,
        "input_tokens": gen.get("input_tokens"),
        "enable_thinking": False,
        "source": "c_adapter_greedy",
      }
    )
  return rows


def import_qwen_rows(path: Path, *, e_kind: str) -> list[dict[str, Any]]:
  rows = load_jsonl(path)
  out: list[dict[str, Any]] = []
  for row in rows:
    out.append(
      {
        "item_id": str(row.get("item_id") or row.get("id") or ""),
        "project_id": str(row.get("project_id") or ""),
        "draft": str(row.get("draft") or ""),
        "generated": str(row.get("generated") or ""),
        "e_kind": e_kind,
        "prompt_tag": str(row.get("prompt_tag") or PROMPT_TAG),
        "generator": str(row.get("generator") or ""),
        "generation": str(row.get("generation") or "greedy"),
        "adapter": row.get("adapter"),
        "input_tokens": row.get("input_tokens"),
        "max_new_tokens": row.get("max_new_tokens"),
        "enable_thinking": row.get("enable_thinking", False),
      }
    )
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=None)
  parser.add_argument("--data-dir", type=Path, default=None)
  parser.add_argument("--e1-only", action="store_true")
  parser.add_argument("--build-e2-from-c", action="store_true")
  parser.add_argument("--adapter-greedy", type=Path, default=None)
  parser.add_argument("--adapter", default=DEFAULT_E_ADAPTER)
  parser.add_argument("--import-e2", type=Path, default=None)
  parser.add_argument("--import-e3", type=Path, default=None)
  args = parser.parse_args()

  root = (args.root or repo_root()).resolve()
  data_dir = (args.data_dir or e_data_dir(root)).resolve()
  data_dir.mkdir(parents=True, exist_ok=True)

  e1_rows = build_e1_rows(root)
  write_jsonl(e_row_path(data_dir, "E1"), e1_rows)
  print(f"E1: wrote {len(e1_rows)} rows -> {e_row_path(data_dir, 'E1')}")

  if args.build_e2_from_c:
    ag_path = (args.adapter_greedy or adapter_greedy_path(root)).resolve()
    e2_rows = build_e2_from_c(root, adapter=str(args.adapter), adapter_greedy=ag_path)
    write_jsonl(e_row_path(data_dir, "E2"), e2_rows)
    print(f"E2: built {len(e2_rows)} rows from {ag_path.relative_to(root)}")

  if args.import_e2:
    e2_rows = import_qwen_rows(args.import_e2.resolve(), e_kind="E2")
    write_jsonl(e_row_path(data_dir, "E2"), e2_rows)
    print(f"E2: imported {len(e2_rows)} rows from {args.import_e2}")

  if args.import_e3:
    e3_rows = import_qwen_rows(args.import_e3.resolve(), e_kind="E3")
    write_jsonl(e_row_path(data_dir, "E3"), e3_rows)
    print(f"E3: imported {len(e3_rows)} rows from {args.import_e3}")

  manifest = write_manifest(data_dir, root)

  if args.e1_only:
    return

  merged = merge_e_rows(data_dir, root)
  manifest = write_manifest(data_dir, root)
  print(f"merged -> {merged} ({sum(manifest['counts'].values())} rows)")
  print(f"manifest -> {data_dir / 'manifest.json'}")


if __name__ == "__main__":
  main()
