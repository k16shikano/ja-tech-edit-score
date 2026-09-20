#!/usr/bin/env python3
"""評価データ E（E1/E2/E3）の全体検証。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from check_a1_probe_smoke import looks_like_prompt_leak
from ead.common import load_jsonl, repo_root, write_json
from ead.e_data import (
  DEGRADED_CHOICE,
  E_KINDS,
  e_data_dir,
  e_row_path,
  expected_manifest,
  load_blind_eval_items_by_id,
  load_judgments_by_id,
  load_keep_section_by_id,
  load_manifest,
)
from ead.e_prompt import PROMPT_TAG
from section_middle_utils import CHOICE_OK


@dataclass
class ValidationReport:
  ok: bool = True
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)
  counts: dict[str, int] = field(default_factory=dict)

  def fail(self, msg: str) -> None:
    self.ok = False
    self.errors.append(msg)


def validate_row_fields(row: dict, *, e_kind: str) -> list[str]:
  errs: list[str] = []
  for key in ("item_id", "draft", "generated", "e_kind", "prompt_tag"):
    if not str(row.get(key) or "").strip():
      errs.append(f"{e_kind} {row.get('item_id')}: missing {key}")
  if str(row.get("e_kind") or "") != e_kind:
    errs.append(f"{row.get('item_id')}: e_kind={row.get('e_kind')!r} expected {e_kind}")
  if str(row.get("prompt_tag") or "") != PROMPT_TAG:
    errs.append(f"{row.get('item_id')}: prompt_tag={row.get('prompt_tag')!r}")
  generated = str(row.get("generated") or "")
  if not generated.strip():
    errs.append(f"{row.get('item_id')}: empty generated")
  elif looks_like_prompt_leak(generated):
    errs.append(f"{row.get('item_id')}: prompt leak in generated")
  return errs


def validate_kind(
  path: Path,
  *,
  e_kind: str,
  root: Path,
  report: ValidationReport,
  expected_count: int,
  required_ids: set[str] | None,
) -> dict[str, dict]:
  if not path.is_file():
    report.fail(f"missing {path}")
    return {}

  rows = load_jsonl(path)
  report.counts[e_kind] = len(rows)
  if len(rows) != expected_count:
    report.fail(f"{e_kind}: count {len(rows)} != expected {expected_count}")

  keep = load_keep_section_by_id(root)
  blind_items = load_blind_eval_items_by_id(root)
  judgments = load_judgments_by_id(root)
  by_id: dict[str, dict] = {}
  seen: set[str] = set()

  for row in rows:
    item_id = str(row.get("item_id") or "")
    if item_id in seen:
      report.fail(f"{e_kind}: duplicate item_id {item_id}")
    seen.add(item_id)
    by_id[item_id] = row

    for err in validate_row_fields(row, e_kind=e_kind):
      report.fail(f"{e_kind}: {err}")

    if e_kind == "E1":
      if judgments.get(item_id) == DEGRADED_CHOICE:
        report.fail(f"E1 includes degraded item {item_id}")
      if not str(row.get("generator") or "").strip():
        report.fail(f"E1 {item_id}: missing generator")
    else:
      if required_ids is not None and item_id not in required_ids:
        report.fail(f"{e_kind} {item_id}: not in e2_e3 draft set")
      if str(row.get("generation") or "") != "greedy":
        report.fail(f"{e_kind} {item_id}: generation must be greedy")
      if row.get("enable_thinking"):
        report.fail(f"{e_kind} {item_id}: enable_thinking must be false")
      if e_kind == "E2" and not row.get("adapter"):
        report.fail(f"E2 {item_id}: missing adapter")
      if e_kind == "E3" and row.get("adapter"):
        report.fail(f"E3 {item_id}: adapter must be null")

    if e_kind == "E1":
      item = keep.get(item_id)
      if item is None:
        report.fail(f"{e_kind} {item_id}: not in keep_section")
        continue
      expected_draft = str(item.get("source_text") or "")
      if str(row.get("draft") or "") != expected_draft:
        report.fail(f"{e_kind} {item_id}: draft != keep_section.source_text")
    else:
      item = blind_items.get(item_id)
      if item is None:
        report.fail(f"{e_kind} {item_id}: not in blind_eval items")
        continue
      expected_draft = str(item.get("draft") or "")
      if str(row.get("draft") or "") != expected_draft:
        report.fail(f"{e_kind} {item_id}: draft != blind_eval.items.draft")

  if e_kind == "E1":
    for item_id, choice in judgments.items():
      if choice == DEGRADED_CHOICE:
        continue
      if choice == CHOICE_OK and item_id not in by_id:
        report.fail(f"E1 missing ok item {item_id}")
  elif required_ids is not None:
    for item_id in sorted(required_ids):
      if item_id not in by_id:
        report.fail(f"{e_kind} missing item {item_id}")

  return by_id


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=None)
  parser.add_argument("--data-dir", type=Path, default=None)
  parser.add_argument("--report", type=Path, default=None)
  args = parser.parse_args()

  root = (args.root or repo_root()).resolve()
  data_dir = (args.data_dir or e_data_dir(root)).resolve()
  report_path = args.report or (data_dir / "validation.json")

  manifest = load_manifest(data_dir, root) if (data_dir / "manifest.json").is_file() else expected_manifest(root)
  expected = manifest.get("expected") or {}
  e2_e3_ids = set(manifest.get("e2_e3_item_ids") or [])

  report = ValidationReport()
  by_kind: dict[str, dict[str, dict]] = {}

  for kind in E_KINDS:
    req = e2_e3_ids if kind in ("E2", "E3") else None
    by_kind[kind] = validate_kind(
      e_row_path(data_dir, kind),
      e_kind=kind,
      root=root,
      report=report,
      expected_count=int(expected.get(kind, 0)),
      required_ids=req,
    )

  e2_ids = set(by_kind.get("E2", {}))
  e3_ids = set(by_kind.get("E3", {}))
  if e2_ids and e3_ids and e2_ids != e3_ids:
    report.fail(
      f"E2/E3 item_id mismatch: only_e2={sorted(e2_ids - e3_ids)[:5]} "
      f"only_e3={sorted(e3_ids - e2_ids)[:5]}"
    )

  merged = data_dir / "e.jsonl"
  if merged.is_file():
    merged_rows = load_jsonl(merged)
    if len(merged_rows) != sum(report.counts.values()):
      report.fail(
        f"e.jsonl rows {len(merged_rows)} != sum of kinds {sum(report.counts.values())}"
      )

  out = {
    "ok": report.ok,
    "counts": report.counts,
    "expected": expected,
    "e2_e3_draft_scope": manifest.get("e2_e3_draft_scope"),
    "errors": report.errors,
    "warnings": report.warnings,
  }
  write_json(report_path, out)

  print(json.dumps(out, ensure_ascii=False, indent=2))
  if not report.ok:
    raise SystemExit(1)
  print(f"validation ok -> {report_path}")


if __name__ == "__main__":
  main()
