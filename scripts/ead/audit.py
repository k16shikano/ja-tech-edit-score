#!/usr/bin/env python3
"""タスク 0-1: スキーマと分割の検証。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = repo_root = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  drafts_from_pref,
  ead_reports,
  ead_work,
  id_from_row,
  ids_from_edit_sft,
  load_jsonl,
  md_table,
  repo_root,
  schema_stats,
  write_json,
)

ADAPTER_CANDIDATES = (
  "outputs/Qwen__Qwen3-8B-pairsplit",
  "outputs/Qwen__Qwen3-8B-pairsplit-v2",
  "outputs/edit-sft/Qwen__Qwen3-8B",
  "outputs/edit-sft",
  "outputs/Qwen__Qwen3-8B-re",
  "outputs/Qwen__Qwen3-8B-revised-2000",
)


def set_count(a: set, b: set) -> dict:
  inter = a & b
  return {"a": len(a), "b": len(b), "intersection": len(inter), "ids": sorted(inter)}


def check_union(name: str, parts: list[tuple[str, set[str]]], expected: set[str]) -> dict:
  union: set[str] = set()
  for _, s in parts:
    union |= s
  return {
    "name": name,
    "expected": len(expected),
    "union": len(union),
    "match": union == expected,
    "part_counts": {label: len(s) for label, s in parts},
    "missing_from_union": sorted(expected - union),
    "extra_in_union": sorted(union - expected),
  }


def inspect_adapter_provenance(root: Path, blind_ids: set[str]) -> dict:
  report: dict = {"candidates": [], "selected": None, "steps": {}}
  train_all = root / "data/edit_sft_all/train.jsonl"
  expected_n = len(load_jsonl(train_all)) if train_all.is_file() else None

  for rel in ADAPTER_CANDIDATES:
    base = root / rel
    adapter_dir = base / "adapter" if (base / "adapter").is_dir() else base
    meta_path = base / "train_meta.json"
    if not adapter_dir.is_dir() and not meta_path.is_file():
      continue
    entry = {"path": str(base.relative_to(root)), "adapter_dir": str(adapter_dir.relative_to(root))}
    if meta_path.is_file():
      entry["train_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    report["candidates"].append(entry)

  selected = None
  for cand in report["candidates"]:
    meta = cand.get("train_meta") or {}
    n_train = meta.get("n_train")
    if expected_n is not None and n_train == expected_n:
      selected = cand
      break
  if selected is None and report["candidates"]:
    selected = report["candidates"][0]

  report["selected"] = selected
  report["expected_train_n"] = expected_n

  if not selected:
    report["steps"] = {
      "status": "unverified",
      "reason": "adapter checkpoint not found",
    }
    return report

  adapter_dir = root / selected["adapter_dir"]
  step1: dict = {"adapter_config": (adapter_dir / "adapter_config.json").is_file()}
  train_meta = selected.get("train_meta") or {}
  step1["train_meta_path"] = str((root / selected["path"]) / "train_meta.json")
  step1["train_meta"] = train_meta
  step1["training_args"] = (adapter_dir / "training_args.bin").is_file()

  train_path = train_meta.get("train_file") or train_meta.get("train_path")
  step1["recorded_train_path"] = train_path

  overlap = None
  if train_path:
    tp = Path(train_path)
    if not tp.is_absolute():
      tp = root / tp
    if tp.is_file():
      train_ids = ids_from_edit_sft(load_jsonl(tp))
      overlap = set_count(train_ids, blind_ids)
  elif expected_n is not None and train_meta.get("n_train") == expected_n:
    train_ids = ids_from_edit_sft(load_jsonl(train_all))
    overlap = set_count(train_ids, blind_ids)

  count_match = train_meta.get("n_train") == expected_n
  report["steps"] = {
    "1_config_and_meta": step1,
    "2_train_blind_overlap": overlap,
    "3_count_match": {
      "meta_n_train": train_meta.get("n_train"),
      "edit_sft_all_train_n": expected_n,
      "match": count_match,
    },
  }
  if overlap is None:
    report["steps"]["status"] = "unverified"
    report["steps"]["reason"] = (
      "training data path not recorded"
      if not count_match
      else "training data path not recorded"
    )
  elif overlap["intersection"] != 0:
    report["steps"]["status"] = "leak"
  elif count_match:
    report["steps"]["status"] = "verified"
  else:
    report["steps"]["status"] = "unverified"
    report["steps"]["reason"] = "n_train does not match edit_sft_all/train.jsonl"
  return report


def build_split_map(root: Path) -> dict[str, str]:
  mapping: dict[str, str] = {}
  sources = [
    ("hunk_train", root / "data/edit_sft_hunk_nopara/train.jsonl"),
    ("hunk_heldout", root / "data/edit_sft_hunk_nopara/heldout.jsonl"),
    ("section_train", root / "data/edit_sft_section/train.jsonl"),
    ("section_heldout", root / "data/edit_sft_section/heldout.jsonl"),
    ("blind_c", root / "data/blind_eval/items.jsonl"),
  ]
  for label, path in sources:
    if not path.is_file():
      continue
    for row in load_jsonl(path):
      item_id = id_from_row(row)
      if item_id:
        mapping[item_id] = label
  d_train_ids = {str(r["item_id"]) for r in load_jsonl(root / "data/d/train.jsonl")}
  d_valid_ids = {str(r["item_id"]) for r in load_jsonl(root / "data/d/valid.jsonl")}
  for item_id in d_train_ids:
    mapping.setdefault(item_id, "d_train")
  for item_id in d_valid_ids:
    mapping[item_id] = "d_valid"
  return mapping


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  args = parser.parse_args()
  root = args.root.resolve()

  paths = {
    "keep_hunk": root / "data/revision_corpus/keep_hunk_nopara.jsonl",
    "keep_section": root / "data/revision_corpus/keep_section.jsonl",
    "canonical": root / "data/revision_corpus/canonical.jsonl",
    "hunk_train": root / "data/edit_sft_hunk_nopara/train.jsonl",
    "hunk_heldout": root / "data/edit_sft_hunk_nopara/heldout.jsonl",
    "section_train": root / "data/edit_sft_section/train.jsonl",
    "section_heldout": root / "data/edit_sft_section/heldout.jsonl",
    "edit_sft_all_train": root / "data/edit_sft_all/train.jsonl",
    "blind_items": root / "data/blind_eval/items.jsonl",
    "pref_train": root / "data/section_middle/pref_train.jsonl",
    "pref_valid": root / "data/section_middle/pref_valid.jsonl",
    "d_train": root / "data/d/train.jsonl",
    "d_valid": root / "data/d/valid.jsonl",
  }

  loaded: dict[str, list[dict]] = {}
  missing = [name for name, path in paths.items() if not path.is_file()]
  if missing:
    raise SystemExit(f"missing required files: {missing}")
  for name, path in paths.items():
    loaded[name] = load_jsonl(path)

  id_sets = {
    "keep_hunk": {str(r["id"]) for r in loaded["keep_hunk"]},
    "keep_section": {str(r["id"]) for r in loaded["keep_section"]},
    "canonical": {str(r["id"]) for r in loaded["canonical"]},
    "hunk_train": ids_from_edit_sft(loaded["hunk_train"]),
    "hunk_heldout": ids_from_edit_sft(loaded["hunk_heldout"]),
    "section_train": ids_from_edit_sft(loaded["section_train"]),
    "section_heldout": ids_from_edit_sft(loaded["section_heldout"]),
    "edit_sft_all_train": ids_from_edit_sft(loaded["edit_sft_all_train"]),
    "blind_items": {str(r["id"]) for r in loaded["blind_items"]},
    "d_all": {str(r["item_id"]) for r in loaded["d_train"] + loaded["d_valid"]},
    "d_train": {str(r["item_id"]) for r in loaded["d_train"]},
    "d_valid": {str(r["item_id"]) for r in loaded["d_valid"]},
  }
  pref_train_drafts = drafts_from_pref(loaded["pref_train"])
  pref_valid_drafts = drafts_from_pref(loaded["pref_valid"])
  section_heldout_drafts = drafts_from_pref(loaded["section_heldout"])
  section_train_drafts = drafts_from_pref(loaded["section_train"])

  s1 = check_union(
    "S1 hunk train+heldout=keep_hunk",
    [("train", id_sets["hunk_train"]), ("heldout", id_sets["hunk_heldout"])],
    id_sets["keep_hunk"],
  )
  s2 = check_union(
    "S2 section train+heldout=keep_section",
    [("train", id_sets["section_train"]), ("heldout", id_sets["section_heldout"])],
    id_sets["keep_section"],
  )
  s3 = check_union(
    "S3 keep_hunk+keep_section=canonical",
    [("hunk", id_sets["keep_hunk"]), ("section", id_sets["keep_section"])],
    id_sets["canonical"],
  )
  hunk_section_heldout = id_sets["hunk_heldout"] | id_sets["section_heldout"]
  s4 = {
    "name": "S4 blind ⊆ hunk_heldout ∪ section_heldout",
    "blind_n": len(id_sets["blind_items"]),
    "heldout_union_n": len(hunk_section_heldout),
    "subset": id_sets["blind_items"] <= hunk_section_heldout,
    "missing": sorted(id_sets["blind_items"] - hunk_section_heldout),
  }

  leaks = {
    "L1 hunk train∩heldout": set_count(id_sets["hunk_train"], id_sets["hunk_heldout"]),
    "L2 section train∩heldout": set_count(id_sets["section_train"], id_sets["section_heldout"]),
    "L3 edit_sft_all train∩blind": set_count(id_sets["edit_sft_all_train"], id_sets["blind_items"]),
    "L4 pref_train drafts∩pref_valid drafts": {
      "intersection": len(pref_train_drafts & pref_valid_drafts),
    },
    "L5 D items∩blind": set_count(id_sets["d_all"], id_sets["blind_items"]),
  }

  x1 = set_count(id_sets["d_all"], id_sets["hunk_train"])
  x2 = set_count(id_sets["d_all"], id_sets["hunk_heldout"])
  pref_valid_item_ids = {
    str(r.get("meta", {}).get("item_id") or r.get("id") or "")
    for r in loaded["pref_valid"]
  }
  pref_valid_item_ids.discard("")
  x3_train_overlap = pref_valid_item_ids & id_sets["section_train"]
  x3 = {
    "draft_intersection_heldout": len(pref_valid_drafts & section_heldout_drafts),
    "item_intersection_heldout": len(pref_valid_item_ids & id_sets["section_heldout"]),
    "item_intersection_train": len(x3_train_overlap),
    "pref_valid_items": len(pref_valid_item_ids),
  }
  x4 = {"intersection": len(drafts_from_pref(loaded["pref_train"]) & section_train_drafts)}

  exclude_ids_d = sorted(x1["ids"])
  exclude_ids_b = sorted(x3_train_overlap)

  split_map = build_split_map(root)
  adapter = inspect_adapter_provenance(root, id_sets["blind_items"])

  schema = [
    schema_stats(loaded["edit_sft_all_train"], name="edit_sft_all/train"),
    schema_stats(loaded["keep_hunk"], name="keep_hunk_nopara"),
    schema_stats(loaded["keep_section"], name="keep_section"),
    schema_stats(loaded["blind_items"], name="blind_eval/items"),
    schema_stats(loaded["pref_valid"], name="section_middle/pref_valid"),
    schema_stats(loaded["d_train"] + loaded["d_valid"], name="d/all"),
  ]

  stop_reasons: list[str] = []
  for key in (s1, s2, s3):
    if not key["match"]:
      stop_reasons.append(key["name"])
  if not s4["subset"]:
    stop_reasons.append(s4["name"])
  for label, info in leaks.items():
    if label.startswith("L") and label != "L4 pref_train drafts∩pref_valid drafts":
      if info.get("intersection", 0) != 0:
        stop_reasons.append(label)

  summary = {
    "structural": {"S1": s1, "S2": s2, "S3": s3, "S4": s4},
    "leaks": leaks,
    "crosses": {"X1": x1, "X2": x2, "X3": x3, "X4": x4},
    "exclude_ids_D_n": len(exclude_ids_d),
    "exclude_ids_B_n": len(exclude_ids_b),
    "phase2_train_note": (
      "pref_valid item ids overlap section train; exclude from excl training"
      if x3_train_overlap
      else "B evaluation clean under edit_sft_all/train (pref_valid is section heldout)"
    ),
    "adapter_provenance": adapter,
    "schema": schema,
    "stop_reasons": stop_reasons,
  }

  work = ead_work()
  reports = ead_reports()
  write_json(work / "exclude_ids_D.json", {"ids": exclude_ids_d})
  write_json(work / "exclude_ids_B.json", {"ids": exclude_ids_b})
  write_json(work / "split_map.json", split_map)
  write_json(reports / "ead-audit.json", summary)

  lines = [
    "# ead-audit",
    "",
    "## 分割の整合性",
    "",
    md_table(
      ["検査", "期待", "結果", "成立"],
      [
        ["S1", "1387+210=1597", f"{s1['part_counts']}", "✓" if s1["match"] else "✗"],
        ["S2", "329+50=379", f"{s2['part_counts']}", "✓" if s2["match"] else "✗"],
        ["S3", "1597+379=1976", f"{s3['part_counts']}", "✓" if s3["match"] else "✗"],
        [
          "S4",
          "60 ⊆ hunk heldout ∪ section heldout",
          f"blind={s4['blind_n']} missing={len(s4['missing'])}",
          "✓" if s4["subset"] else "✗",
        ],
      ],
    ),
    "",
    "## リーク検査",
    "",
    md_table(
      ["検査", "交差", "期待"],
      [
        ["L1", str(leaks["L1 hunk train∩heldout"]["intersection"]), "0"],
        ["L2", str(leaks["L2 section train∩heldout"]["intersection"]), "0"],
        ["L3", str(leaks["L3 edit_sft_all train∩blind"]["intersection"]), "0"],
        ["L4", str(leaks["L4 pref_train drafts∩pref_valid drafts"]["intersection"]), "0"],
        ["L5", str(leaks["L5 D items∩blind"]["intersection"]), "0"],
      ],
    ),
    "",
    "## 交差（判断用）",
    "",
    md_table(
      ["検査", "件数"],
      [
        ["X1 D ∩ hunk train", str(x1["intersection"])],
        ["X2 D ∩ hunk heldout", str(x2["intersection"])],
        [
          "X3 pref_valid ∩ section heldout (items)",
          f"{x3['item_intersection_heldout']}/{x3['pref_valid_items']}",
        ],
        ["X3 pref_valid ∩ section train (items)", str(x3["item_intersection_train"])],
        ["X4 pref_train drafts ∩ section train", str(x4["intersection"])],
      ],
    ),
    "",
    f"Phase 2 学習集合: {summary['phase2_train_note']}",
    "",
    "## 編集 SFT アダプタ由来",
    "",
    f"状態: {adapter['steps'].get('status', 'unknown')}",
    "",
    "```json",
    json.dumps(adapter, ensure_ascii=False, indent=2),
    "```",
    "",
    "## スキーマ統計",
    "",
  ]
  for item in schema:
    lines.append(f"### {item['name']} ({item['n']} 行)")
    lines.append(f"- keys: {', '.join(item['keys'][:12])}{'…' if len(item['keys']) > 12 else ''}")
    lines.append(
      f"- draft 文字長: median={item['char_len']['draft'].get('median')} "
      f"mean={item['char_len']['draft'].get('mean')}"
    )
    lines.append("")

  if stop_reasons:
    lines.extend(["## 停止条件", ""] + [f"- {r}" for r in stop_reasons])

  (reports / "ead-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-audit.md"), "stop_reasons": stop_reasons}, ensure_ascii=False))


if __name__ == "__main__":
  main()
