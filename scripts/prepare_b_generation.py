#!/usr/bin/env python3
"""B 生成実験用 records/manifest/長さレポートを CPU で準備する。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from b_generation_common import (
  CONDITIONS,
  MAX_NEW_TOKENS,
  MAX_TOTAL_TOKENS,
  compute_length_fields,
  dev_group_rank,
  eligibility_from_lengths,
  group_id_from_source_ids,
  load_yaml_config,
  manifest_sha256,
  select_dev_groups,
  sha256_text,
  write_json,
)
from pref_static_utils import load_jsonl
from section_middle_utils import load_jsonl as sm_load_jsonl, revisions_by_id, strip_code_fence
from setwise_triple_utils import reconstruct_triples_from_pref_rows


def original_split_from_meta(meta: dict) -> str:
  split = str((meta or {}).get("split") or "").strip()
  if split == "valid":
    return "valid"
  if split == "train":
    return "train"
  raise ValueError(f"unexpected meta.split {split!r}")


def build_records(
  *,
  keep_section_path: Path,
  revisions_path: Path,
  pref_train_path: Path,
  pref_valid_path: Path,
) -> tuple[list[dict], dict]:
  keep_rows = {str(r["id"]): r for r in sm_load_jsonl(keep_section_path)}
  revisions = revisions_by_id(sm_load_jsonl(revisions_path))
  pref_rows = load_jsonl(str(pref_train_path)) + load_jsonl(str(pref_valid_path))
  triples = reconstruct_triples_from_pref_rows(pref_rows)

  records: list[dict] = []
  checks: dict = {"missing_composer": [], "missing_keep": [], "source_ids": []}
  for triple in triples:
    sid = triple.item_id
    checks["source_ids"].append(sid)
    keep = keep_rows.get(sid)
    rev = revisions.get(sid)
    if keep is None:
      checks["missing_keep"].append(sid)
      continue
    if rev is None:
      checks["missing_composer"].append(sid)
      continue
    draft = str(keep["source_text"])
    human = str(keep["edited_text"])
    composer = strip_code_fence(str(rev.get("text") or ""))
    if draft != triple.draft or human != triple.human or composer != triple.composer:
      raise SystemExit(f"text mismatch for {sid}")
    records.append(
      {
        "source_id": sid,
        "draft": draft,
        "composer": composer,
        "human": human,
      }
    )

  records.sort(key=lambda r: r["source_id"])
  if checks["missing_keep"] or checks["missing_composer"]:
    raise SystemExit(json.dumps(checks, ensure_ascii=False))
  if len(records) != len(set(checks["source_ids"])):
    raise SystemExit("duplicate source_id in triples")
  return records, checks


def assign_splits(
  records: list[dict],
  pref_rows: list[dict],
  *,
  split_seed: int,
  dev_fraction: float,
) -> list[dict]:
  orig_by_id: dict[str, str] = {}
  for row in pref_rows:
    item_id = str((row.get("meta") or {}).get("item_id") or "")
    split = original_split_from_meta(row.get("meta") or {})
    if item_id in orig_by_id and orig_by_id[item_id] != split:
      raise SystemExit(f"conflicting original_split for {item_id}")
    orig_by_id[item_id] = split

  source_to_group: dict[str, str] = {}
  grouping_basis: dict[str, str] = {}
  for rec in records:
    gid, basis = group_id_from_source_ids([rec["source_id"]])
    source_to_group[rec["source_id"]] = gid
    grouping_basis[rec["source_id"]] = basis

  groups: dict[str, list[str]] = defaultdict(list)
  for rec in records:
    groups[source_to_group[rec["source_id"]]].append(rec["source_id"])

  holdout_groups: set[str] = set()
  for gid, members in groups.items():
    if any(orig_by_id.get(sid) == "valid" for sid in members):
      holdout_groups.add(gid)

  train_groups = sorted(g for g in groups if g not in holdout_groups)
  dev_groups = select_dev_groups(train_groups, dev_fraction=dev_fraction, split_seed=split_seed)

  sources: list[dict] = []
  for rec in records:
    sid = rec["source_id"]
    gid = source_to_group[sid]
    original = orig_by_id[sid]
    if gid in holdout_groups:
      split = "holdout"
      if original == "valid":
        split_reason = "original_valid"
      else:
        split_reason = "group_contains_valid"
    elif gid in dev_groups:
      split = "dev"
      split_reason = "hash_dev"
    else:
      split = "train"
      split_reason = "remaining_train"

    sources.append(
      {
        "source_id": sid,
        "document_id": None,
        "group_id": gid,
        "grouping_basis": grouping_basis[sid],
        "original_split": original,
        "split": split,
        "split_reason": split_reason,
        "eligible": True,
        "exclusion_reasons": [],
        "lengths": {},
        "text_sha256": {
          "draft": sha256_text(rec["draft"]),
          "composer": sha256_text(rec["composer"]),
          "human": sha256_text(rec["human"]),
        },
      }
    )
  return sources


def build_model_lock(tokenizer, cfg: dict) -> dict:
  template = getattr(tokenizer, "chat_template", "") or ""
  return {
    "model_id": cfg["model_id"],
    "model_revision": cfg["model_revision"],
    "tokenizer_revision": cfg["tokenizer_revision"],
    "eos_token_id": int(cfg["eos_token_id"]),
    "pad_token_id": int(cfg["pad_token_id"]),
    "enable_thinking": bool(cfg.get("enable_thinking", False)),
    "chat_template_sha256": sha256_text(template),
    "special_tokens": {
      "eos": cfg.get("eos_token"),
      "pad": cfg.get("pad_token"),
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--a2", type=Path, default=Path("data/revision_corpus/keep_section.jsonl"))
  parser.add_argument("--composer", type=Path, default=Path("data/section_middle/revisions.jsonl"))
  parser.add_argument("--pref-train", type=Path, default=Path("data/section_middle/pref_train.jsonl"))
  parser.add_argument("--pref-valid", type=Path, default=Path("data/section_middle/pref_valid.jsonl"))
  parser.add_argument("--dev-fraction", type=float, default=0.1)
  parser.add_argument("--split-seed", type=int, default=20260907)
  parser.add_argument("--model", default="Qwen/Qwen3-8B")
  parser.add_argument("--model-revision", default="b968826d9c46dd6066d109eabc6255188de91218")
  parser.add_argument("--config", type=Path, default=Path("configs/b_generation/common.yaml"))
  parser.add_argument("--out", type=Path, default=Path("data/b_generation"))
  args = parser.parse_args()

  code_root = Path(__file__).resolve().parents[1]
  out_dir = code_root / args.out
  out_dir.mkdir(parents=True, exist_ok=True)

  cfg = load_yaml_config(code_root / args.config)
  cfg["model_id"] = args.model
  cfg["model_revision"] = args.model_revision
  cfg["tokenizer_revision"] = args.model_revision
  system_text = str(cfg.get("system_text", "")).strip()
  enable_thinking = bool(cfg.get("enable_thinking", False))
  eos_token_id = int(cfg["eos_token_id"])

  records, checks = build_records(
    keep_section_path=code_root / args.a2,
    revisions_path=code_root / args.composer,
    pref_train_path=code_root / args.pref_train,
    pref_valid_path=code_root / args.pref_valid,
  )
  if len(records) != 378:
    checks["expected_source_count"] = 378
    checks["actual_source_count"] = len(records)
    write_json(out_dir / "data_checks.json", checks)
    raise SystemExit(f"expected 378 records, got {len(records)}")

  records_path = out_dir / "records.jsonl"
  with records_path.open("w", encoding="utf-8") as f:
    for row in records:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  pref_rows = load_jsonl(str(code_root / args.pref_train)) + load_jsonl(str(code_root / args.pref_valid))
  sources = assign_splits(records, pref_rows, split_seed=args.split_seed, dev_fraction=args.dev_fraction)
  record_by_id = {r["source_id"]: r for r in records}

  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    revision=args.model_revision,
  )
  if tokenizer.pad_token_id != int(cfg["pad_token_id"]):
    print(f"warning: pad_token_id={tokenizer.pad_token_id} config={cfg['pad_token_id']}", flush=True)

  length_rows = []
  train_over_counts = {c: 0 for c in CONDITIONS}
  infer_over_counts = {c: 0 for c in CONDITIONS}
  human_target_over_8192 = 0

  for src in sources:
    rec = record_by_id[src["source_id"]]
    lengths = compute_length_fields(
      tokenizer,
      draft=rec["draft"],
      composer=rec["composer"],
      human=rec["human"],
      system_text=system_text,
      enable_thinking=enable_thinking,
      eos_token_id=eos_token_id,
    )
    eligible, reasons = eligibility_from_lengths(lengths)
    src["lengths"] = lengths
    src["eligible"] = eligible
    src["exclusion_reasons"] = reasons
    target_tokens = lengths["x"]["target_tokens"]
    if target_tokens > MAX_NEW_TOKENS:
      human_target_over_8192 += 1
    for condition in CONDITIONS:
      if lengths[condition]["train_tokens"] > MAX_TOTAL_TOKENS:
        train_over_counts[condition] += 1
      if lengths[condition]["inference_total_tokens"] > MAX_TOTAL_TOKENS:
        infer_over_counts[condition] += 1
    length_rows.append({"source_id": src["source_id"], "lengths": lengths, "eligible": eligible})

  split_stats = defaultdict(lambda: {"total": 0, "eligible": 0, "groups": set()})
  for src in sources:
    split_stats[src["split"]]["total"] += 1
    split_stats[src["split"]]["groups"].add(src["group_id"])
    if src["eligible"]:
      split_stats[src["split"]]["eligible"] += 1

  split_summary = {
    split: {
      "total_sources": stats["total"],
      "eligible_sources": stats["eligible"],
      "excluded_sources": stats["total"] - stats["eligible"],
      "groups": len(stats["groups"]),
    }
    for split, stats in split_stats.items()
  }

  empty_splits = [s for s, v in split_summary.items() if v["eligible_sources"] == 0]
  length_report = {
    "schema_version": "b-length-report-v1",
    "max_total_tokens": MAX_TOTAL_TOKENS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "total_sources": len(sources),
    "eligible_sources": sum(1 for s in sources if s["eligible"]),
    "train_over_by_condition": train_over_counts,
    "infer_over_by_condition": infer_over_counts,
    "human_target_over_8192_reference": human_target_over_8192,
    "split_summary": split_summary,
    "empty_eligible_splits": empty_splits,
    "rows": length_rows,
  }

  model_lock = build_model_lock(tokenizer, cfg)
  write_json(out_dir / "model.lock.json", model_lock)

  manifest = {
    "schema_version": "b-generation-manifest-v1",
    "expected_source_count": 378,
    "records_file": "records.jsonl",
    "split_seed": args.split_seed,
    "dev_group_fraction": args.dev_fraction,
    "model_lock_file": "model.lock.json",
    "length_report_file": "b_length_report.json",
    "sources": sources,
  }
  manifest["manifest_sha256"] = manifest_sha256(manifest)

  write_json(out_dir / "manifest.json", manifest)
  write_json(out_dir / "b_length_report.json", length_report)
  write_json(
    out_dir / "schema_mapping.json",
    {
      "source_id": "meta.item_id from pref_train/pref_valid",
      "draft": "keep_section.source_text",
      "human": "keep_section.edited_text",
      "composer": "revisions.text (code fence stripped)",
      "original_split": "meta.split in pref rows (train/valid)",
    },
  )
  write_json(out_dir / "data_checks.json", {"source_count": len(records), "checks": checks})

  print(
    json.dumps(
      {
        "wrote": str(out_dir),
        "eligible": length_report["eligible_sources"],
        "holdout_eligible": split_summary.get("holdout", {}).get("eligible_sources"),
        "empty_splits": empty_splits,
      },
      ensure_ascii=False,
    ),
    flush=True,
  )
  if empty_splits:
    raise SystemExit(f"eligible split empty: {empty_splits}")


if __name__ == "__main__":
  main()
