#!/usr/bin/env python3
"""編集プロファイルから層別 Hard Eval セットを構築する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_THRESHOLDS = {
  "min_matched_sentences": 2,
  "min_alignment_coverage": 0.7,
  "high_expression_change": 0.15,
  "high_structure_change": 0.20,
  "low_expression_change": 0.15,
}


def load_profiles(path: Path) -> list[dict]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def structure_signal(profile: dict) -> float:
  sc = profile["structure_change"]
  return max(
    sc["order_inversion_rate"],
    sc["grouping_change_rate"],
    sc["paragraph_count_delta_rate"],
  )


def expression_total(profile: dict) -> float:
  return float(profile["expression_change"]["total"])


def expression_matched(profile: dict) -> float:
  return float(profile["expression_change"]["matched"])


def coverage_ok(profile: dict, *, min_coverage: float) -> bool:
  cov = profile["alignment"].get("coverage") or {}
  return cov.get("source", 0.0) >= min_coverage and cov.get("edited", 0.0) >= min_coverage


def classify_profile_legacy(profile: dict, *, thresholds: dict) -> str | None:
  """修正前: coverage 条件なし。"""
  if profile["alignment"]["matched_count"] < thresholds["min_matched_sentences"]:
    return None
  expr = expression_matched(profile)
  struct = structure_signal(profile)
  if struct >= thresholds["high_structure_change"] and expr < thresholds["low_expression_change"]:
    return "structure-dominant"
  if expr >= thresholds["high_expression_change"] and struct < thresholds["high_structure_change"]:
    return "expression-dominant"
  if expr >= thresholds["high_expression_change"] and struct >= thresholds["high_structure_change"]:
    return "mixed"
  return None


def classify_profile(profile: dict, *, thresholds: dict) -> str | None:
  if profile["alignment"]["matched_count"] < thresholds["min_matched_sentences"]:
    return None
  expr_total_val = expression_total(profile)
  expr_matched_val = expression_matched(profile)
  struct = structure_signal(profile)
  high_expr = expr_total_val >= thresholds["high_expression_change"]
  high_struct = struct >= thresholds["high_structure_change"]
  low_expr_matched = expr_matched_val < thresholds["low_expression_change"]

  if (
    high_struct
    and low_expr_matched
    and coverage_ok(profile, min_coverage=thresholds["min_alignment_coverage"])
  ):
    return "structure-dominant"
  if high_expr and not high_struct:
    return "expression-dominant"
  if high_expr and high_struct:
    return "mixed"
  return None


def token_length(tokenizer, base_text: str, candidate_text: str) -> int:
  return len(
    tokenizer(
      base_text,
      candidate_text,
      truncation=False,
      add_special_tokens=True,
    )["input_ids"]
  )


def build_hard_eval_item(profile: dict, *, idx: int, stratum: str, tokenizer) -> dict | None:
  base_text = profile["source_text"].strip("\n") + "\n"
  human_text = profile["edited_text"].strip("\n") + "\n"
  if base_text.strip() == human_text.strip():
    return None

  candidates = [
    {
      "id": "human",
      "text": human_text,
      "generator": "human",
      "prompt_tag": "real-edit",
    },
    {
      "id": "base",
      "text": base_text,
      "generator": "copy",
      "prompt_tag": "identity",
    },
  ]
  for cand in candidates:
    if token_length(tokenizer, base_text, cand["text"]) > 2048:
      return None

  fits_512 = all(token_length(tokenizer, base_text, cand["text"]) <= 512 for cand in candidates)
  slug = profile["id"].replace("/", "__")[:80]
  return {
    "id": f"se-{stratum[:3]}-{idx:03d}-{slug}",
    "seed_text": "",
    "seed_meta": {
      "project_id": profile.get("project_id"),
      "source_profile_id": profile["id"],
      "dataset": profile["dataset"],
      "stratum": stratum,
      "expression_change": profile["expression_change"],
      "structure_change": profile["structure_change"],
      "alignment_summary": {
        "matched_count": profile["alignment"]["matched_count"],
        "added_count": profile["alignment"]["added_count"],
        "deleted_count": profile["alignment"]["deleted_count"],
        "mean_similarity": profile["alignment"]["mean_similarity"],
        "coverage": profile["alignment"].get("coverage", {}),
      },
    },
    "base_text": base_text,
    "base_generator": "draft",
    "candidates": candidates,
    "human": {
      "best_id": "human",
      "rank": ["human", "base"],
      "notes": "real edit vs draft; structure-eval stratum",
    },
    "status": "labeled",
    "meta": {
      "stratum": stratum,
      "fits_512": fits_512,
    },
  }


def write_jsonl(rows: list[dict], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_eval_bucket(
  profiles: list[dict],
  *,
  stratum: str,
  out_dir: Path,
  tokenizer,
) -> tuple[list[dict], int]:
  items: list[dict] = []
  dropped_token = 0
  for idx, profile in enumerate(profiles, start=1):
    item = build_hard_eval_item(profile, idx=idx, stratum=stratum, tokenizer=tokenizer)
    if item is None:
      dropped_token += 1
      continue
    items.append(item)
  out_path = out_dir / f"{stratum.replace('-', '_')}.jsonl"
  write_jsonl(items, out_path)
  return items, dropped_token


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", default="data/section_edit_profile.jsonl")
  parser.add_argument("--out-dir", default="data/hard_eval")
  parser.add_argument("--train-candidates", default="data/structure_train_candidates.jsonl")
  parser.add_argument("--tokenizer", default="outputs/pref-ce-ml2048/model")
  parser.add_argument("--min-matched", type=int, default=2)
  parser.add_argument("--min-coverage", type=float, default=0.7)
  parser.add_argument("--high-expression", type=float, default=0.15)
  parser.add_argument("--high-structure", type=float, default=0.20)
  args = parser.parse_args()

  from transformers import AutoTokenizer

  root = Path(__file__).resolve().parent.parent
  input_path = Path(args.input)
  if not input_path.is_absolute():
    input_path = root / input_path
  out_dir = Path(args.out_dir)
  if not out_dir.is_absolute():
    out_dir = root / out_dir
  train_candidates_path = Path(args.train_candidates)
  if not train_candidates_path.is_absolute():
    train_candidates_path = root / train_candidates_path

  thresholds = {
    "min_matched_sentences": args.min_matched,
    "min_alignment_coverage": args.min_coverage,
    "high_expression_change": args.high_expression,
    "high_structure_change": args.high_structure,
    "low_expression_change": args.high_expression,
  }

  profiles = load_profiles(input_path)
  tokenizer = AutoTokenizer.from_pretrained(
    str(root / args.tokenizer) if not Path(args.tokenizer).is_absolute() else args.tokenizer
  )

  heldout = [p for p in profiles if p.get("dataset") == "heldout"]
  raw = [p for p in profiles if p.get("dataset") == "raw"]

  strata: dict[str, list[dict]] = {
    "structure-dominant": [],
    "expression-dominant": [],
    "mixed": [],
  }
  unclassified_profiles: list[dict] = []
  excluded_from_structure_dominant: list[dict] = []

  for profile in heldout:
    new_label = classify_profile(profile, thresholds=thresholds)
    old_label = classify_profile_legacy(profile, thresholds=thresholds)
    if old_label == "structure-dominant" and new_label != "structure-dominant":
      cov = profile["alignment"].get("coverage", {})
      excluded_from_structure_dominant.append(
        {
          "id": profile["id"],
          "project_id": profile.get("project_id"),
          "reason": "alignment_coverage_below_threshold",
          "coverage_source": cov.get("source"),
          "coverage_edited": cov.get("edited"),
          "expression_change_matched": profile["expression_change"]["matched"],
          "expression_change_total": profile["expression_change"]["total"],
          "structure_signal": structure_signal(profile),
          "matched_count": profile["alignment"]["matched_count"],
          "added_count": profile["alignment"]["added_count"],
          "deleted_count": profile["alignment"]["deleted_count"],
        }
      )
    if new_label is None:
      unclassified_profiles.append(profile)
      continue
    strata[new_label].append(profile)

  train_candidates = []
  for profile in raw:
    if classify_profile(profile, thresholds=thresholds) == "structure-dominant":
      train_candidates.append(
        {
          "id": profile["id"],
          "project_id": profile.get("project_id"),
          "expression_change": profile["expression_change"],
          "structure_change": profile["structure_change"],
          "alignment": {
            "matched_count": profile["alignment"]["matched_count"],
            "mean_similarity": profile["alignment"]["mean_similarity"],
            "coverage": profile["alignment"].get("coverage", {}),
          },
        }
      )

  report = {
    "thresholds": thresholds,
    "structure_signal_definition": "max(order_inversion_rate, grouping_change_rate, paragraph_count_delta_rate)",
    "expression_signal_for_layers": "expression_change.total (high), expression_change.matched (low for structure-dominant)",
    "heldout_total": len(heldout),
    "heldout_unclassified": len(unclassified_profiles),
    "legacy_structure_dominant_count": sum(
      1 for p in heldout if classify_profile_legacy(p, thresholds=thresholds) == "structure-dominant"
    ),
    "excluded_from_structure_dominant_count": len(excluded_from_structure_dominant),
    "excluded_from_structure_dominant_examples": excluded_from_structure_dominant[:10],
    "strata_counts": {},
    "strata_bucket_counts": {},
    "fits_512_counts": {},
    "token_dropped": {},
  }

  for stratum, bucket in strata.items():
    items, dropped_token = write_eval_bucket(
      bucket, stratum=stratum, out_dir=out_dir, tokenizer=tokenizer
    )
    report["strata_bucket_counts"][stratum] = len(bucket)
    report["strata_counts"][stratum] = len(items)
    report["fits_512_counts"][stratum] = sum(1 for it in items if it["meta"]["fits_512"])
    report["token_dropped"][stratum] = dropped_token
    print(f"{stratum:22s} held-out={len(bucket):3d} wrote={len(items):3d} dropped={dropped_token}")

  unclassified_items, unclassified_dropped = write_eval_bucket(
    unclassified_profiles,
    stratum="unclassified",
    out_dir=out_dir,
    tokenizer=tokenizer,
  )
  report["unclassified_eval_count"] = len(unclassified_items)
  report["unclassified_token_dropped"] = unclassified_dropped
  report["unclassified_fits_512_count"] = sum(1 for it in unclassified_items if it["meta"]["fits_512"])
  print(
    f"{'unclassified':22s} held-out={len(unclassified_profiles):3d} "
    f"wrote={len(unclassified_items):3d} dropped={unclassified_dropped}"
  )

  write_jsonl(train_candidates, train_candidates_path)
  report["train_candidates_raw_structure_dominant"] = len(train_candidates)

  report_path = out_dir / "structure_eval_build_report.json"
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print("-" * 64)
  print(f"legacy structure-dominant: {report['legacy_structure_dominant_count']}")
  print(f"excluded by coverage fix: {report['excluded_from_structure_dominant_count']}")
  print(f"unclassified held-out: {len(unclassified_profiles)}")
  print(f"train candidates (raw structure-dominant): {len(train_candidates)}")
  print(f"report: {report_path}")


if __name__ == "__main__":
  main()
