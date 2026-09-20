#!/usr/bin/env python3
"""Phase 3 タスク 3-1: 欠陥関門 G1（規則ベース）。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  draft_from_row,
  ead_reports,
  ead_work,
  fmt_rate,
  load_jsonl,
  md_table,
  normalized_levenshtein,
  repo_root,
  write_json,
  write_jsonl,
)

CONTENT_TERM_RE = re.compile(
  r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*|[一-龥]{2,}|[ァ-ヴー]{2,}|[ぁ-ん]{3,}"
)


def content_terms(text: str) -> set[str]:
  return {m.group(0).lower() for m in CONTENT_TERM_RE.finditer(text)}


def term_rates(draft: str, cand: str) -> tuple[float, float]:
  d_terms = content_terms(draft)
  c_terms = content_terms(cand)
  if not d_terms:
    draft_missing = 0.0
  else:
    draft_missing = len(d_terms - c_terms) / len(d_terms)
  if not c_terms:
    cand_novel = 0.0
  else:
    cand_novel = len(c_terms - d_terms) / len(c_terms)
  return draft_missing, cand_novel


@dataclass
class DefectThresholds:
  epsilon: float
  char_ratio_low: float
  char_ratio_high: float
  draft_missing_high: float
  cand_novel_high: float

  def to_dict(self) -> dict[str, float]:
    return {
      "epsilon": self.epsilon,
      "char_ratio_low": self.char_ratio_low,
      "char_ratio_high": self.char_ratio_high,
      "draft_missing_high": self.draft_missing_high,
      "cand_novel_high": self.cand_novel_high,
    }


@dataclass
class DefectResult:
  rejected: bool
  reason: str | None
  edit_distance_norm: float
  char_ratio: float
  draft_missing_rate: float
  cand_novel_rate: float


def measure_pair(draft: str, cand: str) -> dict[str, float]:
  draft_missing, cand_novel = term_rates(draft, cand)
  return {
    "edit_distance_norm": normalized_levenshtein(draft, cand),
    "char_ratio": len(cand) / max(len(draft), 1),
    "draft_missing_rate": draft_missing,
    "cand_novel_rate": cand_novel,
  }


def detect_defect(draft: str, cand: str, thresholds: DefectThresholds) -> DefectResult:
  m = measure_pair(draft, cand)
  if m["edit_distance_norm"] < thresholds.epsilon:
    return DefectResult(True, "no_edit", **m)
  if m["draft_missing_rate"] > thresholds.draft_missing_high or m["char_ratio"] < thresholds.char_ratio_low:
    return DefectResult(True, "omission", **m)
  if m["cand_novel_rate"] > thresholds.cand_novel_high or m["char_ratio"] > thresholds.char_ratio_high:
    return DefectResult(True, "addition", **m)
  return DefectResult(False, None, **m)


def human_pairs_from_edit_sft(rows: list[dict]) -> list[tuple[str, str]]:
  pairs: list[tuple[str, str]] = []
  for row in rows:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
      continue
    draft = draft_from_row(row) or ""
    cand = str(messages[1].get("content") or "")
    if draft and cand:
      pairs.append((draft, cand))
  return pairs


def calibrate_thresholds(pairs: list[tuple[str, str]], *, min_pass_rate: float) -> DefectThresholds:
  metrics = [measure_pair(d, c) for d, c in pairs]
  edit = np.asarray([m["edit_distance_norm"] for m in metrics], dtype=np.float64)
  ratio = np.asarray([m["char_ratio"] for m in metrics], dtype=np.float64)
  miss = np.asarray([m["draft_missing_rate"] for m in metrics], dtype=np.float64)
  novel = np.asarray([m["cand_novel_rate"] for m in metrics], dtype=np.float64)

  epsilon = float(np.quantile(edit, 0.01))
  char_ratio_low = float(np.quantile(ratio, 0.01))
  char_ratio_high = float(np.quantile(ratio, 0.99))
  draft_missing_high = float(np.quantile(miss, 0.99))
  cand_novel_high = float(np.quantile(novel, 0.99))

  thresholds = DefectThresholds(
    epsilon=max(epsilon, 1e-6),
    char_ratio_low=char_ratio_low,
    char_ratio_high=char_ratio_high,
    draft_missing_high=draft_missing_high,
    cand_novel_high=cand_novel_high,
  )

  for _ in range(20):
    passed = sum(1 for d, c in pairs if not detect_defect(d, c, thresholds).rejected)
    if passed / len(pairs) >= min_pass_rate:
      break
    thresholds = DefectThresholds(
      epsilon=thresholds.epsilon * 0.85,
      char_ratio_low=thresholds.char_ratio_low * 0.95,
      char_ratio_high=thresholds.char_ratio_high * 1.05,
      draft_missing_high=thresholds.draft_missing_high * 1.05,
      cand_novel_high=thresholds.cand_novel_high * 1.05,
    )
  return thresholds


def rows_from_d(rows: list[dict]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  for row in rows:
    draft = str(row.get("draft") or "")
    cand = str(row.get("y") or "")
    if not draft or not cand:
      continue
    out.append(
      {
        "row_id": row.get("row_id"),
        "item_id": row.get("item_id"),
        "position": row.get("position"),
        "source": row.get("y_kind") or "d",
        "draft": draft,
        "candidate": cand,
      }
    )
  return out


def rows_from_adapter_samples(rows: list[dict]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  for row in rows:
    draft = str(row.get("draft") or "")
    cand = str(row.get("generated") or "")
    if not draft or not cand:
      continue
    out.append(
      {
        "row_id": f"{row.get('id')}::sample{row.get('sample_index')}",
        "item_id": row.get("id"),
        "position": None,
        "source": "adapter",
        "draft": draft,
        "candidate": cand,
      }
    )
  return out


def apply_gate(rows: list[dict[str, Any]], thresholds: DefectThresholds) -> list[dict[str, Any]]:
  scored: list[dict[str, Any]] = []
  for row in rows:
    result = detect_defect(row["draft"], row["candidate"], thresholds)
    scored.append(
      {
        **row,
        "rejected": result.rejected,
        "reject_reason": result.reason,
        "edit_distance_norm": result.edit_distance_norm,
        "char_ratio": result.char_ratio,
        "draft_missing_rate": result.draft_missing_rate,
        "cand_novel_rate": result.cand_novel_rate,
      }
    )
  return scored


def summarize_gate(rows: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
  n = len(rows)
  rejected = [r for r in rows if r["rejected"]]
  by_reason: dict[str, int] = {}
  for row in rejected:
    reason = str(row.get("reject_reason") or "unknown")
    by_reason[reason] = by_reason.get(reason, 0) + 1
  return {
    "name": name,
    "n": n,
    "rejected_n": len(rejected),
    "pass_n": n - len(rejected),
    "pass_rate": ((n - len(rejected)) / n) if n else None,
    "by_reason": by_reason,
  }


def label_a_stats(d_rows: list[dict[str, Any]]) -> dict[str, Any]:
  a_rows = [r for r in d_rows if str(r.get("position")) == "a"]
  if not a_rows:
    return {"n": 0}
  rejected = sum(1 for r in a_rows if r["rejected"])
  return {
    "n": len(a_rows),
    "rejected_n": rejected,
    "reject_rate": rejected / len(a_rows),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--calibration-train", type=Path, default=Path("data/edit_sft_all/train.jsonl"))
  parser.add_argument("--d-train", type=Path, default=Path("data/d/train.jsonl"))
  parser.add_argument("--d-valid", type=Path, default=Path("data/d/valid.jsonl"))
  parser.add_argument(
    "--adapter-samples",
    type=Path,
    default=Path("outputs/edit-sft-eval-v3/adapter_samples.jsonl"),
  )
  parser.add_argument("--min-human-pass-rate", type=float, default=0.99)
  args = parser.parse_args()

  root = args.root.resolve()
  calib_rows = load_jsonl(root / args.calibration_train)
  human_pairs = human_pairs_from_edit_sft(calib_rows)
  if not human_pairs:
    raise SystemExit("no human pairs for calibration")

  thresholds = calibrate_thresholds(human_pairs, min_pass_rate=args.min_human_pass_rate)
  human_results = apply_gate(
    [{"draft": d, "candidate": c, "source": "human_calib"} for d, c in human_pairs],
    thresholds,
  )
  human_summary = summarize_gate(human_results, name="edit_sft_all_train_human")

  d_train = apply_gate(rows_from_d(load_jsonl(root / args.d_train)), thresholds)
  d_valid = apply_gate(rows_from_d(load_jsonl(root / args.d_valid)), thresholds)
  d_all = d_train + d_valid
  adapter_rows = apply_gate(rows_from_adapter_samples(load_jsonl(root / args.adapter_samples)), thresholds)

  summary = {
    "thresholds": thresholds.to_dict(),
    "calibration_n": len(human_pairs),
    "human_calib": human_summary,
    "d_train": summarize_gate(d_train, name="d_train"),
    "d_valid": summarize_gate(d_valid, name="d_valid"),
    "d_all": summarize_gate(d_all, name="d_all"),
    "d_label_a_train": label_a_stats(d_train),
    "d_label_a_valid": label_a_stats(d_valid),
    "d_label_a_all": label_a_stats(d_all),
    "adapter_samples": summarize_gate(adapter_rows, name="adapter_samples"),
  }

  work = ead_work()
  write_json(work / "gate_defect_thresholds.json", summary)
  write_jsonl(work / "gate_defect_d_scored.jsonl", d_all)
  write_jsonl(work / "gate_defect_adapter_scored.jsonl", adapter_rows)

  reports = ead_reports()
  write_json(reports / "ead-gate-defect.json", summary)

  lines = [
    "# ead-gate-defect",
    "",
    "欠陥関門 G1。閾値は `edit_sft_all/train.jsonl` の人間推敲分布から校準。",
    "",
    "## 閾値",
    "",
    md_table(
      ["項目", "値"],
      [[k, f"{v:.6f}"] for k, v in summary["thresholds"].items()],
    ),
    "",
    "## 通過率",
    "",
    md_table(
      ["集合", "n", "通過", "通過率", "棄却内訳"],
      [
        [
          s["name"],
          s["n"],
          s["pass_n"],
          fmt_pct(s["pass_rate"]),
          ", ".join(f"{k}:{v}" for k, v in sorted(s.get("by_reason", {}).items())),
        ]
        for s in [
          summary["human_calib"],
          summary["d_train"],
          summary["d_valid"],
          summary["d_all"],
          summary["adapter_samples"],
        ]
      ],
    ),
    "",
    "## D ラベル a の棄却",
    "",
    md_table(
      ["集合", "n", "棄却", "棄却率"],
      [
        [
          key.replace("d_label_a_", "d_"),
          stats["n"],
          stats.get("rejected_n", "—"),
          fmt_pct(stats.get("reject_rate")),
        ]
        for key, stats in summary.items()
        if key.startswith("d_label_a_")
      ],
    ),
    "",
    "## Acceptance (P3 G1)",
    "",
    f"- 人間推敲通過率 >= 0.99: {'✓' if (summary['human_calib']['pass_rate'] or 0) >= 0.99 else '✗'} ({fmt_pct(summary['human_calib']['pass_rate'])})",
    f"- D ラベル a 棄却 > 0: {'✓' if (summary['d_label_a_all'].get('rejected_n') or 0) > 0 else '✗'} ({summary['d_label_a_all'].get('rejected_n', 0)}/{summary['d_label_a_all'].get('n', 0)})",
  ]
  (reports / "ead-gate-defect.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-gate-defect.md"), "human_pass": summary["human_calib"]["pass_rate"]}, ensure_ascii=False))


def fmt_pct(x: float | None) -> str:
  from ead.common import fmt_pct as _fp

  return _fp(x)


if __name__ == "__main__":
  main()
