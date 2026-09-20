#!/usr/bin/env python3
"""タスク 0-3: 転移非対称性。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  FEATURE_NAMES,
  ead_reports,
  fmt_pct,
  load_jsonl,
  md_table,
  repo_root,
  surface_features,
  write_json,
)

try:
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import cross_val_score
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
except ImportError as exc:
  raise SystemExit(f"sklearn required: {exc}") from exc


def load_d_rows(root: Path) -> list[dict]:
  out: list[dict] = []
  for path in (root / "data/d/train.jsonl", root / "data/d/valid.jsonl"):
    for row in load_jsonl(path):
      if row.get("y_kind") == "human":
        label = 1
      elif str(row.get("y_kind", "")).startswith("qwen"):
        label = 0
      else:
        continue
      out.append({"draft": str(row["draft"]), "text": str(row["y"]), "label": label})
  return out


def load_adapter_rows(root: Path) -> list[dict]:
  path = root / "outputs/edit-sft-eval-v3/adapter_samples.jsonl"
  if not path.is_file():
    return []
  out: list[dict] = []
  for row in load_jsonl(path):
    draft = str(row["draft"])
    out.append({"draft": draft, "text": str(row["generated"]), "label": 0})
    out.append({"draft": draft, "text": str(row["gold"]), "label": 1})
  return out


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
  xs = [[surface_features(r["draft"], r["text"])[name] for name in FEATURE_NAMES] for r in rows]
  ys = [int(r["label"]) for r in rows]
  return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def cv_auc(X: np.ndarray, y: np.ndarray) -> float | None:
  if len(y) < 10 or len(set(y.tolist())) < 2:
    return None
  pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
  )
  folds = min(5, len(set(y.tolist())) * 2, len(y))
  try:
    scores = cross_val_score(pipe, X, y, cv=max(2, folds), scoring="roc_auc")
    return float(np.mean(scores))
  except ValueError:
    return None


def length_summary(rows: list[dict]) -> dict:
  draft_lens = [len(r["draft"]) for r in rows]
  cand_lens = [len(r["text"]) for r in rows]
  ratios = [c / max(d, 1) for d, c in zip(draft_lens, cand_lens)]
  return {
    "n": len(rows),
    "draft_median": float(np.median(draft_lens)) if draft_lens else None,
    "cand_median": float(np.median(cand_lens)) if cand_lens else None,
    "ratio_median": float(np.median(ratios)) if ratios else None,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  args = parser.parse_args()
  root = args.root.resolve()

  d_rows = load_d_rows(root)
  adapter_rows = load_adapter_rows(root)
  X_d, y_d = to_xy(d_rows)
  X_a, y_a = to_xy(adapter_rows)
  auc_d = cv_auc(X_d, y_d)
  auc_a = cv_auc(X_a, y_a)

  summary = {
    "d_rows": len(d_rows),
    "adapter_rows": len(adapter_rows),
    "d_auc": auc_d,
    "adapter_auc": auc_a,
    "d_length": length_summary(d_rows),
    "adapter_length": length_summary(adapter_rows),
    "asymmetric": bool(auc_d is not None and auc_a is not None and auc_d - auc_a > 0.05),
  }

  reports = ead_reports()
  write_json(reports / "ead-transfer-diag.json", summary)
  lines = [
    "# ead-transfer-diag",
    "",
    md_table(
      ["集合", "件数", "AUC", "下書き median", "候補 median", "文字数比 median"],
      [
        [
          "D",
          str(summary["d_rows"]),
          fmt_pct(auc_d),
          fmt_pct(summary["d_length"]["draft_median"]),
          fmt_pct(summary["d_length"]["cand_median"]),
          fmt_pct(summary["d_length"]["ratio_median"]),
        ],
        [
          "adapter_samples",
          str(summary["adapter_rows"]),
          fmt_pct(auc_a),
          fmt_pct(summary["adapter_length"]["draft_median"]),
          fmt_pct(summary["adapter_length"]["cand_median"]),
          fmt_pct(summary["adapter_length"]["ratio_median"]),
        ],
      ],
    ),
    "",
    f"転移非対称: {'確認' if summary['asymmetric'] else '未確認'}",
  ]
  (reports / "ead-transfer-diag.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-transfer-diag.md"), **summary}, ensure_ascii=False))


if __name__ == "__main__":
  main()
