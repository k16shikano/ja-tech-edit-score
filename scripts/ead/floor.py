#!/usr/bin/env python3
"""タスク 0-4: 表層特徴床。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  FEATURE_NAMES,
  POSITION_RANK,
  ead_reports,
  fmt_pct,
  fmt_rate,
  load_jsonl,
  md_table,
  repo_root,
  rps_ordinal,
  rps_sanity_baselines,
  surface_features,
  write_json,
)

try:
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
  from sklearn.preprocessing import StandardScaler
except ImportError as exc:
  raise SystemExit(f"sklearn required: {exc}") from exc


def rows_to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
  xs: list[list[float]] = []
  ys: list[int] = []
  drafts: list[str] = []
  for row in rows:
    draft = str(row["draft"])
    cand = str(row["y"])
    feats = surface_features(draft, cand)
    xs.append([feats[name] for name in FEATURE_NAMES])
    ys.append(POSITION_RANK[str(row["position"])])
    drafts.append(draft)
  return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64), drafts


def cumulative_probs(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
  sig = 1.0 / (1.0 + np.exp(-scores))
  probs = np.stack([sig(t - scores) for t in thresholds], axis=1)
  probs = np.clip(probs, 1e-6, 1 - 1e-6)
  p0 = probs[:, 0:1]
  rest = np.diff(probs, axis=1, prepend=p0)
  rest = np.clip(rest, 1e-6, 1.0)
  rest = rest / rest.sum(axis=1, keepdims=True)
  return rest


def fit_ordinal(X_train: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegression, StandardScaler]:
  scaler = StandardScaler()
  Xs = scaler.fit_transform(X_train)
  clf = LogisticRegression(max_iter=2000, solver="lbfgs")
  clf.fit(Xs, y_train)
  return clf, scaler


def predict_classes(clf: LogisticRegression, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
  return clf.predict(scaler.transform(X))


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  return float(np.mean(np.abs(y_true - y_pred) <= 1))


def decile_buckets(deltas: list[float]) -> list[int]:
  if not deltas:
    return []
  arr = np.asarray(deltas, dtype=np.float64)
  edges = np.quantile(arr, np.linspace(0, 1, 11))
  edges = np.unique(edges)
  if len(edges) <= 2:
    return [0] * len(deltas)
  return list(np.digitize(arr, edges[1:-1], right=True))


def univariate_auc(X: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
  out: dict[str, float | None] = {}
  binary = (y >= 2).astype(int)
  for i, name in enumerate(FEATURE_NAMES):
    col = X[:, i]
    if len(set(col)) <= 1 or len(set(binary)) <= 1:
      out[name] = None
      continue
    try:
      out[name] = float(roc_auc_score(binary, col))
    except ValueError:
      out[name] = None
  return out


def eval_split(name: str, clf, scaler, X, y, drafts) -> dict:
  pred = predict_classes(clf, scaler, X)
  prob = clf.predict_proba(scaler.transform(X))
  prec, rec, f1, _ = precision_recall_fscore_support(
    y == 0, pred == 0, labels=[True], average="binary", zero_division=0
  )
  deltas = X[:, FEATURE_NAMES.index("delta_chars")].tolist()
  deciles = decile_buckets(deltas)
  strata: dict[str, dict] = {}
  for d in sorted(set(deciles)):
    mask = np.array(deciles) == d
    if mask.sum() == 0:
      continue
    strata[str(d)] = {
      "n": int(mask.sum()),
      "accuracy": float(accuracy_score(y[mask], pred[mask])),
    }
  return {
    "split": name,
    "n": int(len(y)),
    "rps": rps_ordinal(y, prob),
    "accuracy_4": float(accuracy_score(y, pred)),
    "adjacent_accuracy": adjacent_accuracy(y, pred),
    "label_a_precision": float(prec),
    "label_a_recall": float(rec),
    "label_a_f1": float(f1),
    "strata_delta_decile": strata,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--train", type=Path, default=Path("data/d/train.jsonl"))
  parser.add_argument("--valid", type=Path, default=Path("data/d/valid.jsonl"))
  args = parser.parse_args()

  root = args.root.resolve()
  train_rows = load_jsonl(root / args.train)
  valid_rows = load_jsonl(root / args.valid)
  X_train, y_train, _ = rows_to_xy(train_rows)
  X_valid, y_valid, drafts_valid = rows_to_xy(valid_rows)

  clf, scaler = fit_ordinal(X_train, y_train)
  delta_idx = FEATURE_NAMES.index("delta_chars")
  clf1 = LogisticRegression(max_iter=2000, solver="lbfgs")
  scaler1 = StandardScaler()
  X1_train = scaler1.fit_transform(X_train[:, [delta_idx]])
  clf1.fit(X1_train, y_train)

  valid_full = eval_split("valid", clf, scaler, X_valid, y_valid, drafts_valid)
  pred1 = clf1.predict(scaler1.transform(X_valid[:, [delta_idx]]))
  delta_only = {
    "accuracy_4": float(accuracy_score(y_valid, pred1)),
    "adjacent_accuracy": adjacent_accuracy(y_valid, pred1),
    "label_a_recall": float(
      precision_recall_fscore_support(
        y_valid == 0, pred1 == 0, labels=[True], average="binary", zero_division=0
      )[1]
    ),
  }

  rps_sanity = rps_sanity_baselines(y_valid, y_train)

  summary = {
    "train_n": len(train_rows),
    "valid_n": len(valid_rows),
    "valid": valid_full,
    "delta_chars_only_valid": delta_only,
    "univariate_auc_valid": univariate_auc(X_valid, y_valid),
    "rps_sanity_valid": rps_sanity,
    "features": FEATURE_NAMES,
  }

  reports = ead_reports()
  write_json(reports / "ead-floor.json", summary)

  lines = [
    "# ead-floor",
    "",
    "表層特徴のみの多クラスロジスティック回帰（D train → valid）。",
    "",
    md_table(
      ["指標", "valid"],
      [
        ["RPS", fmt_pct(valid_full["rps"])],
        ["4値精度", fmt_pct(valid_full["accuracy_4"])],
        ["隣接許容精度", fmt_pct(valid_full["adjacent_accuracy"])],
        ["ラベル a 再現率", fmt_pct(valid_full["label_a_recall"])],
        ["ラベル a 適合率", fmt_pct(valid_full["label_a_precision"])],
      ],
    ),
    "",
    "## RPS 健全性（D valid）",
    "",
    "一様予測器と train 周辺分布予測器。床の RPS がこれより大幅に低い場合は計算バグの疑い。",
    "",
    md_table(
      ["予測器", "RPS"],
      [
        ["一様（各 0.25）", fmt_pct(rps_sanity["uniform"])],
        ["周辺分布（train 比率を全事例に）", fmt_pct(rps_sanity["marginal_train"])],
        ["表層特徴モデル（本床）", fmt_pct(valid_full["rps"])],
      ],
    ),
    "",
    "## Δ文字数のみ",
    "",
    md_table(
      ["指標", "valid"],
      [
        ["4値精度", fmt_pct(delta_only["accuracy_4"])],
        ["隣接許容精度", fmt_pct(delta_only["adjacent_accuracy"])],
        ["ラベル a 再現率", fmt_pct(delta_only["label_a_recall"])],
      ],
    ),
    "",
    "## 単変量 AUC（a/b vs c/d）",
    "",
    md_table(
      ["特徴", "AUC"],
      [[name, fmt_pct(summary["univariate_auc_valid"].get(name))] for name in FEATURE_NAMES],
    ),
  ]
  (reports / "ead-floor.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(
    json.dumps(
      {
        "wrote": str(reports / "ead-floor.md"),
        "rps_valid": valid_full["rps"],
        "rps_sanity": rps_sanity,
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
