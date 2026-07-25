#!/usr/bin/env python3
"""機械負例の線形分離可能性を診断する。

pref-bt と同じ特徴ベクトル（ruri 埋め込み・差分・コサイン類似度・長さ）を使い、
「人間編集 vs 機械推敲案」だけを目的にした線形分類器を、
プロジェクト単位の交差検証（Leave-One-Project-Out に相当する GroupKFold）で学習・評価する。

これはこのアーキテクチャの理論上の上限に相当する。
- 交差検証 AUC が高い: 表現力はある。混合学習の配分や量の問題。
- 交差検証 AUC が低い: ruri 埋め込み＋線形層ではこの区別を表現できない。

参考として「人間編集 vs 下書きコピー」の同じ診断も出す（こちらは高いはず）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from pref_static_utils import (
  assemble_pointwise_feature_vector,
  encode_texts,
  load_jsonl,
  load_sentence_model_from_artifact,
)


def build_items(pref_rows: list[dict]) -> list[dict]:
  items = []
  for row in pref_rows:
    if row.get("meta", {}).get("pair_order") != "chosen_first":
      continue
    items.append(
      {
        "id": row["id"],
        "project_id": str(row.get("meta", {}).get("project_id") or ""),
        "source": str(row["source_text"]),
        "human": str(row["candidate_a"]),
        "machine": str(row["candidate_b"]),
      }
    )
  return items


def feature_diffs(
  items: list[dict],
  emb: dict[str, np.ndarray],
  pos_key: str,
  neg_key: str,
) -> np.ndarray:
  diffs = []
  for it in items:
    f_pos = assemble_pointwise_feature_vector(
      emb[it["source"]],
      emb[it[pos_key]],
      len_source=float(len(it["source"])),
      len_candidate=float(len(it[pos_key])),
    )
    f_neg = assemble_pointwise_feature_vector(
      emb[it["source"]],
      emb[it[neg_key]],
      len_source=float(len(it["source"])),
      len_candidate=float(len(it[neg_key])),
    )
    diffs.append(f_pos - f_neg)
  return np.vstack(diffs)


def grouped_cv_auc(
  diffs: np.ndarray,
  groups: list[str],
  n_splits: int,
) -> dict:
  # BT と同じ形: x = f(chosen) - f(rejected) を正、-x を負として切片なしで学習する。
  X = np.concatenate([diffs, -diffs], axis=0)
  y = np.concatenate([np.ones(len(diffs)), np.zeros(len(diffs))])
  g = np.array(groups + groups)

  aucs = []
  accs = []
  fold_sizes = []
  gkf = GroupKFold(n_splits=n_splits)
  for train_idx, test_idx in gkf.split(X, y, groups=g):
    scaler = StandardScaler().fit(X[train_idx])
    clf = LogisticRegression(max_iter=2000, fit_intercept=False, C=1.0)
    clf.fit(scaler.transform(X[train_idx]), y[train_idx])
    prob = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    aucs.append(float(roc_auc_score(y[test_idx], prob)))
    accs.append(float(((prob > 0.5) == (y[test_idx] > 0.5)).mean()))
    fold_sizes.append(int(len(test_idx) // 2))
  return {
    "n_items": len(diffs),
    "n_splits": n_splits,
    "fold_items": fold_sizes,
    "auc_per_fold": [round(a, 4) for a in aucs],
    "auc_mean": round(float(np.mean(aucs)), 4),
    "accuracy_per_fold": [round(a, 4) for a in accs],
    "accuracy_mean": round(float(np.mean(accs)), 4),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--pref",
    default="data/pref_dataset_machine_neg.jsonl",
    help="build_machine_neg_pref.py の出力",
  )
  parser.add_argument(
    "--model-dir",
    default="outputs/pref-bt-machine-neg",
    help="埋め込み設定を読む pref-bt モデルディレクトリ",
  )
  parser.add_argument("--splits", type=int, default=5)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--out", default="outputs/machine_neg_separability.json")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  rows = load_jsonl(str((root / args.pref).resolve()))
  items = build_items(rows)
  projects = sorted({it["project_id"] for it in items})
  print(f"items: {len(items)}  projects: {projects}")

  artifact = load((root / args.model_dir / "model.joblib").resolve())
  model = load_sentence_model_from_artifact(artifact, device="cpu")
  normalize = bool(artifact["normalize_embeddings"])
  prefix = str(artifact.get("text_prefix", ""))

  unique_texts = sorted(
    {t for it in items for t in (it["source"], it["human"], it["machine"])}
  )
  print(f"encoding {len(unique_texts)} unique texts ...")
  vecs = encode_texts(
    model,
    unique_texts,
    batch_size=args.batch_size,
    normalize_embeddings=normalize,
    text_prefix=prefix,
    show_progress_bar=True,
  )
  emb = {t: v for t, v in zip(unique_texts, vecs, strict=True)}

  groups = [it["project_id"] for it in items]
  n_splits = min(args.splits, len(set(groups)))

  result = {
    "pref": str(args.pref),
    "model_dir": str(args.model_dir),
    "embedding_model": artifact.get("sentence_model_name"),
    "human_vs_machine": grouped_cv_auc(
      feature_diffs(items, emb, "human", "machine"), groups, n_splits
    ),
    "human_vs_copy": grouped_cv_auc(
      feature_diffs(items, emb, "human", "source"), groups, n_splits
    ),
  }

  for key in ("human_vs_machine", "human_vs_copy"):
    r = result[key]
    print(
      f"{key}: auc_mean={r['auc_mean']} acc_mean={r['accuracy_mean']} "
      f"auc_per_fold={r['auc_per_fold']}"
    )

  out_path = (root / args.out).resolve()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {out_path}")


if __name__ == "__main__":
  main()
