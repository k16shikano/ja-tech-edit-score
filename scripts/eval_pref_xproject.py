#!/usr/bin/env python3
"""leave-one-project-out 交差検証。

書籍（project_id）を 1 つずつ held-out にして残りで学習し、
未知の書籍への汎化性能を測る。学習データを減らすためのものではなく、
モデル構成の比較にのみ使う（配布用モデルは全データで学習する）。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pref_static_utils import (
  build_feature_matrix,
  build_label_array,
  collect_unique_texts,
  encode_text_map,
  load_jsonl,
)


def project_of(row: dict) -> str:
  return str(row.get("meta", {}).get("project_id", "") or "(unknown)")


def fit_classifier(x_train: np.ndarray, y_train: np.ndarray, *, c: float, max_iter: int) -> Pipeline:
  classifier = Pipeline(
    steps=[
      ("scaler", StandardScaler()),
      (
        "classifier",
        LogisticRegression(
          C=c,
          max_iter=max_iter,
          solver="lbfgs",
          random_state=0,
        ),
      ),
    ]
  )
  classifier.fit(x_train, y_train)
  return classifier


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="preference jsonl (swap 拡張済み)")
  parser.add_argument("--model", default="hotchpotch/static-embedding-japanese", help="sentence embedding model")
  parser.add_argument(
    "--truncate-dim",
    type=int,
    default=256,
    help="embedding dimension truncation (0 以下で無効化)",
  )
  parser.add_argument(
    "--text-prefix",
    default="",
    help="埋め込み前に付けるプレフィックス（例: Ruri の「文章: 」、E5 の「passage: 」）",
  )
  parser.add_argument(
    "--max-seq-length",
    type=int,
    default=0,
    help="トークン長上限（0 でモデル既定。CPU では 512 推奨）",
  )
  parser.add_argument("--batch-size", type=int, default=128, help="embedding batch size")
  parser.add_argument("--max-iter", type=int, default=2000, help="logistic regression max_iter")
  parser.add_argument("--c", type=float, default=1.0, help="inverse regularization strength")
  parser.add_argument("--report", default="", help="JSON report output path")
  args = parser.parse_args()

  rows = load_jsonl(args.input)
  if not rows:
    raise SystemExit("input is empty")

  indices_by_project: dict[str, list[int]] = defaultdict(list)
  for i, row in enumerate(rows):
    indices_by_project[project_of(row)].append(i)
  projects = sorted(indices_by_project, key=lambda p: -len(indices_by_project[p]))
  if len(projects) < 2:
    raise SystemExit("need at least 2 projects for leave-one-project-out")

  truncate_dim = args.truncate_dim if args.truncate_dim > 0 else None
  model = SentenceTransformer(args.model, device="cpu", truncate_dim=truncate_dim)
  if args.max_seq_length > 0:
    model.max_seq_length = args.max_seq_length
  unique_texts = collect_unique_texts(rows)
  texts_to_encode = [args.text_prefix + t for t in unique_texts]
  encoded = encode_text_map(
    model,
    texts_to_encode,
    batch_size=args.batch_size,
    normalize_embeddings=True,
  )
  # 特徴行列はプレフィックスなし原文をキーにするため、対応を戻す
  text_to_embedding = {
    original: encoded[args.text_prefix + original]
    for original in unique_texts
  }

  # 特徴行列は全行ぶん一度だけ作り、fold ごとにスライスする
  features = build_feature_matrix(rows, text_to_embedding)
  labels = build_label_array(rows)

  folds = []
  for held_out in projects:
    eval_idx = np.asarray(indices_by_project[held_out], dtype=np.int64)
    train_mask = np.ones(len(rows), dtype=bool)
    train_mask[eval_idx] = False

    classifier = fit_classifier(features[train_mask], labels[train_mask], c=args.c, max_iter=args.max_iter)
    probs = classifier.predict_proba(features[eval_idx])
    preds = classifier.predict(features[eval_idx])
    y_eval = labels[eval_idx]

    fold = {
      "project_id": held_out,
      "eval_samples": int(eval_idx.size),
      "train_samples": int(train_mask.sum()),
      "accuracy": float(accuracy_score(y_eval, preds)),
      "log_loss": float(log_loss(y_eval, probs, labels=[0, 1])),
    }
    folds.append(fold)
    print(
      f"{held_out:28s} n={fold['eval_samples']:5d} "
      f"acc={fold['accuracy']:.4f} log_loss={fold['log_loss']:.4f}"
    )

  total = sum(f["eval_samples"] for f in folds)
  micro_accuracy = sum(f["accuracy"] * f["eval_samples"] for f in folds) / total
  macro_accuracy = float(np.mean([f["accuracy"] for f in folds]))
  micro_log_loss = sum(f["log_loss"] * f["eval_samples"] for f in folds) / total

  print("-" * 64)
  print(f"projects: {len(folds)}  rows: {total}")
  print(f"micro accuracy (行数加重): {micro_accuracy:.4f}")
  print(f"macro accuracy (書籍平均): {macro_accuracy:.4f}")
  print(f"micro log_loss: {micro_log_loss:.4f}")

  if args.report:
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
      "embedding_model": args.model,
      "truncate_dim": truncate_dim,
      "text_prefix": args.text_prefix,
      "max_seq_length": args.max_seq_length if args.max_seq_length > 0 else None,
      "c": args.c,
      "input": args.input,
      "total_rows": total,
      "micro_accuracy": micro_accuracy,
      "macro_accuracy": macro_accuracy,
      "micro_log_loss": micro_log_loss,
      "folds": folds,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {report_path}")


if __name__ == "__main__":
  main()
