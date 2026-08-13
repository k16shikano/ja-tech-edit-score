#!/usr/bin/env python3
"""段落遷移分類器を学習する（凍結 ruri 埋め込み + ロジスティック回帰）。

検証は leave-one-project-out。成果物は全データで再学習したモデル。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import dump
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pref_static_utils import encode_texts, load_jsonl, normalize_truncate_dim


def collect_unique_paragraph_texts(rows: list[dict]) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for row in rows:
    for key in ("text_a", "text_b"):
      value = str(row[key])
      if value not in seen:
        seen.add(value)
        texts.append(value)
  return texts


def assemble_transition_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
  product = emb_a * emb_b
  abs_diff = np.abs(emb_a - emb_b)
  return np.concatenate([emb_a, emb_b, product, abs_diff]).astype(np.float32, copy=False)


def build_feature_matrix(rows: list[dict], text_to_embedding: dict[str, np.ndarray]) -> np.ndarray:
  features = [
    assemble_transition_features(text_to_embedding[row["text_a"]], text_to_embedding[row["text_b"]])
    for row in rows
  ]
  return np.vstack(features)


def build_label_array(rows: list[dict]) -> np.ndarray:
  return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def project_of(row: dict) -> str:
  return str(row.get("project_id") or row.get("meta", {}).get("project_id") or "(unknown)")


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


def eval_fold(classifier: Pipeline, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
  preds = classifier.predict(x_eval)
  probs = classifier.predict_proba(x_eval)
  classes = list(classifier.named_steps["classifier"].classes_)
  pos_idx = classes.index(1) if 1 in classes else 0
  pos_probs = probs[:, pos_idx]
  metrics = {
    "accuracy": float(accuracy_score(y_eval, preds)),
  }
  if len(np.unique(y_eval)) >= 2:
    metrics["auc"] = float(roc_auc_score(y_eval, pos_probs))
  else:
    metrics["auc"] = float("nan")
  return metrics


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", default="data/paragraph_transitions.jsonl")
  parser.add_argument("--output-dir", default="outputs/paragraph-transition")
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--max-iter", type=int, default=2000)
  parser.add_argument("--c", type=float, default=1.0)
  args = parser.parse_args()

  root = Path(__file__).resolve().parent.parent
  input_path = Path(args.input)
  if not input_path.is_absolute():
    input_path = root / input_path
  output_dir = Path(args.output_dir)
  if not output_dir.is_absolute():
    output_dir = root / output_dir

  rows = load_jsonl(str(input_path))
  if len(rows) < 2:
    raise SystemExit("need transition pairs")

  indices_by_project: dict[str, list[int]] = defaultdict(list)
  for i, row in enumerate(rows):
    indices_by_project[project_of(row)].append(i)
  projects = sorted(indices_by_project, key=lambda p: -len(indices_by_project[p]))
  if len(projects) < 2:
    raise SystemExit("need at least 2 projects for leave-one-project-out")

  truncate_dim = normalize_truncate_dim(args.truncate_dim)
  encoder = SentenceTransformer(args.model, device="cpu", truncate_dim=truncate_dim)
  if args.max_seq_length > 0:
    encoder.max_seq_length = args.max_seq_length

  unique_texts = collect_unique_paragraph_texts(rows)
  print(f"unique paragraphs: {len(unique_texts)}")
  embeddings = encode_texts(
    encoder,
    unique_texts,
    batch_size=args.batch_size,
    normalize_embeddings=True,
    text_prefix=args.text_prefix,
    show_progress_bar=True,
  )
  text_to_embedding = {text: emb for text, emb in zip(unique_texts, embeddings, strict=True)}

  features = build_feature_matrix(rows, text_to_embedding)
  labels = build_label_array(rows)

  folds = []
  for held_out in projects:
    eval_idx = np.asarray(indices_by_project[held_out], dtype=np.int64)
    train_mask = np.ones(len(rows), dtype=bool)
    train_mask[eval_idx] = False

    classifier = fit_classifier(features[train_mask], labels[train_mask], c=args.c, max_iter=args.max_iter)
    metrics = eval_fold(classifier, features[eval_idx], labels[eval_idx])
    fold = {
      "project_id": held_out,
      "eval_samples": int(eval_idx.size),
      "train_samples": int(train_mask.sum()),
      **metrics,
    }
    folds.append(fold)
    print(
      f"{held_out:28s} n={fold['eval_samples']:5d} "
      f"acc={fold['accuracy']:.4f} auc={fold['auc']:.4f}"
    )

  total = sum(f["eval_samples"] for f in folds)
  micro_acc = sum(f["accuracy"] * f["eval_samples"] for f in folds) / total
  macro_acc = float(np.mean([f["accuracy"] for f in folds]))
  auc_values = [f["auc"] for f in folds if not np.isnan(f["auc"])]
  micro_auc = (
    sum(f["auc"] * f["eval_samples"] for f in folds if not np.isnan(f["auc"]))
    / max(1, sum(f["eval_samples"] for f in folds if not np.isnan(f["auc"])))
  )
  macro_auc = float(np.mean(auc_values)) if auc_values else float("nan")

  print("-" * 64)
  print(f"projects: {len(folds)}  rows: {total}")
  print(f"micro accuracy: {micro_acc:.4f}")
  print(f"macro accuracy: {macro_acc:.4f}")
  print(f"micro auc: {micro_auc:.4f}")
  print(f"macro auc: {macro_auc:.4f}")

  final_classifier = fit_classifier(features, labels, c=args.c, max_iter=args.max_iter)
  output_dir.mkdir(parents=True, exist_ok=True)
  artifact = {
    "sentence_model_name": args.model,
    "truncate_dim": truncate_dim,
    "text_prefix": args.text_prefix,
    "max_seq_length": args.max_seq_length if args.max_seq_length > 0 else None,
    "classifier": final_classifier,
  }
  model_path = output_dir / "model.joblib"
  dump(artifact, model_path)

  metrics = {
    "embedding_model": args.model,
    "text_prefix": args.text_prefix,
    "max_seq_length": args.max_seq_length if args.max_seq_length > 0 else None,
    "input": str(input_path),
    "total_rows": total,
    "positive_rows": int((labels == 1).sum()),
    "negative_rows": int((labels == 0).sum()),
    "micro_accuracy": micro_acc,
    "macro_accuracy": macro_acc,
    "micro_auc": micro_auc,
    "macro_auc": macro_auc,
    "folds": folds,
  }
  metrics_path = output_dir / "metrics.json"
  metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"model: {model_path}")
  print(f"metrics: {metrics_path}")


if __name__ == "__main__":
  main()
