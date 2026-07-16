#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from joblib import dump
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pref_static_utils import (
  build_feature_matrix,
  build_label_array,
  collect_unique_texts,
  encode_text_map,
  load_jsonl,
  normalize_truncate_dim,
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m", help="sentence embedding model")
  parser.add_argument("--train-file", required=True, help="train jsonl")
  parser.add_argument("--eval-file", required=True, help="valid jsonl")
  parser.add_argument("--output-dir", required=True, help="output directory")
  parser.add_argument("--max-train-samples", type=int, default=-1)
  parser.add_argument("--max-eval-samples", type=int, default=-1)
  parser.add_argument(
    "--truncate-dim",
    type=int,
    default=0,
    help="embedding dimension truncation (0 以下で無効)",
  )
  parser.add_argument(
    "--text-prefix",
    default="文章: ",
    help="埋め込み前プレフィックス（Ruri は「文章: 」）",
  )
  parser.add_argument(
    "--max-seq-length",
    type=int,
    default=512,
    help="トークン長上限（0 でモデル既定）",
  )
  parser.add_argument("--batch-size", type=int, default=32, help="embedding batch size")
  parser.add_argument("--max-iter", type=int, default=2000, help="logistic regression max_iter")
  parser.add_argument("--c", type=float, default=1.0, help="inverse regularization strength")
  args = parser.parse_args()

  train_rows = load_jsonl(args.train_file, limit=args.max_train_samples)
  eval_rows = load_jsonl(args.eval_file, limit=args.max_eval_samples)

  if not train_rows:
    raise SystemExit("train set is empty")
  if not eval_rows:
    raise SystemExit("eval set is empty")

  truncate_dim = normalize_truncate_dim(args.truncate_dim)
  model = SentenceTransformer(args.model, device="cpu", truncate_dim=truncate_dim)
  if args.max_seq_length > 0:
    model.max_seq_length = args.max_seq_length
  normalize_embeddings = True

  all_rows = train_rows + eval_rows
  unique_texts = collect_unique_texts(all_rows)
  text_to_embedding = encode_text_map(
    model,
    unique_texts,
    batch_size=args.batch_size,
    normalize_embeddings=normalize_embeddings,
    text_prefix=args.text_prefix,
  )

  x_train = build_feature_matrix(train_rows, text_to_embedding)
  y_train = build_label_array(train_rows)
  x_eval = build_feature_matrix(eval_rows, text_to_embedding)
  y_eval = build_label_array(eval_rows)

  classifier = Pipeline(
    steps=[
      ("scaler", StandardScaler()),
      (
        "classifier",
        LogisticRegression(
          C=args.c,
          max_iter=args.max_iter,
          solver="lbfgs",
          random_state=0,
        ),
      ),
    ]
  )
  classifier.fit(x_train, y_train)

  eval_probs = classifier.predict_proba(x_eval)
  eval_preds = classifier.predict(x_eval)
  eval_accuracy = float(accuracy_score(y_eval, eval_preds))
  eval_log_loss = float(log_loss(y_eval, eval_probs))
  report = classification_report(y_eval, eval_preds, output_dict=True, zero_division=0)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  artifact = {
    "classifier": classifier,
    "sentence_model_name": args.model,
    "truncate_dim": truncate_dim,
    "text_prefix": args.text_prefix,
    "max_seq_length": args.max_seq_length if args.max_seq_length > 0 else None,
    "normalize_embeddings": normalize_embeddings,
    "feature_version": "source-a-b-diff-sim-v1",
  }
  dump(artifact, output_dir / "model.joblib")

  metrics = {
    "eval_accuracy": eval_accuracy,
    "eval_log_loss": eval_log_loss,
    "train_samples": len(train_rows),
    "eval_samples": len(eval_rows),
    "truncate_dim": truncate_dim,
    "text_prefix": args.text_prefix,
    "max_seq_length": artifact["max_seq_length"],
    "embedding_model": args.model,
    "feature_version": artifact["feature_version"],
    "classification_report": report,
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )

  print(f"saved: {output_dir}")
  print(f"eval_accuracy: {eval_accuracy}")
  print(f"eval_log_loss: {eval_log_loss}")


if __name__ == "__main__":
  main()
