#!/usr/bin/env python3
"""凍結埋め込み + 線形ヘッドの Bradley-Terry 報酬モデル。

候補 1 件にスカラー s(source, candidate) を出し、
P(A ≻ B) = σ(s(A) - s(B)) で学習する。絶対スコア・閾値ループ向け。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from torch import nn

from pref_static_utils import (
  assemble_pointwise_feature_vector,
  collect_unique_texts,
  encode_text_map,
  load_jsonl,
  normalize_truncate_dim,
)


def unique_preference_pairs(rows: list[dict]) -> list[dict]:
  """swap 拡張を除き、chosen/rejected が一度だけ出る行に絞る。"""
  selected: list[dict] = []
  for row in rows:
    meta = row.get("meta", {})
    if meta.get("pair_order", "chosen_first") != "chosen_first":
      continue
    if int(row["label"]) != 1:
      continue
    selected.append(row)
  return selected


def build_pointwise_matrix(
  rows: list[dict],
  text_to_embedding: dict[str, np.ndarray],
  *,
  key: str,
) -> np.ndarray:
  features = [
    assemble_pointwise_feature_vector(
      text_to_embedding[row["source_text"]],
      text_to_embedding[row[key]],
      len_source=float(len(row["source_text"])),
      len_candidate=float(len(row[key])),
    )
    for row in rows
  ]
  return np.vstack(features)


class LinearRewardHead(nn.Module):
  def __init__(self, in_dim: int) -> None:
    super().__init__()
    self.linear = nn.Linear(in_dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x).squeeze(-1)


def train_bt_head(
  x_chosen: np.ndarray,
  x_rejected: np.ndarray,
  *,
  epochs: int,
  lr: float,
  weight_decay: float,
  seed: int,
) -> tuple[LinearRewardHead, StandardScaler, dict[str, float]]:
  scaler = StandardScaler()
  x_all = np.vstack([x_chosen, x_rejected])
  scaler.fit(x_all)
  chosen = torch.tensor(scaler.transform(x_chosen), dtype=torch.float32)
  rejected = torch.tensor(scaler.transform(x_rejected), dtype=torch.float32)

  torch.manual_seed(seed)
  head = LinearRewardHead(chosen.shape[1])
  optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

  head.train()
  last_loss = 0.0
  for _ in range(epochs):
    optimizer.zero_grad()
    s_w = head(chosen)
    s_l = head(rejected)
    loss = torch.nn.functional.softplus(-(s_w - s_l)).mean()
    loss.backward()
    optimizer.step()
    last_loss = float(loss.item())

  head.eval()
  with torch.no_grad():
    s_w = head(chosen)
    s_l = head(rejected)
    train_acc = float((s_w > s_l).float().mean().item())
  return head, scaler, {"train_bt_loss": last_loss, "train_pair_accuracy": train_acc}


def eval_pair_accuracy(
  head: LinearRewardHead,
  scaler: StandardScaler,
  x_chosen: np.ndarray,
  x_rejected: np.ndarray,
) -> dict[str, float]:
  chosen = torch.tensor(scaler.transform(x_chosen), dtype=torch.float32)
  rejected = torch.tensor(scaler.transform(x_rejected), dtype=torch.float32)
  head.eval()
  with torch.no_grad():
    s_w = head(chosen)
    s_l = head(rejected)
    margin = s_w - s_l
    acc = float((margin > 0).float().mean().item())
    loss = float(torch.nn.functional.softplus(-margin).mean().item())
    mean_margin = float(margin.mean().item())
  return {
    "pair_accuracy": acc,
    "bt_loss": loss,
    "mean_margin": mean_margin,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--train-file", required=True, help="preference jsonl (swap 込み可)")
  parser.add_argument("--eval-file", required=True, help="preference jsonl")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--lr", type=float, default=1e-2)
  parser.add_argument("--weight-decay", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  train_rows = unique_preference_pairs(load_jsonl(args.train_file))
  eval_rows = unique_preference_pairs(load_jsonl(args.eval_file))
  if not train_rows or not eval_rows:
    raise SystemExit("train/eval preference pairs are empty after filtering swaps")

  truncate_dim = normalize_truncate_dim(args.truncate_dim)
  encoder = SentenceTransformer(args.model, device="cpu", truncate_dim=truncate_dim)
  if args.max_seq_length > 0:
    encoder.max_seq_length = args.max_seq_length

  all_rows = train_rows + eval_rows
  text_to_embedding = encode_text_map(
    encoder,
    collect_unique_texts(all_rows),
    batch_size=args.batch_size,
    normalize_embeddings=True,
    text_prefix=args.text_prefix,
  )

  x_train_w = build_pointwise_matrix(train_rows, text_to_embedding, key="candidate_a")
  x_train_l = build_pointwise_matrix(train_rows, text_to_embedding, key="candidate_b")
  x_eval_w = build_pointwise_matrix(eval_rows, text_to_embedding, key="candidate_a")
  x_eval_l = build_pointwise_matrix(eval_rows, text_to_embedding, key="candidate_b")

  head, scaler, train_metrics = train_bt_head(
    x_train_w,
    x_train_l,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
  )
  eval_metrics = eval_pair_accuracy(head, scaler, x_eval_w, x_eval_l)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  artifact = {
    "kind": "pref-bt",
    "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
    "scaler": scaler,
    "input_dim": int(x_train_w.shape[1]),
    "sentence_model_name": args.model,
    "truncate_dim": truncate_dim,
    "text_prefix": args.text_prefix,
    "max_seq_length": args.max_seq_length if args.max_seq_length > 0 else None,
    "normalize_embeddings": True,
    "feature_version": "source-cand-pointwise-v1",
  }
  dump(artifact, output_dir / "model.joblib")

  metrics = {
    "embedding_model": args.model,
    "text_prefix": args.text_prefix,
    "max_seq_length": artifact["max_seq_length"],
    "train_pairs": len(train_rows),
    "eval_pairs": len(eval_rows),
    **{f"train_{k}": v for k, v in train_metrics.items()},
    **{f"eval_{k}": v for k, v in eval_metrics.items()},
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"saved: {output_dir}")
  print(f"eval_pair_accuracy: {eval_metrics['pair_accuracy']:.4f}")
  print(f"eval_bt_loss: {eval_metrics['bt_loss']:.4f}")
  print(f"eval_mean_margin: {eval_metrics['mean_margin']:.4f}")


if __name__ == "__main__":
  main()
