#!/usr/bin/env python3
"""アンカー付きペアを線形/非線形ヘッドで学習し、方向感度を比較する実験。

凍結した日本語埋め込みモデル（ruri）の特徴の上に、線形ヘッドと
多層パーセプトロン（MLP）ヘッドをそれぞれ Bradley-Terry 損失で学習し、
次の 4 つを測る。

- 検証ペア正解率（従来課題。下書き対人間編集）
- 順方向: margin = s(下書き, 人間編集) - s(下書き, 下書き) が正の率
- 逆方向: margin = s(人間編集, 下書き) - s(人間編集, 人間編集) が負の率
- 劣化版: 冗長な言い回しを挿入した版のマージンが負の率

埋め込みは outputs/anchor-exp/embeddings.joblib にキャッシュする。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump, load
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from torch import nn

from pref_static_utils import (
  assemble_pointwise_feature_vector,
  collect_unique_texts,
  encode_text_map,
  load_jsonl,
)
from train_pref_bt import unique_preference_pairs


def degrade(text: str) -> str:
  t = text.replace("。", "ということになります。", 3)
  t = t.replace("、", "、まあ、", 2)
  return t


def load_embeddings(
  texts: list[str],
  *,
  model_name: str,
  text_prefix: str,
  max_seq_length: int,
  batch_size: int,
  cache_path: Path,
) -> dict[str, np.ndarray]:
  cache: dict[str, np.ndarray] = {}
  if cache_path.is_file():
    cache = load(cache_path)
  missing = [t for t in texts if t not in cache]
  if missing:
    print(f"encoding {len(missing)} new texts (cached: {len(cache)})", flush=True)
    encoder = SentenceTransformer(model_name, device="cpu")
    if max_seq_length > 0:
      encoder.max_seq_length = max_seq_length
    new = encode_text_map(
      encoder,
      missing,
      batch_size=batch_size,
      normalize_embeddings=True,
      text_prefix=text_prefix,
    )
    cache.update(new)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dump(cache, cache_path)
  return cache


def feat(emb: dict[str, np.ndarray], source: str, candidate: str) -> np.ndarray:
  return assemble_pointwise_feature_vector(
    emb[source],
    emb[candidate],
    len_source=float(len(source)),
    len_candidate=float(len(candidate)),
  )


def build_matrix(rows: list[dict], emb: dict, key: str) -> np.ndarray:
  return np.vstack([feat(emb, r["source_text"], r[key]) for r in rows]).astype(np.float32)


class MLPHead(nn.Module):
  def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x).squeeze(-1)


def train_head(
  kind: str,
  x_w: np.ndarray,
  x_l: np.ndarray,
  *,
  hidden: int,
  dropout: float,
  epochs: int,
  lr: float,
  weight_decay: float,
  batch_size: int,
  seed: int,
) -> tuple[nn.Module, StandardScaler]:
  scaler = StandardScaler()
  scaler.fit(np.vstack([x_w, x_l]))
  w = torch.tensor(scaler.transform(x_w), dtype=torch.float32)
  l = torch.tensor(scaler.transform(x_l), dtype=torch.float32)

  torch.manual_seed(seed)
  if kind == "linear":
    head: nn.Module = nn.Linear(w.shape[1], 1)
    forward = lambda x: head(x).squeeze(-1)  # noqa: E731
  else:
    head = MLPHead(w.shape[1], hidden, dropout)
    forward = head

  optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
  n = w.shape[0]
  head.train()
  for epoch in range(epochs):
    perm = torch.randperm(n)
    total = 0.0
    for start in range(0, n, batch_size):
      idx = perm[start : start + batch_size]
      loss = torch.nn.functional.softplus(-(forward(w[idx]) - forward(l[idx]))).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total += float(loss.item()) * len(idx)
    if (epoch + 1) % 10 == 0:
      print(f"  [{kind}] epoch {epoch + 1}/{epochs} loss={total / n:.4f}", flush=True)
  head.eval()
  return head, scaler


@torch.no_grad()
def scores(head: nn.Module, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
  t = torch.tensor(scaler.transform(x), dtype=torch.float32)
  out = head(t)
  if out.dim() > 1:
    out = out.squeeze(-1)
  return out.numpy()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-file", default="data/pref_split_anchor/train.jsonl")
  parser.add_argument("--eval-file", default="data/pref_split/valid.jsonl")
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=512)
  parser.add_argument("--encode-batch-size", type=int, default=32)
  parser.add_argument("--hidden", type=int, default=256)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--epochs", type=int, default=60)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--n-degrade", type=int, default=60)
  parser.add_argument("--cache", default="outputs/anchor-exp/embeddings.joblib")
  parser.add_argument("--report", default="outputs/anchor-exp/report.json")
  args = parser.parse_args()

  train_rows = unique_preference_pairs(load_jsonl(args.train_file))
  eval_rows = unique_preference_pairs(load_jsonl(args.eval_file))
  print(f"train pairs: {len(train_rows)}, eval pairs: {len(eval_rows)}", flush=True)

  probes = []
  for i, r in enumerate(eval_rows):
    d, e = r["candidate_b"], r["candidate_a"]
    probe = {"draft": d, "edit": e}
    if i < args.n_degrade:
      bad = degrade(d)
      if bad != d:
        probe["bad"] = bad
    probes.append(probe)

  texts = collect_unique_texts(train_rows + eval_rows)
  seen = set(texts)
  for p in probes:
    for t in p.values():
      if t not in seen:
        seen.add(t)
        texts.append(t)
  emb = load_embeddings(
    texts,
    model_name=args.model,
    text_prefix=args.text_prefix,
    max_seq_length=args.max_seq_length,
    batch_size=args.encode_batch_size,
    cache_path=Path(args.cache),
  )

  x_train_w = build_matrix(train_rows, emb, "candidate_a")
  x_train_l = build_matrix(train_rows, emb, "candidate_b")
  x_eval_w = build_matrix(eval_rows, emb, "candidate_a")
  x_eval_l = build_matrix(eval_rows, emb, "candidate_b")

  report: dict = {"train_file": args.train_file, "n_train": len(train_rows)}
  for kind in ("linear", "mlp"):
    print(f"== {kind} head ==", flush=True)
    head, scaler = train_head(
      kind,
      x_train_w,
      x_train_l,
      hidden=args.hidden,
      dropout=args.dropout,
      epochs=args.epochs,
      lr=args.lr,
      weight_decay=args.weight_decay,
      batch_size=args.batch_size,
      seed=args.seed,
    )
    margin_eval = scores(head, scaler, x_eval_w) - scores(head, scaler, x_eval_l)

    def self_margin(base: str, cand: str) -> float:
      x = np.vstack([feat(emb, base, cand), feat(emb, base, base)]).astype(np.float32)
      s = scores(head, scaler, x)
      return float(s[0] - s[1])

    fwd = np.array([self_margin(p["draft"], p["edit"]) for p in probes])
    rev = np.array([self_margin(p["edit"], p["draft"]) for p in probes])
    deg = np.array([self_margin(p["draft"], p["bad"]) for p in probes if "bad" in p])

    result = {
      "eval_pair_accuracy": float((margin_eval > 0).mean()),
      "forward_positive_rate": float((fwd > 0).mean()),
      "forward_median": float(np.median(fwd)),
      "reverse_negative_rate": float((rev < 0).mean()),
      "reverse_median": float(np.median(rev)),
      "degrade_negative_rate": float((deg < 0).mean()),
      "degrade_median": float(np.median(deg)),
      "n_degrade": int(deg.size),
    }
    report[kind] = result
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

  out = Path(args.report)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"report: {out}", flush=True)


if __name__ == "__main__":
  main()
