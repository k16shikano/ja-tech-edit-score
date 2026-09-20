#!/usr/bin/env python3
"""凍結 ruri 文埋め込み + 文列 Transformer + 線形ヘッドの Bradley-Terry 報酬モデル。

文書を文単位に分割し、各文を ruri-v3-30m（凍結）でベクトル化する。
先頭に [DOC]、段落先頭文に段落開始埋め込みを載せた系列を小さい Transformer に読ませ、
[DOC] 位置の出力と pref-bt 同型の点特徴から s(source, candidate) を出す。
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from pref_static_utils import (
  assemble_pointwise_feature_vector,
  encode_text_map,
  load_jsonl,
  normalize_truncate_dim,
)
from sentseq_utils import (
  PARA_BOUNDARY_MODE,
  SENTSEQ_SPLIT_VERSION,
  SentenceUnit,
  split_document_sentences,
  truncate_sentence_units,
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


def collect_unique_texts(rows: list[dict]) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for row in rows:
    for key in ("source_text", "candidate_a", "candidate_b"):
      value = str(row[key])
      if value not in seen:
        seen.add(value)
        texts.append(value)
  return texts


class LinearRewardHead(nn.Module):
  def __init__(self, in_dim: int) -> None:
    super().__init__()
    self.linear = nn.Linear(in_dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x).squeeze(-1)


class SentSeqEncoder(nn.Module):
  """[DOC] + 文ベクトル列を Transformer で読み、文書ベクトルを返す。"""

  def __init__(
    self,
    *,
    embed_dim: int,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    max_sents: int = 128,
  ) -> None:
    super().__init__()
    self.embed_dim = embed_dim
    self.d_model = d_model
    self.max_sents = max_sents
    self.doc_token = nn.Parameter(torch.zeros(1, 1, d_model))
    self.sent_proj = nn.Linear(embed_dim, d_model)
    self.pos_emb = nn.Embedding(max_sents + 1, d_model)
    self.para_start_emb = nn.Embedding(1, d_model)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      batch_first=True,
      activation="gelu",
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

  def forward(
    self,
    sent_emb: torch.Tensor,
    *,
    para_start_mask: torch.Tensor,
    seq_len: torch.Tensor,
  ) -> torch.Tensor:
    """sent_emb: [B, S, embed_dim], para_start_mask: [B, S] bool, seq_len: [B] int."""
    bsz, max_s, _ = sent_emb.shape
    doc = self.doc_token.expand(bsz, -1, -1)
    sent = self.sent_proj(sent_emb)
    pos_ids = torch.arange(1, max_s + 1, device=sent_emb.device).unsqueeze(0).expand(bsz, -1)
    sent = sent + self.pos_emb(pos_ids)
    para_bias = self.para_start_emb(torch.zeros(1, dtype=torch.long, device=sent_emb.device))
    sent = sent + para_start_mask.unsqueeze(-1).float() * para_bias

    seq = torch.cat([doc, sent], dim=1)
    max_len = seq.shape[1]
    pad_mask = torch.arange(max_len, device=sent_emb.device).unsqueeze(0) >= (
      seq_len.unsqueeze(1) + 1
    )
    out = self.transformer(seq, src_key_padding_mask=pad_mask)
    return out[:, 0, :]


class SentSeqRewardModel(nn.Module):
  def __init__(
    self,
    encoder: SentSeqEncoder,
    *,
    feature_dim: int,
  ) -> None:
    super().__init__()
    self.encoder = encoder
    self.head = LinearRewardHead(feature_dim)

  def document_vectors(
    self,
    sent_emb: torch.Tensor,
    *,
    para_start_mask: torch.Tensor,
    seq_len: torch.Tensor,
  ) -> torch.Tensor:
    return self.encoder(sent_emb, para_start_mask=para_start_mask, seq_len=seq_len)

  def score_from_doc_vectors(
    self,
    source_vec: torch.Tensor,
    candidate_vec: torch.Tensor,
    *,
    len_source: torch.Tensor,
    len_candidate: torch.Tensor,
  ) -> torch.Tensor:
    feats = []
    for i in range(source_vec.shape[0]):
      feat = assemble_pointwise_feature_vector(
        source_vec[i].detach().cpu().numpy(),
        candidate_vec[i].detach().cpu().numpy(),
        len_source=float(len_source[i].item()),
        len_candidate=float(len_candidate[i].item()),
      )
      feats.append(feat)
    x = torch.tensor(np.stack(feats), dtype=torch.float32, device=source_vec.device)
    return self.head(x)

  def score_from_doc_vectors_fast(
    self,
    source_vec: torch.Tensor,
    candidate_vec: torch.Tensor,
    *,
    len_source: torch.Tensor,
    len_candidate: torch.Tensor,
  ) -> torch.Tensor:
    """勾配を保ったまま特徴を組み立てる。"""
    diff = source_vec - candidate_vec
    abs_diff = diff.abs()
    cos = (source_vec * candidate_vec).sum(dim=-1, keepdim=True)
    len_s = len_source.unsqueeze(-1)
    len_c = len_candidate.unsqueeze(-1)
    numeric = torch.cat(
      [
        cos,
        len_s,
        len_c,
        len_c - len_s,
        (len_c - len_s).abs(),
      ],
      dim=-1,
    )
    x = torch.cat([source_vec, candidate_vec, diff, abs_diff, numeric], dim=-1)
    return self.head(x)


@dataclass
class SentSeqTrainConfig:
  sentence_model_name: str = "cl-nagoya/ruri-v3-30m"
  embed_backend: str = "sentence-transformers"
  modernbert_checkpoint: str = ""
  truncate_dim: int | None = None
  text_prefix: str = "文章: "
  max_seq_length: int = 256
  encode_batch_size: int = 64
  d_model: int = 256
  nhead: int = 4
  num_layers: int = 2
  dim_feedforward: int = 512
  dropout: float = 0.1
  max_sents: int = 128
  batch_size: int = 64
  epochs: int = 20
  lr: float = 1e-4
  weight_decay: float = 1e-2
  seed: int = 0


@dataclass
class PreparedSentSeqData:
  text_units: dict[str, list[SentenceUnit]]
  sent_to_idx: dict[str, int]
  sent_emb: torch.Tensor


def prepare_sentence_data(
  texts: list[str],
  *,
  sent_to_embedding: dict[str, np.ndarray],
  max_sents: int,
  device: torch.device,
) -> PreparedSentSeqData:
  unique_sents: list[str] = []
  seen: set[str] = set()
  text_units: dict[str, list[SentenceUnit]] = {}
  for text in texts:
    units = truncate_sentence_units(
      split_document_sentences(text),
      max_sents=max_sents,
    )
    text_units[text] = units
    for unit in units:
      if unit.text not in seen:
        seen.add(unit.text)
        unique_sents.append(unit.text)

  sent_to_idx = {s: i for i, s in enumerate(unique_sents)}
  if unique_sents:
    emb = np.stack(
      [sent_to_embedding[s] for s in unique_sents],
      axis=0,
    ).astype(np.float32, copy=False)
    sent_emb = torch.tensor(emb, dtype=torch.float32, device=device)
  else:
    sent_emb = torch.zeros((0, 0), dtype=torch.float32, device=device)
  return PreparedSentSeqData(
    text_units=text_units,
    sent_to_idx=sent_to_idx,
    sent_emb=sent_emb,
  )


def build_document_batch_tensors(
  texts: list[str],
  prepared: PreparedSentSeqData,
  *,
  embed_dim: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  bsz = len(texts)
  max_s = 0
  unit_lists: list[list[SentenceUnit]] = []
  for text in texts:
    units = prepared.text_units[text]
    unit_lists.append(units)
    max_s = max(max_s, len(units))
  if max_s == 0:
    max_s = 1

  sent_emb = torch.zeros((bsz, max_s, embed_dim), dtype=torch.float32, device=device)
  para_start_mask = torch.zeros((bsz, max_s), dtype=torch.bool, device=device)
  seq_len = torch.zeros((bsz,), dtype=torch.long, device=device)

  for i, units in enumerate(unit_lists):
    seq_len[i] = len(units)
    for j, unit in enumerate(units):
      idx = prepared.sent_to_idx[unit.text]
      sent_emb[i, j] = prepared.sent_emb[idx]
      para_start_mask[i, j] = unit.is_para_start
  return sent_emb, para_start_mask, seq_len


def encode_texts_to_doc_vectors(
  model: SentSeqRewardModel,
  texts: list[str],
  prepared: PreparedSentSeqData,
  *,
  device: torch.device,
) -> torch.Tensor:
  if not texts:
    return torch.zeros((0, model.encoder.d_model), device=device)
  embed_dim = model.encoder.embed_dim
  sent_emb, para_start_mask, seq_len = build_document_batch_tensors(
    texts,
    prepared,
    embed_dim=embed_dim,
    device=device,
  )
  return model.document_vectors(
    sent_emb,
    para_start_mask=para_start_mask,
    seq_len=seq_len,
  )


def resolve_device(name: str) -> torch.device:
  if name == "cuda":
    if not torch.cuda.is_available():
      raise SystemExit("--device cuda requested but CUDA is not available")
    return torch.device("cuda")
  return torch.device("cpu")


def collect_unique_sentences(texts: list[str], *, max_sents: int) -> list[str]:
  """文書群から、切り詰め後に登場するユニーク文を収集する。"""
  unique_sents: list[str] = []
  seen_sents: set[str] = set()
  for text in texts:
    units = truncate_sentence_units(
      split_document_sentences(text),
      max_sents=max_sents,
    )
    for unit in units:
      if unit.text not in seen_sents:
        seen_sents.add(unit.text)
        unique_sents.append(unit.text)
  return unique_sents


def encode_unique_sentences(
  rows: list[dict],
  cfg: SentSeqTrainConfig,
  *,
  device: torch.device,
  show_progress_bar: bool = True,
) -> dict[str, np.ndarray]:
  """行群の全文書からユニーク文を集めて一括埋め込みする。

  埋め込みモデルは凍結しており fold に依存しないので、LOPO 評価では
  fold ループの前に一度だけ呼んで使い回せる。
  """
  all_texts = collect_unique_texts(rows)
  unique_sents = collect_unique_sentences(all_texts, max_sents=cfg.max_sents)
  if cfg.embed_backend == "modernbert":
    from modernbert_embed import build_encoder, encode_text_map as encode_modernbert_map

    mb_encoder = build_encoder(
      base_model=cfg.sentence_model_name,
      checkpoint_dir=cfg.modernbert_checkpoint,
      max_seq_length=cfg.max_seq_length if cfg.max_seq_length > 0 else 256,
      device=device,
    )
    return encode_modernbert_map(
      mb_encoder,
      unique_sents,
      batch_size=cfg.encode_batch_size,
      show_progress_bar=show_progress_bar,
    )
  truncate_dim = normalize_truncate_dim(cfg.truncate_dim)
  encoder = SentenceTransformer(
    cfg.sentence_model_name,
    device=str(device),
    truncate_dim=truncate_dim,
  )
  if cfg.max_seq_length > 0:
    encoder.max_seq_length = cfg.max_seq_length
  for param in encoder.parameters():
    param.requires_grad = False
  return encode_text_map(
    encoder,
    unique_sents,
    batch_size=cfg.encode_batch_size,
    normalize_embeddings=True,
    text_prefix=cfg.text_prefix,
    show_progress_bar=show_progress_bar,
  )


def train_sentseq_model(
  train_rows: list[dict],
  valid_rows: list[dict],
  cfg: SentSeqTrainConfig,
  *,
  device: torch.device,
  log_prefix: str = "",
  precomputed_embeddings: dict[str, np.ndarray] | None = None,
  init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[SentSeqRewardModel, dict[str, float], dict[str, float], PreparedSentSeqData]:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  all_texts = collect_unique_texts(train_rows + valid_rows)
  unique_sents = collect_unique_sentences(all_texts, max_sents=cfg.max_sents)

  if precomputed_embeddings is not None:
    missing = [s for s in unique_sents if s not in precomputed_embeddings]
    if missing:
      raise SystemExit(
        f"precomputed embeddings missing {len(missing)} sentences "
        f"(e.g. {missing[0][:40]!r})"
      )
    sent_to_embedding = {s: precomputed_embeddings[s] for s in unique_sents}
  else:
    sent_to_embedding = encode_unique_sentences(
      train_rows + valid_rows,
      cfg,
      device=device,
      show_progress_bar=bool(log_prefix),
    )
  embed_dim = next(iter(sent_to_embedding.values())).shape[0] if sent_to_embedding else 0
  if embed_dim == 0:
    raise SystemExit("no sentences found in training data")

  prepared = prepare_sentence_data(
    all_texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=cfg.max_sents,
    device=device,
  )

  feature_dim = cfg.d_model * 4 + 5
  sent_encoder = SentSeqEncoder(
    embed_dim=embed_dim,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
    max_sents=cfg.max_sents,
  )
  model = SentSeqRewardModel(sent_encoder, feature_dim=feature_dim).to(device)
  if init_state is not None:
    model.load_state_dict(init_state)
    print(f"{log_prefix}warm start: loaded initial weights", flush=True)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
  )

  n = len(train_rows)
  best_valid_acc = -1.0
  best_epoch = -1
  best_state: dict[str, torch.Tensor] | None = None
  best_train_metrics: dict[str, float] = {}
  best_valid_metrics: dict[str, float] = {}
  last_train_metrics: dict[str, float] = {}

  print(
    f"{log_prefix}train: pairs={n} device={device.type} embed_dim={embed_dim} "
    f"unique_sents={len(unique_sents)} epochs={cfg.epochs} bs={cfg.batch_size}",
    flush=True,
  )

  for epoch in range(cfg.epochs):
    model.train()
    order = list(range(n))
    random.shuffle(order)
    epoch_loss = 0.0
    n_batches = 0
    train_correct = 0
    train_total = 0

    for start in range(0, n, cfg.batch_size):
      batch = [train_rows[i] for i in order[start : start + cfg.batch_size]]
      sources = [r["source_text"] for r in batch]
      chosen = [r["candidate_a"] for r in batch]
      rejected = [r["candidate_b"] for r in batch]

      v_src = encode_texts_to_doc_vectors(model, sources, prepared, device=device)
      v_w = encode_texts_to_doc_vectors(model, chosen, prepared, device=device)
      v_l = encode_texts_to_doc_vectors(model, rejected, prepared, device=device)
      len_s = torch.tensor(
        [float(len(t)) for t in sources], dtype=torch.float32, device=device
      )
      len_w = torch.tensor(
        [float(len(t)) for t in chosen], dtype=torch.float32, device=device
      )
      len_l = torch.tensor(
        [float(len(t)) for t in rejected], dtype=torch.float32, device=device
      )

      s_w = model.score_from_doc_vectors_fast(v_src, v_w, len_source=len_s, len_candidate=len_w)
      s_l = model.score_from_doc_vectors_fast(v_src, v_l, len_source=len_s, len_candidate=len_l)
      loss = torch.nn.functional.softplus(-(s_w - s_l)).mean()

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      epoch_loss += float(loss.item())
      n_batches += 1
      train_correct += int((s_w > s_l).sum().item())
      train_total += len(batch)

    last_train_metrics = {
      "train_bt_loss": epoch_loss / max(n_batches, 1),
      "train_pair_accuracy": train_correct / max(train_total, 1),
    }
    valid_metrics = eval_sentseq_pairs(
      model, valid_rows, prepared, device=device, batch_size=cfg.batch_size
    )
    print(
      f"{log_prefix}epoch {epoch + 1}/{cfg.epochs} "
      f"train_loss={last_train_metrics['train_bt_loss']:.4f} "
      f"train_acc={last_train_metrics['train_pair_accuracy']:.4f} "
      f"valid_acc={valid_metrics['pair_accuracy']:.4f}",
      flush=True,
    )
    if valid_metrics["pair_accuracy"] > best_valid_acc:
      best_valid_acc = valid_metrics["pair_accuracy"]
      best_epoch = epoch + 1
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      best_train_metrics = dict(last_train_metrics)
      best_valid_metrics = {f"valid_{k}": v for k, v in valid_metrics.items()}

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()
  print(f"{log_prefix}best epoch: {best_epoch}/{cfg.epochs} valid_acc={best_valid_acc:.4f}", flush=True)
  best_train_metrics["best_epoch"] = float(best_epoch)
  return model, best_train_metrics, best_valid_metrics, prepared


@torch.no_grad()
def eval_sentseq_pairs(
  model: SentSeqRewardModel,
  rows: list[dict],
  prepared: PreparedSentSeqData,
  *,
  device: torch.device,
  batch_size: int,
) -> dict[str, float]:
  model.eval()
  margins: list[float] = []
  for start in range(0, len(rows), batch_size):
    batch = rows[start : start + batch_size]
    sources = [r["source_text"] for r in batch]
    chosen = [r["candidate_a"] for r in batch]
    rejected = [r["candidate_b"] for r in batch]
    v_src = encode_texts_to_doc_vectors(model, sources, prepared, device=device)
    v_w = encode_texts_to_doc_vectors(model, chosen, prepared, device=device)
    v_l = encode_texts_to_doc_vectors(model, rejected, prepared, device=device)
    len_s = torch.tensor(
      [float(len(t)) for t in sources], dtype=torch.float32, device=device
    )
    len_w = torch.tensor(
      [float(len(t)) for t in chosen], dtype=torch.float32, device=device
    )
    len_l = torch.tensor(
      [float(len(t)) for t in rejected], dtype=torch.float32, device=device
    )
    s_w = model.score_from_doc_vectors_fast(v_src, v_w, len_source=len_s, len_candidate=len_w)
    s_l = model.score_from_doc_vectors_fast(v_src, v_l, len_source=len_s, len_candidate=len_l)
    margins.extend((s_w - s_l).float().cpu().tolist())
  m = torch.tensor(margins)
  return {
    "pair_accuracy": float((m > 0).float().mean().item()),
    "bt_loss": float(torch.nn.functional.softplus(-m).mean().item()),
    "mean_margin": float(m.mean().item()),
  }


def build_model_config(cfg: SentSeqTrainConfig, *, embed_dim: int, feature_dim: int) -> dict:
  return {
    "kind": "pref-sentseq",
    "d_model": cfg.d_model,
    "nhead": cfg.nhead,
    "num_layers": cfg.num_layers,
    "dim_feedforward": cfg.dim_feedforward,
    "dropout": cfg.dropout,
    "max_sents": cfg.max_sents,
    "embed_dim": embed_dim,
    "feature_dim": feature_dim,
    "sentence_model_name": cfg.sentence_model_name,
    "embed_backend": cfg.embed_backend,
    "modernbert_base_model": cfg.sentence_model_name,
    "modernbert_checkpoint": cfg.modernbert_checkpoint or None,
    "truncate_dim": normalize_truncate_dim(cfg.truncate_dim),
    "text_prefix": cfg.text_prefix,
    "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
    "normalize_embeddings": True,
    "sent_split_version": SENTSEQ_SPLIT_VERSION,
    "para_boundary_mode": PARA_BOUNDARY_MODE,
    "feature_version": "source-cand-pointwise-v1",
  }


def save_sentseq_model(
  model: SentSeqRewardModel,
  cfg: SentSeqTrainConfig,
  *,
  embed_dim: int,
  output_dir: Path,
) -> None:
  feature_dim = cfg.d_model * 4 + 5
  artifact = {
    "kind": "pref-sentseq",
    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "config": build_model_config(cfg, embed_dim=embed_dim, feature_dim=feature_dim),
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  torch.save(artifact, output_dir / "model.pt")


def load_sentseq_model_from_artifact(
  artifact: dict,
  *,
  device: torch.device,
) -> SentSeqRewardModel:
  config = artifact["config"]
  if config.get("kind") not in ("pref-sentseq", "pref-detect"):
    raise ValueError("not a pref-sentseq artifact")
  embed_dim = int(config["embed_dim"])
  feature_dim = int(config["feature_dim"])
  encoder = SentSeqEncoder(
    embed_dim=embed_dim,
    d_model=int(config["d_model"]),
    nhead=int(config["nhead"]),
    num_layers=int(config["num_layers"]),
    dim_feedforward=int(config["dim_feedforward"]),
    dropout=float(config["dropout"]),
    max_sents=int(config["max_sents"]),
  )
  model = SentSeqRewardModel(encoder, feature_dim=feature_dim)
  model.load_state_dict(artifact["model_state_dict"])
  model.to(device)
  model.eval()
  return model


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument(
    "--embed-backend",
    choices=["sentence-transformers", "modernbert"],
    default="sentence-transformers",
  )
  parser.add_argument(
    "--modernbert-checkpoint",
    default="",
    help="D 学習済み checkpoint ディレクトリ。空なら --model の素の ModernBERT",
  )
  parser.add_argument("--train-file", required=True, help="preference jsonl (swap 込み可)")
  parser.add_argument("--eval-file", required=True, help="preference jsonl")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--num-layers", type=int, default=2)
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  parser.add_argument(
    "--init-from",
    default="",
    help="学習済み model.pt から重みを引き継いで追学習（構成は artifact 側に合わせる）",
  )
  args = parser.parse_args()

  device = resolve_device(args.device)
  train_rows = unique_preference_pairs(load_jsonl(args.train_file))
  eval_rows = unique_preference_pairs(load_jsonl(args.eval_file))
  if not train_rows or not eval_rows:
    raise SystemExit("train/eval preference pairs are empty after filtering swaps")

  init_state = None
  init_config: dict = {}
  if args.init_from:
    artifact = torch.load(args.init_from, map_location="cpu", weights_only=False)
    if artifact.get("kind") != "pref-sentseq":
      raise SystemExit(f"not a pref-sentseq artifact: {args.init_from}")
    init_state = artifact["model_state_dict"]
    init_config = artifact["config"]
    print(f"warm start from: {args.init_from}", flush=True)

  cfg = SentSeqTrainConfig(
    sentence_model_name=init_config.get("sentence_model_name", args.model),
    embed_backend=init_config.get("embed_backend", args.embed_backend),
    modernbert_checkpoint=init_config.get("modernbert_checkpoint")
    or str(args.modernbert_checkpoint or "").strip(),
    truncate_dim=init_config.get("truncate_dim", normalize_truncate_dim(args.truncate_dim)),
    text_prefix=init_config.get("text_prefix", args.text_prefix),
    max_seq_length=init_config.get("max_seq_length") or args.max_seq_length,
    encode_batch_size=args.encode_batch_size,
    d_model=init_config.get("d_model", args.d_model),
    nhead=init_config.get("nhead", SentSeqTrainConfig.nhead),
    num_layers=init_config.get("num_layers", args.num_layers),
    dim_feedforward=init_config.get("dim_feedforward", SentSeqTrainConfig.dim_feedforward),
    dropout=init_config.get("dropout", SentSeqTrainConfig.dropout),
    max_sents=init_config.get("max_sents", args.max_sents),
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
  )

  model, train_metrics, valid_metrics, prepared = train_sentseq_model(
    train_rows,
    eval_rows,
    cfg,
    device=device,
    init_state=init_state,
  )
  embed_dim = model.encoder.embed_dim
  output_dir = Path(args.output_dir)
  save_sentseq_model(model, cfg, embed_dim=embed_dim, output_dir=output_dir)

  metrics = {
    "embedding_model": cfg.sentence_model_name,
    "embed_backend": cfg.embed_backend,
    "modernbert_checkpoint": cfg.modernbert_checkpoint or None,
    "text_prefix": cfg.text_prefix,
    "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
    "max_sents": cfg.max_sents,
    "d_model": cfg.d_model,
    "num_layers": cfg.num_layers,
    "epochs": cfg.epochs,
    "lr": cfg.lr,
    "batch_size": cfg.batch_size,
    "init_from": args.init_from or None,
    "train_pairs": len(train_rows),
    "eval_pairs": len(eval_rows),
    **train_metrics,
    **valid_metrics,
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"saved: {output_dir}")
  print(f"valid_pair_accuracy: {valid_metrics['valid_pair_accuracy']:.4f}")
  print(f"valid_bt_loss: {valid_metrics['valid_bt_loss']:.4f}")
  print(f"valid_mean_margin: {valid_metrics['valid_mean_margin']:.4f}")


if __name__ == "__main__":
  main()
