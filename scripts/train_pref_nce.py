#!/usr/bin/env python3
"""下書きからその人の推敲ベクトルを置き、候補とのコサインを InfoNCE で学ぶ。

正例は同じ下書きの人間の推敲。負例は下書きそのものと Composer の推敲。
ruri の重みは更新しない。文書ベクトルは文列 Transformer。
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn

from pref_static_utils import encode_text_map, load_jsonl, normalize_truncate_dim
from sentseq_utils import PARA_BOUNDARY_MODE, SENTSEQ_SPLIT_VERSION
from setwise_triple_utils import reconstruct_triples_from_pref_rows
from train_pref_sentseq import (
  SentSeqEncoder,
  build_document_batch_tensors,
  collect_unique_sentences,
  prepare_sentence_data,
  resolve_device,
)

KIND = "pref-nce"
HUMAN_INDEX = 0
DRAFT_INDEX = 1
COMPOSER_INDEX = 2
COMPARE_EPS = 1e-6


@dataclass
class NceTrainConfig:
  sentence_model_name: str = "cl-nagoya/ruri-v3-30m"
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
  batch_size: int = 32
  epochs: int = 40
  lr: float = 3e-4
  weight_decay: float = 1e-2
  seed: int = 0
  tau: float = 0.07


class PrefNceModel(nn.Module):
  def __init__(self, encoder: SentSeqEncoder, *, tau: float) -> None:
    super().__init__()
    self.encoder = encoder
    d_model = encoder.d_model
    self.query_head = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.GELU(),
      nn.Linear(d_model, d_model),
    )
    self.tau = float(tau)

  def document_vectors(
    self,
    sent_emb: torch.Tensor,
    *,
    para_start_mask: torch.Tensor,
    seq_len: torch.Tensor,
  ) -> torch.Tensor:
    return self.encoder(sent_emb, para_start_mask=para_start_mask, seq_len=seq_len)

  def queries_from_draft(self, draft_vec: torch.Tensor) -> torch.Tensor:
    return F.normalize(self.query_head(draft_vec), dim=-1)


def encode_texts_with_encoder(
  encoder: SentSeqEncoder,
  texts: list[str],
  prepared,
  *,
  device: torch.device,
) -> torch.Tensor:
  if not texts:
    return torch.zeros((0, encoder.d_model), device=device)
  sent_emb, para_start_mask, seq_len = build_document_batch_tensors(
    texts,
    prepared,
    embed_dim=encoder.embed_dim,
    device=device,
  )
  return encoder(sent_emb, para_start_mask=para_start_mask, seq_len=seq_len)


def nce_logits(
  model: PrefNceModel,
  draft_vec: torch.Tensor,
  human_vec: torch.Tensor,
  composer_vec: torch.Tensor,
) -> torch.Tensor:
  query = model.queries_from_draft(draft_vec)
  keys = torch.stack(
    [
      F.normalize(human_vec, dim=-1),
      F.normalize(draft_vec, dim=-1),
      F.normalize(composer_vec, dim=-1),
    ],
    dim=1,
  )
  scale = 1.0 / max(model.tau, 1e-8)
  return scale * torch.einsum("bd,bnd->bn", query, keys)


def nce_loss(logits: torch.Tensor) -> torch.Tensor:
  target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
  return F.cross_entropy(logits, target)


def _strictly_greater(a: float, b: float) -> bool:
  return a > b + COMPARE_EPS


def encode_triple_vectors(
  model: PrefNceModel,
  triples: list,
  prepared,
  *,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  drafts = [t.draft for t in triples]
  humans = [t.human for t in triples]
  composers = [t.composer for t in triples]
  draft_vec = encode_texts_with_encoder(model.encoder, drafts, prepared, device=device)
  human_vec = encode_texts_with_encoder(model.encoder, humans, prepared, device=device)
  composer_vec = encode_texts_with_encoder(model.encoder, composers, prepared, device=device)
  return draft_vec, human_vec, composer_vec


@torch.no_grad()
def eval_triples(
  model: PrefNceModel,
  triples: list,
  prepared,
  *,
  device: torch.device,
  batch_size: int,
) -> dict:
  model.eval()
  losses: list[float] = []
  items: list[dict] = []
  for start in range(0, len(triples), batch_size):
    batch = triples[start : start + batch_size]
    draft_vec, human_vec, composer_vec = encode_triple_vectors(
      model, batch, prepared, device=device
    )
    logits = nce_logits(model, draft_vec, human_vec, composer_vec)
    losses.append(float(nce_loss(logits).item()))
    scores = logits.detach().cpu()
    for i, triple in enumerate(batch):
      human_s = float(scores[i, HUMAN_INDEX])
      draft_s = float(scores[i, DRAFT_INDEX])
      composer_s = float(scores[i, COMPOSER_INDEX])
      items.append(
        {
          "item_id": triple.item_id,
          "scores": {"human": human_s, "draft": draft_s, "composer": composer_s},
          "draft_over_human": _strictly_greater(draft_s, human_s),
          "composer_over_human": _strictly_greater(composer_s, human_s),
        }
      )
  n = len(items)
  n_draft_over_human = sum(1 for row in items if row["draft_over_human"])
  n_composer_over_human = sum(1 for row in items if row["composer_over_human"])
  return {
    "nce_loss": float(sum(losses) / max(len(losses), 1)),
    "n": n,
    "n_draft_over_human": n_draft_over_human,
    "n_composer_over_human": n_composer_over_human,
    "items": items,
  }


def encode_corpus_sentences(
  texts: list[str],
  cfg: NceTrainConfig,
  *,
  device: torch.device,
  show_progress_bar: bool,
) -> dict[str, np.ndarray]:
  unique_sents = collect_unique_sentences(texts, max_sents=cfg.max_sents)
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


def collect_triple_texts(triples: list) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for triple in triples:
    for value in (triple.draft, triple.human, triple.composer):
      if value not in seen:
        seen.add(value)
        texts.append(value)
  return texts


def build_model_config(cfg: NceTrainConfig, *, embed_dim: int) -> dict:
  return {
    "kind": KIND,
    "d_model": cfg.d_model,
    "nhead": cfg.nhead,
    "num_layers": cfg.num_layers,
    "dim_feedforward": cfg.dim_feedforward,
    "dropout": cfg.dropout,
    "max_sents": cfg.max_sents,
    "embed_dim": embed_dim,
    "sentence_model_name": cfg.sentence_model_name,
    "truncate_dim": normalize_truncate_dim(cfg.truncate_dim),
    "text_prefix": cfg.text_prefix,
    "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
    "normalize_embeddings": True,
    "sent_split_version": SENTSEQ_SPLIT_VERSION,
    "para_boundary_mode": PARA_BOUNDARY_MODE,
    "tau": cfg.tau,
    "loss": "infonce_human_pos_draft_composer_neg",
  }


def save_nce_model(
  model: PrefNceModel,
  cfg: NceTrainConfig,
  *,
  embed_dim: int,
  output_dir: Path,
) -> None:
  config = build_model_config(cfg, embed_dim=embed_dim)
  artifact = {
    "kind": KIND,
    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "config": config,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  torch.save(artifact, output_dir / "model.pt")
  (output_dir / "meta.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def load_nce_model_from_artifact(artifact: dict, *, device: torch.device) -> PrefNceModel:
  config = artifact["config"]
  if config.get("kind") != KIND:
    raise ValueError("not a pref-nce artifact")
  encoder = SentSeqEncoder(
    embed_dim=int(config["embed_dim"]),
    d_model=int(config["d_model"]),
    nhead=int(config["nhead"]),
    num_layers=int(config["num_layers"]),
    dim_feedforward=int(config["dim_feedforward"]),
    dropout=float(config["dropout"]),
    max_sents=int(config["max_sents"]),
  )
  model = PrefNceModel(encoder, tau=float(config.get("tau", 0.07)))
  model.load_state_dict(artifact["model_state_dict"])
  model.to(device)
  model.eval()
  return model


def train_nce_model(
  train_triples: list,
  valid_triples: list,
  cfg: NceTrainConfig,
  *,
  device: torch.device,
  precomputed_embeddings: dict[str, np.ndarray] | None = None,
  log_prefix: str = "",
) -> tuple[PrefNceModel, dict, dict]:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  all_texts = collect_triple_texts(train_triples + valid_triples)
  if precomputed_embeddings is not None:
    unique_sents = collect_unique_sentences(all_texts, max_sents=cfg.max_sents)
    missing = [s for s in unique_sents if s not in precomputed_embeddings]
    if missing:
      raise SystemExit(f"precomputed embeddings missing {len(missing)} sentences")
    sent_to_embedding = {s: precomputed_embeddings[s] for s in unique_sents}
  else:
    sent_to_embedding = encode_corpus_sentences(
      all_texts,
      cfg,
      device=device,
      show_progress_bar=bool(log_prefix),
    )
  embed_dim = next(iter(sent_to_embedding.values())).shape[0]
  prepared = prepare_sentence_data(
    all_texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=cfg.max_sents,
    device=device,
  )
  encoder = SentSeqEncoder(
    embed_dim=embed_dim,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
    max_sents=cfg.max_sents,
  ).to(device)
  model = PrefNceModel(encoder, tau=cfg.tau).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

  n = len(train_triples)
  best_train_loss = float("inf")
  best_epoch = -1
  best_state: dict[str, torch.Tensor] | None = None
  best_train_metrics: dict = {}

  print(
    f"{log_prefix}train: triples={n} valid={len(valid_triples)} "
    f"device={device.type} embed_dim={embed_dim} epochs={cfg.epochs} "
    f"bs={cfg.batch_size} tau={cfg.tau}",
    flush=True,
  )

  for epoch in range(cfg.epochs):
    model.train()
    order = list(range(n))
    random.shuffle(order)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, n, cfg.batch_size):
      batch = [train_triples[i] for i in order[start : start + cfg.batch_size]]
      optimizer.zero_grad()
      draft_vec, human_vec, composer_vec = encode_triple_vectors(
        model, batch, prepared, device=device
      )
      logits = nce_logits(model, draft_vec, human_vec, composer_vec)
      loss = nce_loss(logits)
      loss.backward()
      optimizer.step()
      epoch_loss += float(loss.item())
      n_batches += 1
    train_loss = epoch_loss / max(n_batches, 1)
    last_valid = eval_triples(
      model,
      valid_triples,
      prepared,
      device=device,
      batch_size=cfg.batch_size,
    )
    print(
      f"{log_prefix}epoch {epoch + 1}/{cfg.epochs} "
      f"train_nce={train_loss:.4f} "
      f"valid_nce={last_valid['nce_loss']:.4f} "
      f"valid_draft_over_human={last_valid['n_draft_over_human']}/{last_valid['n']} "
      f"valid_composer_over_human={last_valid['n_composer_over_human']}/{last_valid['n']}",
      flush=True,
    )
    if train_loss < best_train_loss:
      best_train_loss = train_loss
      best_epoch = epoch + 1
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      best_train_metrics = {
        "train_nce_loss": train_loss,
        "best_epoch": float(best_epoch),
      }

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()
  print(
    f"{log_prefix}best epoch: {best_epoch}/{cfg.epochs} train_nce={best_train_loss:.4f}",
    flush=True,
  )
  valid_at_best = eval_triples(
    model,
    valid_triples,
    prepared,
    device=device,
    batch_size=cfg.batch_size,
  )
  return model, best_train_metrics, valid_at_best


def _limit(items: list, n: int) -> list:
  if n <= 0:
    return items
  return items[:n]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--train-file", default="data/section_middle/pref_train.jsonl")
  parser.add_argument("--eval-file", default="data/section_middle/pref_valid.jsonl")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--nhead", type=int, default=4)
  parser.add_argument("--num-layers", type=int, default=2)
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  parser.add_argument("--max-train", type=int, default=0)
  parser.add_argument("--max-valid", type=int, default=0)
  parser.add_argument("--tau", type=float, default=0.07)
  args = parser.parse_args()

  device = resolve_device(args.device)
  train_triples = _limit(
    reconstruct_triples_from_pref_rows(load_jsonl(args.train_file)),
    args.max_train,
  )
  valid_triples = _limit(
    reconstruct_triples_from_pref_rows(load_jsonl(args.eval_file)),
    args.max_valid,
  )
  if not train_triples:
    raise SystemExit("no training triples")
  cfg = NceTrainConfig(
    sentence_model_name=args.model,
    truncate_dim=normalize_truncate_dim(args.truncate_dim),
    text_prefix=args.text_prefix,
    max_seq_length=args.max_seq_length,
    encode_batch_size=args.encode_batch_size,
    d_model=args.d_model,
    nhead=args.nhead,
    num_layers=args.num_layers,
    max_sents=args.max_sents,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
    tau=args.tau,
  )
  model, train_metrics, valid_metrics = train_nce_model(
    train_triples,
    valid_triples,
    cfg,
    device=device,
  )
  output_dir = Path(args.output_dir)
  embed_dim = model.encoder.embed_dim
  save_nce_model(model, cfg, embed_dim=embed_dim, output_dir=output_dir)
  items = valid_metrics.pop("items")
  report = {
    **train_metrics,
    "valid_nce_loss": valid_metrics["nce_loss"],
    "valid_n": valid_metrics["n"],
    "valid_n_draft_over_human": valid_metrics["n_draft_over_human"],
    "valid_n_composer_over_human": valid_metrics["n_composer_over_human"],
    "checkpoint_rule": "lowest train InfoNCE",
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  (output_dir / "valid_items.json").write_text(
    json.dumps({"summary": valid_metrics, "items": items}, ensure_ascii=False, indent=2)
    + "\n",
    encoding="utf-8",
  )
  print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
  main()
