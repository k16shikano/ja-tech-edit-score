#!/usr/bin/env python3
"""凍結 ruri 文埋め込み + 文列 Transformer + GPM 選好埋め込み。"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pref_static_utils import load_jsonl, normalize_truncate_dim
from sentseq_utils import PARA_BOUNDARY_MODE, SENTSEQ_SPLIT_VERSION
from setwise_triple_utils import reconstruct_triples_from_pref_rows
from train_pref_sentseq import (
  SentSeqEncoder,
  SentSeqTrainConfig,
  collect_unique_texts,
  encode_texts_to_doc_vectors,
  encode_unique_sentences,
  prepare_sentence_data,
  resolve_device,
  unique_preference_pairs,
)

KIND = "pref-gpm"
TRIPLE_PAIR_KINDS = ("gold_vs_draft", "gold_vs_gen", "gen_vs_draft")
FEATURE_VERSION = "source-cand-pointwise-v1"


def validate_head_dim(head_dim: int) -> None:
  if head_dim % 2 != 0:
    raise ValueError(f"head_dim must be even, got {head_dim}")


def pairwise_preference_score(
  v_a: torch.Tensor,
  v_b: torch.Tensor,
) -> torch.Tensor:
  if v_a.shape[-1] % 2 != 0:
    raise ValueError(f"head_dim must be even, got {v_a.shape[-1]}")
  k = v_a.shape[-1] // 2
  va = v_a.reshape(*v_a.shape[:-1], k, 2)
  vb = v_b.reshape(*v_b.shape[:-1], k, 2)
  return (va[..., 0] * vb[..., 1] - va[..., 1] * vb[..., 0]).sum(dim=-1)


def load_hunk_preference_rows(rows: list[dict]) -> list[dict]:
  return unique_preference_pairs(rows)


def load_triple_preference_rows(
  rows: list[dict],
  *,
  drop_gen_over_draft: bool = False,
) -> list[dict]:
  out: list[dict] = []
  for row in rows:
    pair_kind = str(row.get("pair_kind") or "")
    if pair_kind not in TRIPLE_PAIR_KINDS:
      continue
    if int(row.get("label", 0)) != 1:
      continue
    if drop_gen_over_draft and pair_kind == "gen_vs_draft":
      continue
    out.append(row)
  return out


def merge_training_rows(
  hunk_rows: list[dict],
  triple_rows: list[dict],
  *,
  drop_gen_over_draft: bool,
) -> list[dict]:
  return list(hunk_rows) + load_triple_preference_rows(
    triple_rows,
    drop_gen_over_draft=drop_gen_over_draft,
  )


def section_has_preference_cycle(
  s_gold_draft: float,
  s_gold_gen: float,
  s_gen_draft: float,
) -> bool:
  cycle_a = s_gold_gen > 0.0 and s_gen_draft > 0.0 and s_gold_draft <= 0.0
  cycle_b = s_gold_draft > 0.0 and s_gen_draft <= 0.0 and s_gold_gen <= 0.0
  return cycle_a or cycle_b


def count_hunk_draft_over_gold(scores: list[float]) -> int:
  return sum(1 for s in scores if s <= 0.0)


def count_section_draft_over_gold(
  s_gold_draft: list[float],
) -> int:
  return count_hunk_draft_over_gold(s_gold_draft)


def count_section_gen_over_gold(
  s_gold_gen: list[float],
) -> int:
  return count_hunk_draft_over_gold(s_gold_gen)


def count_section_draft_over_gen(
  s_gen_draft: list[float],
) -> int:
  return sum(1 for s in s_gen_draft if s <= 0.0)


def count_section_cycles(
  s_gold_draft: list[float],
  s_gold_gen: list[float],
  s_gen_draft: list[float],
) -> int:
  n = min(len(s_gold_draft), len(s_gold_gen), len(s_gen_draft))
  total = 0
  for i in range(n):
    if section_has_preference_cycle(
      s_gold_draft[i],
      s_gold_gen[i],
      s_gen_draft[i],
    ):
      total += 1
  return total


class PreferenceEmbeddingHead(nn.Module):
  def __init__(
    self,
    in_dim: int,
    head_dim: int,
    *,
    head_hidden: int = 0,
  ) -> None:
    super().__init__()
    validate_head_dim(head_dim)
    if head_hidden > 0:
      self.net = nn.Sequential(
        nn.Linear(in_dim, head_hidden),
        nn.GELU(),
        nn.Linear(head_hidden, head_dim),
      )
    else:
      self.net = nn.Linear(in_dim, head_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class GpmRewardModel(nn.Module):
  def __init__(
    self,
    encoder: SentSeqEncoder,
    *,
    feature_dim: int,
    head_dim: int,
    head_hidden: int = 0,
    normalize_embedding: bool = False,
  ) -> None:
    super().__init__()
    validate_head_dim(head_dim)
    self.encoder = encoder
    self.head = PreferenceEmbeddingHead(
      feature_dim,
      head_dim,
      head_hidden=head_hidden,
    )
    self.normalize_embedding = normalize_embedding
    self.head_dim = head_dim

  def document_vectors(
    self,
    sent_emb: torch.Tensor,
    *,
    para_start_mask: torch.Tensor,
    seq_len: torch.Tensor,
  ) -> torch.Tensor:
    return self.encoder(sent_emb, para_start_mask=para_start_mask, seq_len=seq_len)

  def _assemble_features(
    self,
    source_vec: torch.Tensor,
    candidate_vec: torch.Tensor,
    *,
    len_source: torch.Tensor,
    len_candidate: torch.Tensor,
  ) -> torch.Tensor:
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
    return torch.cat([source_vec, candidate_vec, diff, abs_diff, numeric], dim=-1)

  def embed_from_doc_vectors(
    self,
    source_vec: torch.Tensor,
    candidate_vec: torch.Tensor,
    *,
    len_source: torch.Tensor,
    len_candidate: torch.Tensor,
  ) -> torch.Tensor:
    x = self._assemble_features(
      source_vec,
      candidate_vec,
      len_source=len_source,
      len_candidate=len_candidate,
    )
    v = self.head(x)
    if self.normalize_embedding:
      v = nn.functional.normalize(v, dim=-1)
    return v

  def pair_score(
    self,
    v_a: torch.Tensor,
    v_b: torch.Tensor,
  ) -> torch.Tensor:
    return pairwise_preference_score(v_a, v_b)

  def score_pairs_from_doc_vectors(
    self,
    source_vec: torch.Tensor,
    candidate_a_vec: torch.Tensor,
    candidate_b_vec: torch.Tensor,
    *,
    len_source: torch.Tensor,
    len_a: torch.Tensor,
    len_b: torch.Tensor,
  ) -> torch.Tensor:
    v_a = self.embed_from_doc_vectors(
      source_vec,
      candidate_a_vec,
      len_source=len_source,
      len_candidate=len_a,
    )
    v_b = self.embed_from_doc_vectors(
      source_vec,
      candidate_b_vec,
      len_source=len_source,
      len_candidate=len_b,
    )
    return self.pair_score(v_a, v_b)


@dataclass
class GpmTrainConfig:
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
  head_dim: int = 4
  head_hidden: int = 0
  normalize_embedding: bool = False


def feature_dim_for(cfg: GpmTrainConfig) -> int:
  return cfg.d_model * 4 + 5


def build_model_config(
  cfg: GpmTrainConfig,
  *,
  embed_dim: int,
  feature_dim: int,
) -> dict:
  return {
    "kind": KIND,
    "d_model": cfg.d_model,
    "nhead": cfg.nhead,
    "num_layers": cfg.num_layers,
    "dim_feedforward": cfg.dim_feedforward,
    "dropout": cfg.dropout,
    "max_sents": cfg.max_sents,
    "embed_dim": embed_dim,
    "feature_dim": feature_dim,
    "head_dim": cfg.head_dim,
    "head_hidden": cfg.head_hidden,
    "normalize_embedding": cfg.normalize_embedding,
    "sentence_model_name": cfg.sentence_model_name,
    "truncate_dim": normalize_truncate_dim(cfg.truncate_dim),
    "text_prefix": cfg.text_prefix,
    "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
    "normalize_embeddings": True,
    "sent_split_version": SENTSEQ_SPLIT_VERSION,
    "para_boundary_mode": PARA_BOUNDARY_MODE,
    "feature_version": FEATURE_VERSION,
  }


def save_gpm_model(
  model: GpmRewardModel,
  cfg: GpmTrainConfig,
  *,
  embed_dim: int,
  output_dir: Path,
) -> None:
  feature_dim = feature_dim_for(cfg)
  artifact = {
    "kind": KIND,
    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "config": build_model_config(cfg, embed_dim=embed_dim, feature_dim=feature_dim),
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  torch.save(artifact, output_dir / "model.pt")


def load_gpm_model_from_artifact(
  artifact: dict,
  *,
  device: torch.device,
) -> GpmRewardModel:
  config = artifact["config"]
  if config.get("kind") != KIND:
    raise ValueError("not a pref-gpm artifact")
  embed_dim = int(config["embed_dim"])
  feature_dim = int(config["feature_dim"])
  head_dim = int(config["head_dim"])
  encoder = SentSeqEncoder(
    embed_dim=embed_dim,
    d_model=int(config["d_model"]),
    nhead=int(config["nhead"]),
    num_layers=int(config["num_layers"]),
    dim_feedforward=int(config["dim_feedforward"]),
    dropout=float(config["dropout"]),
    max_sents=int(config["max_sents"]),
  )
  model = GpmRewardModel(
    encoder,
    feature_dim=feature_dim,
    head_dim=head_dim,
    head_hidden=int(config.get("head_hidden", 0)),
    normalize_embedding=bool(config.get("normalize_embedding", False)),
  )
  model.load_state_dict(artifact["model_state_dict"])
  model.to(device)
  model.eval()
  return model


@torch.no_grad()
def score_preference_rows(
  model: GpmRewardModel,
  rows: list[dict],
  prepared,
  *,
  device: torch.device,
  batch_size: int,
) -> list[float]:
  model.eval()
  scores: list[float] = []
  for start in range(0, len(rows), batch_size):
    batch = rows[start : start + batch_size]
    sources = [r["source_text"] for r in batch]
    cand_a = [r["candidate_a"] for r in batch]
    cand_b = [r["candidate_b"] for r in batch]
    v_src = encode_texts_to_doc_vectors(model, sources, prepared, device=device)
    v_a = encode_texts_to_doc_vectors(model, cand_a, prepared, device=device)
    v_b = encode_texts_to_doc_vectors(model, cand_b, prepared, device=device)
    len_s = torch.tensor(
      [float(len(t)) for t in sources], dtype=torch.float32, device=device
    )
    len_a = torch.tensor(
      [float(len(t)) for t in cand_a], dtype=torch.float32, device=device
    )
    len_b = torch.tensor(
      [float(len(t)) for t in cand_b], dtype=torch.float32, device=device
    )
    s = model.score_pairs_from_doc_vectors(
      v_src,
      v_a,
      v_b,
      len_source=len_s,
      len_a=len_a,
      len_b=len_b,
    )
    scores.extend([float(x) for x in s.cpu().tolist()])
  return scores


@torch.no_grad()
def eval_gpm_validation(
  model: GpmRewardModel,
  hunk_rows: list[dict],
  triple_rows: list[dict],
  prepared,
  *,
  device: torch.device,
  batch_size: int,
) -> dict[str, float]:
  model.eval()
  hunk_scores = score_preference_rows(
    model,
    hunk_rows,
    prepared,
    device=device,
    batch_size=batch_size,
  )
  triple_scores = score_preference_rows(
    model,
    triple_rows,
    prepared,
    device=device,
    batch_size=batch_size,
  )
  all_scores = hunk_scores + triple_scores
  valid_pairs_total = len(all_scores)
  valid_pair_accuracy = (
    sum(1 for s in all_scores if s > 0.0) / valid_pairs_total
    if valid_pairs_total
    else 0.0
  )

  valid_hunk_draft_over_gold = count_hunk_draft_over_gold(hunk_scores)

  triples = reconstruct_triples_from_pref_rows(triple_rows)
  s_gold_draft: list[float] = []
  s_gold_gen: list[float] = []
  s_gen_draft: list[float] = []
  for triple in triples:
    rows_for_triple = [
      r
      for r in triple_rows
      if str((r.get("meta") or {}).get("item_id") or "") == triple.item_id
    ]
    by_kind = {str(r.get("pair_kind") or ""): r for r in rows_for_triple}
    batch = [
      by_kind["gold_vs_draft"],
      by_kind["gold_vs_gen"],
      by_kind["gen_vs_draft"],
    ]
    scores = score_preference_rows(
      model,
      batch,
      prepared,
      device=device,
      batch_size=len(batch),
    )
    s_gold_draft.append(scores[0])
    s_gold_gen.append(scores[1])
    s_gen_draft.append(scores[2])

  return {
    "valid_pairs_total": float(valid_pairs_total),
    "valid_pair_accuracy": valid_pair_accuracy,
    "valid_hunk_draft_over_gold": float(valid_hunk_draft_over_gold),
    "valid_section_draft_over_gold": float(count_section_draft_over_gold(s_gold_draft)),
    "valid_section_gen_over_gold": float(count_section_gen_over_gold(s_gold_gen)),
    "valid_section_draft_over_gen": float(count_section_draft_over_gen(s_gen_draft)),
    "valid_section_cycles": float(
      count_section_cycles(s_gold_draft, s_gold_gen, s_gen_draft)
    ),
  }


def train_gpm_model(
  train_rows: list[dict],
  valid_hunk_rows: list[dict],
  valid_triple_rows: list[dict],
  cfg: GpmTrainConfig,
  *,
  device: torch.device,
  log_prefix: str = "",
  precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[GpmRewardModel, dict[str, float], dict[str, float], object]:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)
  validate_head_dim(cfg.head_dim)

  all_texts = collect_unique_texts(
    train_rows + valid_hunk_rows + valid_triple_rows
  )
  sentseq_cfg = SentSeqTrainConfig(
    sentence_model_name=cfg.sentence_model_name,
    truncate_dim=cfg.truncate_dim,
    text_prefix=cfg.text_prefix,
    max_seq_length=cfg.max_seq_length,
    encode_batch_size=cfg.encode_batch_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
    max_sents=cfg.max_sents,
    batch_size=cfg.batch_size,
    epochs=cfg.epochs,
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    seed=cfg.seed,
  )

  if precomputed_embeddings is not None:
    from train_pref_sentseq import collect_unique_sentences

    unique_sents = collect_unique_sentences(all_texts, max_sents=cfg.max_sents)
    missing = [s for s in unique_sents if s not in precomputed_embeddings]
    if missing:
      raise SystemExit(
        f"precomputed embeddings missing {len(missing)} sentences "
        f"(e.g. {missing[0][:40]!r})"
      )
    sent_to_embedding = {s: precomputed_embeddings[s] for s in unique_sents}
  else:
    sent_to_embedding = encode_unique_sentences(
      train_rows + valid_hunk_rows + valid_triple_rows,
      sentseq_cfg,
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
  )
  model = GpmRewardModel(
    encoder,
    feature_dim=feature_dim_for(cfg),
    head_dim=cfg.head_dim,
    head_hidden=cfg.head_hidden,
    normalize_embedding=cfg.normalize_embedding,
  ).to(device)
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
    f"head_dim={cfg.head_dim} epochs={cfg.epochs} bs={cfg.batch_size}",
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

      s_wl = model.score_pairs_from_doc_vectors(
        v_src,
        v_w,
        v_l,
        len_source=len_s,
        len_a=len_w,
        len_b=len_l,
      )
      loss = torch.nn.functional.softplus(-s_wl).mean()

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      epoch_loss += float(loss.item())
      n_batches += 1
      train_correct += int((s_wl > 0).sum().item())
      train_total += len(batch)

    last_train_metrics = {
      "train_gpm_loss": epoch_loss / max(n_batches, 1),
      "train_pair_accuracy": train_correct / max(train_total, 1),
    }
    valid_metrics = eval_gpm_validation(
      model,
      valid_hunk_rows,
      valid_triple_rows,
      prepared,
      device=device,
      batch_size=cfg.batch_size,
    )
    print(
      f"{log_prefix}epoch {epoch + 1}/{cfg.epochs} "
      f"train_loss={last_train_metrics['train_gpm_loss']:.4f} "
      f"train_acc={last_train_metrics['train_pair_accuracy']:.4f} "
      f"valid_acc={valid_metrics['valid_pair_accuracy']:.4f} "
      f"valid_cycles={int(valid_metrics['valid_section_cycles'])}",
      flush=True,
    )
    if valid_metrics["valid_pair_accuracy"] > best_valid_acc:
      best_valid_acc = valid_metrics["valid_pair_accuracy"]
      best_epoch = epoch + 1
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      best_train_metrics = dict(last_train_metrics)
      best_valid_metrics = dict(valid_metrics)

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()
  print(
    f"{log_prefix}best epoch: {best_epoch}/{cfg.epochs} "
    f"valid_acc={best_valid_acc:.4f}",
    flush=True,
  )
  best_train_metrics["best_epoch"] = float(best_epoch)
  prefixed_valid = {k: v for k, v in best_valid_metrics.items()}
  return model, best_train_metrics, prefixed_valid, prepared


def _load_optional_jsonl(path: str) -> list[dict]:
  if not str(path).strip():
    return []
  return load_jsonl(path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument(
    "--train-hunk-file",
    default="data/pref_keep_split_hunk/train.jsonl",
  )
  parser.add_argument(
    "--train-triple-file",
    default="data/section_middle/pref_train.jsonl",
  )
  parser.add_argument(
    "--valid-hunk-file",
    default="data/pref_keep_split_hunk/valid.jsonl",
  )
  parser.add_argument(
    "--valid-triple-file",
    default="data/section_middle/pref_valid.jsonl",
  )
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--head-dim", type=int, default=4)
  parser.add_argument("--head-hidden", type=int, default=0)
  parser.add_argument("--normalize-embedding", action="store_true")
  parser.add_argument("--drop-gen-over-draft", action="store_true")
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--num-layers", type=int, default=2)
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  args = parser.parse_args()

  validate_head_dim(args.head_dim)
  device = resolve_device(args.device)

  train_hunk = load_hunk_preference_rows(_load_optional_jsonl(args.train_hunk_file))
  train_triple_raw = _load_optional_jsonl(args.train_triple_file)
  train_rows = merge_training_rows(
    train_hunk,
    train_triple_raw,
    drop_gen_over_draft=args.drop_gen_over_draft,
  )
  valid_hunk = load_hunk_preference_rows(_load_optional_jsonl(args.valid_hunk_file))
  valid_triple = load_triple_preference_rows(
    _load_optional_jsonl(args.valid_triple_file),
    drop_gen_over_draft=False,
  )
  if not train_rows:
    raise SystemExit("training preference pairs are empty")
  if not valid_hunk and not valid_triple:
    raise SystemExit("validation preference pairs are empty")

  cfg = GpmTrainConfig(
    sentence_model_name=args.model,
    truncate_dim=normalize_truncate_dim(args.truncate_dim),
    text_prefix=args.text_prefix,
    max_seq_length=args.max_seq_length,
    encode_batch_size=args.encode_batch_size,
    d_model=args.d_model,
    num_layers=args.num_layers,
    max_sents=args.max_sents,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
    head_dim=args.head_dim,
    head_hidden=args.head_hidden,
    normalize_embedding=args.normalize_embedding,
  )

  model, train_metrics, valid_metrics, _prepared = train_gpm_model(
    train_rows,
    valid_hunk,
    valid_triple,
    cfg,
    device=device,
  )
  output_dir = Path(args.output_dir)
  save_gpm_model(model, cfg, embed_dim=model.encoder.embed_dim, output_dir=output_dir)

  metrics = {
    "embedding_model": cfg.sentence_model_name,
    "text_prefix": cfg.text_prefix,
    "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
    "max_sents": cfg.max_sents,
    "d_model": cfg.d_model,
    "num_layers": cfg.num_layers,
    "head_dim": cfg.head_dim,
    "head_hidden": cfg.head_hidden,
    "normalize_embedding": cfg.normalize_embedding,
    "drop_gen_over_draft": args.drop_gen_over_draft,
    "epochs": cfg.epochs,
    "lr": cfg.lr,
    "batch_size": cfg.batch_size,
    "train_pairs": len(train_rows),
    "valid_hunk_pairs": len(valid_hunk),
    "valid_triple_pairs": len(valid_triple),
    **train_metrics,
    **valid_metrics,
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"saved: {output_dir}")
  print(f"valid_pair_accuracy: {valid_metrics['valid_pair_accuracy']:.4f}")


if __name__ == "__main__":
  main()
