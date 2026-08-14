#!/usr/bin/env python3
"""凍結 ruri + source/K 候補の文 token joint Transformer setwise 順位モデルを、節三つ組みで学習する。"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from pref_static_utils import encode_text_map, load_jsonl, normalize_truncate_dim
from sentseq_utils import PARA_BOUNDARY_MODE, SENTSEQ_SPLIT_VERSION
from setwise_model import (
  ARTIFACT_KIND,
  LOSS_MODE_FULL_ORDER,
  LOSS_MODE_HUMAN_TOP,
  SetwiseRankingModel,
  artifact_kind_for_loss_mode,
  build_model_config,
  build_setwise_batch_tensors,
  checkpoint_selection_key,
  collect_unique_sentences,
  collect_unique_texts_for_triples,
  compute_setwise_metrics,
  compute_training_loss,
  load_setwise_model_from_artifact,
  normalize_loss_mode,
  rank_indices_from_roles,
  validate_loss_mode,
)
from setwise_triple_utils import (
  CANONICAL_ROLES,
  ROLE_COMPOSER,
  ROLE_DRAFT,
  ROLE_HUMAN,
  SetwiseTriple,
  reconstruct_triples_from_pref_rows,
)
from train_pref_sentseq import prepare_sentence_data, resolve_device


@dataclass
class SetwiseTrainConfig:
  sentence_model_name: str = "cl-nagoya/ruri-v3-30m"
  truncate_dim: int | None = None
  text_prefix: str = "文章: "
  max_seq_length: int = 256
  encode_batch_size: int = 64
  d_model: int = 256
  nhead: int = 4
  local_num_layers: int = 1
  joint_num_layers: int = 1
  dim_feedforward: int = 512
  dropout: float = 0.1
  max_sents: int = 128
  batch_size: int = 32
  epochs: int = 40
  lr: float = 3e-4
  weight_decay: float = 1e-2
  seed: int = 0
  loss_mode: str = LOSS_MODE_FULL_ORDER


def encode_unique_sentences(
  texts: list[str],
  cfg: SetwiseTrainConfig,
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


def _role_indices_for_batch(
  permuted_roles: list[list[str]],
  *,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  human_idx: list[int] = []
  composer_idx: list[int] = []
  draft_idx: list[int] = []
  rank_rows: list[list[int]] = []
  for roles in permuted_roles:
    role_to_idx = {role: idx for idx, role in enumerate(roles)}
    human_idx.append(role_to_idx[ROLE_HUMAN])
    composer_idx.append(role_to_idx[ROLE_COMPOSER])
    draft_idx.append(role_to_idx[ROLE_DRAFT])
    rank_rows.append(rank_indices_from_roles(roles, best_first_roles=CANONICAL_ROLES))
  return (
    torch.tensor(human_idx, dtype=torch.long, device=device),
    torch.tensor(composer_idx, dtype=torch.long, device=device),
    torch.tensor(draft_idx, dtype=torch.long, device=device),
    torch.tensor(rank_rows, dtype=torch.long, device=device),
  )


def _permute_triple_batch(
  batch: list[SetwiseTriple],
) -> tuple[list[str], list[list[str]], list[list[str]]]:
  sources: list[str] = []
  candidate_texts: list[list[str]] = []
  candidate_roles: list[list[str]] = []
  for triple in batch:
    roles = list(CANONICAL_ROLES)
    texts = triple.canonical_candidates()
    perm = list(range(3))
    random.shuffle(perm)
    sources.append(triple.source_text)
    candidate_texts.append([texts[i] for i in perm])
    candidate_roles.append([roles[i] for i in perm])
  return sources, candidate_texts, candidate_roles


@torch.no_grad()
def eval_setwise_triples(
  model: SetwiseRankingModel,
  triples: list[SetwiseTriple],
  prepared,
  *,
  device: torch.device,
  batch_size: int,
) -> dict[str, float]:
  model.eval()
  all_logits: list[torch.Tensor] = []
  all_rank: list[torch.Tensor] = []
  all_human: list[torch.Tensor] = []
  all_composer: list[torch.Tensor] = []
  all_draft: list[torch.Tensor] = []
  for start in range(0, len(triples), batch_size):
    batch = triples[start : start + batch_size]
    sources = [t.source_text for t in batch]
    candidate_texts = [t.canonical_candidates() for t in batch]
    candidate_roles = [list(CANONICAL_ROLES) for _ in batch]
    tensors = build_setwise_batch_tensors(
      source_texts=sources,
      candidate_texts=candidate_texts,
      prepared=prepared,
      embed_dim=model.embed_dim,
      device=device,
    )
    logits = model(
      tensors.source_sent_emb,
      tensors.source_para_start_mask,
      tensors.source_seq_len,
      tensors.candidate_sent_emb,
      tensors.candidate_para_start_mask,
      tensors.candidate_seq_len,
    )
    human_idx, composer_idx, draft_idx, rank_idx = _role_indices_for_batch(
      candidate_roles,
      device=device,
    )
    all_logits.append(logits)
    all_rank.append(rank_idx)
    all_human.append(human_idx)
    all_composer.append(composer_idx)
    all_draft.append(draft_idx)
  logits = torch.cat(all_logits, dim=0)
  rank_idx = torch.cat(all_rank, dim=0)
  human_idx = torch.cat(all_human, dim=0)
  composer_idx = torch.cat(all_composer, dim=0)
  draft_idx = torch.cat(all_draft, dim=0)
  return compute_setwise_metrics(
    logits,
    rank_idx,
    human_index=human_idx,
    composer_index=composer_idx,
    draft_index=draft_idx,
  )


def train_setwise_model(
  train_triples: list[SetwiseTriple],
  valid_triples: list[SetwiseTriple],
  cfg: SetwiseTrainConfig,
  *,
  device: torch.device,
  log_prefix: str = "",
  precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[SetwiseRankingModel, dict[str, float], dict[str, float], object]:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  all_texts = collect_unique_texts_for_triples(train_triples + valid_triples)
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
  model = SetwiseRankingModel(
    embed_dim=embed_dim,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    local_num_layers=cfg.local_num_layers,
    joint_num_layers=cfg.joint_num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
    max_sents=cfg.max_sents,
  ).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
  )

  n = len(train_triples)
  best_key: tuple[float, float, float, float] | None = None
  best_epoch = -1
  best_state: dict[str, torch.Tensor] | None = None
  best_train_metrics: dict[str, float] = {}
  best_valid_metrics: dict[str, float] = {}
  last_train_metrics: dict[str, float] = {}

  print(
    f"{log_prefix}train: triples={n} valid={len(valid_triples)} device={device.type} "
    f"embed_dim={embed_dim} unique_sents={len(unique_sents)} "
    f"epochs={cfg.epochs} bs={cfg.batch_size} loss_mode={cfg.loss_mode}",
    flush=True,
  )

  for epoch in range(cfg.epochs):
    model.train()
    order = list(range(n))
    random.shuffle(order)
    epoch_loss = 0.0
    n_batches = 0
    train_logits: list[torch.Tensor] = []
    train_rank: list[torch.Tensor] = []
    train_human: list[torch.Tensor] = []
    train_composer: list[torch.Tensor] = []
    train_draft: list[torch.Tensor] = []

    for start in range(0, n, cfg.batch_size):
      batch = [train_triples[i] for i in order[start : start + cfg.batch_size]]
      sources, candidate_texts, candidate_roles = _permute_triple_batch(batch)
      tensors = build_setwise_batch_tensors(
        source_texts=sources,
        candidate_texts=candidate_texts,
        prepared=prepared,
        embed_dim=embed_dim,
        device=device,
      )
      logits = model(
        tensors.source_sent_emb,
        tensors.source_para_start_mask,
        tensors.source_seq_len,
        tensors.candidate_sent_emb,
        tensors.candidate_para_start_mask,
        tensors.candidate_seq_len,
      )
      human_idx, composer_idx, draft_idx, rank_idx = _role_indices_for_batch(
        candidate_roles,
        device=device,
      )
      loss = compute_training_loss(
        logits,
        rank_idx,
        human_idx,
        loss_mode=cfg.loss_mode,
      )
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      epoch_loss += float(loss.item())
      n_batches += 1
      train_logits.append(logits.detach())
      train_rank.append(rank_idx)
      train_human.append(human_idx)
      train_composer.append(composer_idx)
      train_draft.append(draft_idx)

    train_logits_cat = torch.cat(train_logits, dim=0)
    train_rank_cat = torch.cat(train_rank, dim=0)
    train_metrics = compute_setwise_metrics(
      train_logits_cat,
      train_rank_cat,
      human_index=torch.cat(train_human, dim=0),
      composer_index=torch.cat(train_composer, dim=0),
      draft_index=torch.cat(train_draft, dim=0),
    )
    train_metrics["train_objective_loss"] = epoch_loss / max(n_batches, 1)
    if normalize_loss_mode(cfg.loss_mode) == LOSS_MODE_HUMAN_TOP:
      train_metrics["train_human_top_loss"] = train_metrics["train_objective_loss"]
    else:
      train_metrics["train_listwise_loss"] = train_metrics["train_objective_loss"]
    last_train_metrics = train_metrics
    valid_metrics = eval_setwise_triples(
      model,
      valid_triples,
      prepared,
      device=device,
      batch_size=cfg.batch_size,
    )
    objective_key = (
      "human_top_loss"
      if normalize_loss_mode(cfg.loss_mode) == LOSS_MODE_HUMAN_TOP
      else "listwise_loss"
    )
    print(
      f"{log_prefix}epoch {epoch + 1}/{cfg.epochs} "
      f"train_loss={train_metrics['train_objective_loss']:.4f} "
      f"valid_loss={valid_metrics[objective_key]:.4f} "
      f"valid_human_top1={valid_metrics['human_top1']:.4f}",
      flush=True,
    )
    key = checkpoint_selection_key(valid_metrics, loss_mode=cfg.loss_mode)
    if best_key is None or key > best_key:
      best_key = key
      best_epoch = epoch + 1
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      best_train_metrics = dict(train_metrics)
      best_valid_metrics = {f"valid_{k}": v for k, v in valid_metrics.items()}

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()
  print(
    f"{log_prefix}best epoch: {best_epoch}/{cfg.epochs} "
    f"valid_human_top1={best_valid_metrics.get('valid_human_top1', 0.0):.4f}",
    flush=True,
  )
  best_train_metrics["best_epoch"] = float(best_epoch)
  return model, best_train_metrics, best_valid_metrics, prepared


def save_setwise_model(
  model: SetwiseRankingModel,
  cfg: SetwiseTrainConfig,
  *,
  embed_dim: int,
  output_dir: Path,
) -> None:
  config = build_model_config(
    {
      "d_model": cfg.d_model,
      "nhead": cfg.nhead,
      "local_num_layers": cfg.local_num_layers,
      "joint_num_layers": cfg.joint_num_layers,
      "dim_feedforward": cfg.dim_feedforward,
      "dropout": cfg.dropout,
      "max_sents": cfg.max_sents,
      "sentence_model_name": cfg.sentence_model_name,
      "truncate_dim": normalize_truncate_dim(cfg.truncate_dim),
      "text_prefix": cfg.text_prefix,
      "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
      "normalize_embeddings": True,
      "sent_split_version": SENTSEQ_SPLIT_VERSION,
      "para_boundary_mode": PARA_BOUNDARY_MODE,
    },
    embed_dim=embed_dim,
    loss_mode=cfg.loss_mode,
  )
  artifact = {
    "kind": artifact_kind_for_loss_mode(cfg.loss_mode),
    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "config": config,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  torch.save(artifact, output_dir / "model.pt")
  (output_dir / "meta.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--train-file", required=True)
  parser.add_argument("--eval-file", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--local-num-layers", type=int, default=1)
  parser.add_argument("--joint-num-layers", type=int, default=1)
  parser.add_argument(
    "--num-layers",
    type=int,
    default=0,
    help="legacy: if >0, uses local=1 and joint=max(1, num_layers-1)",
  )
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--loss-mode",
    default=LOSS_MODE_FULL_ORDER,
    choices=[LOSS_MODE_FULL_ORDER, LOSS_MODE_HUMAN_TOP],
    help="full_order: human>composer>draft ListMLE; human_top: human above others only",
  )
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  args = parser.parse_args()

  device = resolve_device(args.device)
  train_triples = reconstruct_triples_from_pref_rows(load_jsonl(args.train_file))
  valid_triples = reconstruct_triples_from_pref_rows(load_jsonl(args.eval_file))
  if not train_triples or not valid_triples:
    raise SystemExit("train/eval triples are empty after reconstruction")

  local_num_layers = args.local_num_layers
  joint_num_layers = args.joint_num_layers
  if args.num_layers > 0:
    local_num_layers = 1
    joint_num_layers = max(1, args.num_layers - 1)

  cfg = SetwiseTrainConfig(
    sentence_model_name=args.model,
    truncate_dim=normalize_truncate_dim(args.truncate_dim),
    text_prefix=args.text_prefix,
    max_seq_length=args.max_seq_length,
    encode_batch_size=args.encode_batch_size,
    d_model=args.d_model,
    local_num_layers=local_num_layers,
    joint_num_layers=joint_num_layers,
    max_sents=args.max_sents,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
    loss_mode=validate_loss_mode(args.loss_mode),
  )
  model, train_metrics, valid_metrics, _prepared = train_setwise_model(
    train_triples,
    valid_triples,
    cfg,
    device=device,
  )
  output_dir = Path(args.output_dir)
  save_setwise_model(model, cfg, embed_dim=model.embed_dim, output_dir=output_dir)
  artifact_kind = artifact_kind_for_loss_mode(cfg.loss_mode)
  checkpoint_rule = (
    "strict human_top1, strict human_over_composer, strict human_over_draft, -human_top_loss"
    if normalize_loss_mode(cfg.loss_mode) == LOSS_MODE_HUMAN_TOP
    else "strict human_top1, strict human_over_composer, strict exact_order, -listwise_loss"
  )
  metrics = {
    "kind": artifact_kind,
    "loss_mode": cfg.loss_mode,
    "embedding_model": cfg.sentence_model_name,
    "text_prefix": cfg.text_prefix,
    "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
    "max_sents": cfg.max_sents,
    "d_model": cfg.d_model,
    "local_num_layers": cfg.local_num_layers,
    "joint_num_layers": cfg.joint_num_layers,
    "epochs": cfg.epochs,
    "lr": cfg.lr,
    "batch_size": cfg.batch_size,
    "train_triples": len(train_triples),
    "valid_triples": len(valid_triples),
    "checkpoint_rule": checkpoint_rule,
    **train_metrics,
    **valid_metrics,
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"saved: {output_dir}")
  print(f"valid_human_top1: {valid_metrics['valid_human_top1']:.4f}")
  objective_key = (
    "valid_human_top_loss"
    if normalize_loss_mode(cfg.loss_mode) == LOSS_MODE_HUMAN_TOP
    else "valid_listwise_loss"
  )
  print(f"{objective_key}: {valid_metrics[objective_key]:.4f}")


if __name__ == "__main__":
  main()
