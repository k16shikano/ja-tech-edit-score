#!/usr/bin/env python3
"""下書きと候補の対を、人間の推敲なら 1、下書きと Composer なら 0 として検出する。

点の計算は文列型と同じ（差、絶対差、コサイン、文字数）。
既定では Composer 対下書きの対は学習に入れない。
--composer-over-draft を付けると、検出に加えて Composer の点が下書きより高くなる項を足す。
ruri の重みは更新しない。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from pref_static_utils import encode_text_map, load_jsonl, normalize_truncate_dim
from sentseq_utils import PARA_BOUNDARY_MODE, SENTSEQ_SPLIT_VERSION
from setwise_triple_utils import reconstruct_triples_from_pref_rows
from train_pref_sentseq import (
  SentSeqEncoder,
  SentSeqRewardModel,
  SentSeqTrainConfig,
  collect_unique_sentences,
  encode_texts_to_doc_vectors,
  prepare_sentence_data,
  resolve_device,
)

KIND = "pref-detect"
ROLE_HUMAN = "human"
ROLE_DRAFT = "draft"
ROLE_COMPOSER = "composer"
COMPARE_EPS = 1e-6
FEATURE_VERSION = "source-cand-pointwise-v1"


def labeled_examples_from_triples(triples: list) -> list[dict]:
  """各三つ組みから、下書きを基準にした 3 件の検出例を出す。"""
  examples: list[dict] = []
  for triple in triples:
    examples.append(
      {
        "item_id": triple.item_id,
        "source_text": triple.draft,
        "candidate": triple.human,
        "label": 1.0,
        "role": ROLE_HUMAN,
      }
    )
    examples.append(
      {
        "item_id": triple.item_id,
        "source_text": triple.draft,
        "candidate": triple.draft,
        "label": 0.0,
        "role": ROLE_DRAFT,
      }
    )
    examples.append(
      {
        "item_id": triple.item_id,
        "source_text": triple.draft,
        "candidate": triple.composer,
        "label": 0.0,
        "role": ROLE_COMPOSER,
      }
    )
  return examples


def detect_objective(
  human: torch.Tensor,
  draft: torch.Tensor,
  composer: torch.Tensor,
  *,
  composer_over_draft: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """検出の損失。composer_over_draft なら Composer が下書きより上になる項を足す。"""
  logits = torch.stack([human, draft, composer], dim=1)
  labels = torch.tensor(
    [1.0, 0.0, 0.0], dtype=logits.dtype, device=logits.device
  ).expand(logits.shape[0], -1)
  bce = F.binary_cross_entropy_with_logits(logits, labels)
  if not composer_over_draft:
    return bce, bce, None
  cd = F.softplus(-(composer - draft)).mean()
  return bce + cd, bce, cd


def _strictly_greater(a: float, b: float) -> bool:
  return a > b + COMPARE_EPS


def collect_triple_texts(triples: list) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for triple in triples:
    for value in (triple.draft, triple.human, triple.composer):
      if value not in seen:
        seen.add(value)
        texts.append(value)
  return texts


def encode_corpus_sentences(
  texts: list[str],
  cfg: SentSeqTrainConfig,
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


def score_candidates(
  model: SentSeqRewardModel,
  sources: list[str],
  candidates: list[str],
  prepared,
  *,
  device: torch.device,
) -> torch.Tensor:
  v_src = encode_texts_to_doc_vectors(model, sources, prepared, device=device)
  v_cand = encode_texts_to_doc_vectors(model, candidates, prepared, device=device)
  len_s = torch.tensor([float(len(t)) for t in sources], dtype=torch.float32, device=device)
  len_c = torch.tensor(
    [float(len(t)) for t in candidates], dtype=torch.float32, device=device
  )
  return model.score_from_doc_vectors_fast(
    v_src, v_cand, len_source=len_s, len_candidate=len_c
  )


@torch.no_grad()
def eval_triples(
  model: SentSeqRewardModel,
  triples: list,
  prepared,
  *,
  device: torch.device,
  batch_size: int,
) -> dict:
  model.eval()
  examples = labeled_examples_from_triples(triples)
  losses: list[float] = []
  for start in range(0, len(examples), batch_size):
    batch = examples[start : start + batch_size]
    logits = score_candidates(
      model,
      [row["source_text"] for row in batch],
      [row["candidate"] for row in batch],
      prepared,
      device=device,
    )
    labels = torch.tensor(
      [row["label"] for row in batch], dtype=torch.float32, device=device
    )
    losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).item()))

  items: list[dict] = []
  for start in range(0, len(triples), batch_size):
    batch = triples[start : start + batch_size]
    sources = [t.draft for t in batch for _ in range(3)]
    candidates = [text for t in batch for text in (t.human, t.draft, t.composer)]
    scores = score_candidates(model, sources, candidates, prepared, device=device)
    scores = scores.detach().cpu()
    for i, triple in enumerate(batch):
      human_s = float(scores[i * 3 + 0])
      draft_s = float(scores[i * 3 + 1])
      composer_s = float(scores[i * 3 + 2])
      items.append(
        {
          "item_id": triple.item_id,
          "scores": {"human": human_s, "draft": draft_s, "composer": composer_s},
          "draft_over_human": _strictly_greater(draft_s, human_s),
          "composer_over_human": _strictly_greater(composer_s, human_s),
          "composer_over_draft": _strictly_greater(composer_s, draft_s),
        }
      )
  n = len(items)
  return {
    "bce_loss": float(sum(losses) / max(len(losses), 1)),
    "n": n,
    "n_draft_over_human": sum(1 for row in items if row["draft_over_human"]),
    "n_composer_over_human": sum(1 for row in items if row["composer_over_human"]),
    "n_composer_over_draft": sum(1 for row in items if row["composer_over_draft"]),
    "items": items,
  }


def feature_dim_for(cfg: SentSeqTrainConfig) -> int:
  return cfg.d_model * 4 + 5


def build_model_config(
  cfg: SentSeqTrainConfig,
  *,
  embed_dim: int,
  composer_over_draft: bool = False,
) -> dict:
  loss = (
    "bce_human_one_plus_composer_over_draft"
    if composer_over_draft
    else "bce_human_one_draft_composer_zero"
  )
  return {
    "kind": KIND,
    "d_model": cfg.d_model,
    "nhead": cfg.nhead,
    "num_layers": cfg.num_layers,
    "dim_feedforward": cfg.dim_feedforward,
    "dropout": cfg.dropout,
    "max_sents": cfg.max_sents,
    "embed_dim": embed_dim,
    "feature_dim": feature_dim_for(cfg),
    "sentence_model_name": cfg.sentence_model_name,
    "truncate_dim": normalize_truncate_dim(cfg.truncate_dim),
    "text_prefix": cfg.text_prefix,
    "max_seq_length": cfg.max_seq_length if cfg.max_seq_length > 0 else None,
    "normalize_embeddings": True,
    "sent_split_version": SENTSEQ_SPLIT_VERSION,
    "para_boundary_mode": PARA_BOUNDARY_MODE,
    "feature_version": FEATURE_VERSION,
    "loss": loss,
    "composer_over_draft": composer_over_draft,
  }


def save_detect_model(
  model: SentSeqRewardModel,
  cfg: SentSeqTrainConfig,
  *,
  embed_dim: int,
  output_dir: Path,
  composer_over_draft: bool = False,
) -> None:
  config = build_model_config(
    cfg, embed_dim=embed_dim, composer_over_draft=composer_over_draft
  )
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


def load_detect_model_from_artifact(artifact: dict, *, device: torch.device) -> SentSeqRewardModel:
  config = artifact["config"]
  if config.get("kind") != KIND:
    raise ValueError("not a pref-detect artifact")
  encoder = SentSeqEncoder(
    embed_dim=int(config["embed_dim"]),
    d_model=int(config["d_model"]),
    nhead=int(config["nhead"]),
    num_layers=int(config["num_layers"]),
    dim_feedforward=int(config["dim_feedforward"]),
    dropout=float(config["dropout"]),
    max_sents=int(config["max_sents"]),
  )
  model = SentSeqRewardModel(encoder, feature_dim=int(config["feature_dim"]))
  model.load_state_dict(artifact["model_state_dict"])
  model.to(device)
  model.eval()
  return model


def train_detect_model(
  train_triples: list,
  valid_triples: list,
  cfg: SentSeqTrainConfig,
  *,
  device: torch.device,
  precomputed_embeddings: dict[str, np.ndarray] | None = None,
  log_prefix: str = "",
  composer_over_draft: bool = False,
) -> tuple[SentSeqRewardModel, dict, dict]:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  train_examples = labeled_examples_from_triples(train_triples)
  n_train = len(train_triples) if composer_over_draft else len(train_examples)
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
  model = SentSeqRewardModel(encoder, feature_dim=feature_dim_for(cfg)).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

  n = n_train
  best_train_loss = float("inf")
  best_epoch = -1
  best_state: dict[str, torch.Tensor] | None = None
  best_train_metrics: dict = {}

  print(
    f"{log_prefix}train: triples={len(train_triples)} examples={len(train_examples)} "
    f"valid={len(valid_triples)} device={device.type} embed_dim={embed_dim} "
    f"epochs={cfg.epochs} bs={cfg.batch_size} composer_over_draft={composer_over_draft}",
    flush=True,
  )

  for epoch in range(cfg.epochs):
    model.train()
    order = list(range(n))
    random.shuffle(order)
    epoch_loss = 0.0
    epoch_bce = 0.0
    n_batches = 0
    for start in range(0, n, cfg.batch_size):
      optimizer.zero_grad()
      if composer_over_draft:
        batch_triples = [train_triples[i] for i in order[start : start + cfg.batch_size]]
        sources = [t.draft for t in batch_triples for _ in range(3)]
        candidates = [text for t in batch_triples for text in (t.human, t.draft, t.composer)]
        scores = score_candidates(model, sources, candidates, prepared, device=device)
        scores = scores.view(len(batch_triples), 3)
        loss, bce, _cd = detect_objective(
          scores[:, 0],
          scores[:, 1],
          scores[:, 2],
          composer_over_draft=True,
        )
        epoch_bce += float(bce.item())
      else:
        batch = [train_examples[i] for i in order[start : start + cfg.batch_size]]
        logits = score_candidates(
          model,
          [row["source_text"] for row in batch],
          [row["candidate"] for row in batch],
          prepared,
          device=device,
        )
        labels = torch.tensor(
          [row["label"] for row in batch], dtype=torch.float32, device=device
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        epoch_bce += float(loss.item())
      loss.backward()
      optimizer.step()
      epoch_loss += float(loss.item())
      n_batches += 1
    train_loss = epoch_loss / max(n_batches, 1)
    train_bce = epoch_bce / max(n_batches, 1)
    last_valid = eval_triples(
      model,
      valid_triples,
      prepared,
      device=device,
      batch_size=cfg.batch_size,
    )
    print(
      f"{log_prefix}epoch {epoch + 1}/{cfg.epochs} "
      f"train_loss={train_loss:.4f} "
      f"train_bce={train_bce:.4f} "
      f"valid_bce={last_valid['bce_loss']:.4f} "
      f"valid_draft_over_human={last_valid['n_draft_over_human']}/{last_valid['n']} "
      f"valid_composer_over_human={last_valid['n_composer_over_human']}/{last_valid['n']}",
      flush=True,
    )
    if train_loss < best_train_loss:
      best_train_loss = train_loss
      best_epoch = epoch + 1
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      best_train_metrics = {
        "train_loss": train_loss,
        "train_bce_loss": train_bce,
        "best_epoch": float(best_epoch),
        "composer_over_draft": composer_over_draft,
      }

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()
  print(
    f"{log_prefix}best epoch: {best_epoch}/{cfg.epochs} train_loss={best_train_loss:.4f}",
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
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  parser.add_argument("--max-train", type=int, default=0)
  parser.add_argument("--max-valid", type=int, default=0)
  parser.add_argument(
    "--composer-over-draft",
    action="store_true",
    help="検出に加えて、Composer の点が下書きより高くなる項を足す",
  )
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
  cfg = SentSeqTrainConfig(
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
  )
  model, train_metrics, valid_metrics = train_detect_model(
    train_triples,
    valid_triples,
    cfg,
    device=device,
    composer_over_draft=args.composer_over_draft,
  )
  output_dir = Path(args.output_dir)
  save_detect_model(
    model,
    cfg,
    embed_dim=model.encoder.embed_dim,
    output_dir=output_dir,
    composer_over_draft=args.composer_over_draft,
  )
  items = valid_metrics.pop("items")
  report = {
    **train_metrics,
    "valid_bce_loss": valid_metrics["bce_loss"],
    "valid_n": valid_metrics["n"],
    "valid_n_draft_over_human": valid_metrics["n_draft_over_human"],
    "valid_n_composer_over_human": valid_metrics["n_composer_over_human"],
    "valid_n_composer_over_draft": valid_metrics["n_composer_over_draft"],
    "checkpoint_rule": "lowest train loss",
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
