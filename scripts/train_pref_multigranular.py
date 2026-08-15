#!/usr/bin/env python3
"""一対採点の文列モデルを、段階 2 から 6 の設定で学ぶ。"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pref_pair_utils import (
  PairExample,
  examples_from_hunk_rows,
  examples_from_triples,
  unique_texts_from_examples,
)
from pref_static_utils import load_jsonl, normalize_truncate_dim
from sentseq_utils import PARA_BOUNDARY_MODE, SENTSEQ_SPLIT_VERSION
from setwise_model import (
  ARTIFACT_KIND_JOINT_HUNK_REPLAY,
  ARTIFACT_KIND_PAIR_INDEPENDENT,
  LOSS_MODE_HUMAN_TOP,
  SCORE_MODE_INDEPENDENT,
  SCORE_MODE_JOINT,
  SetwiseRankingModel,
  build_model_config,
  build_setwise_batch_tensors,
  candidate_logits,
  collect_unique_sentences,
  compute_setwise_metrics,
  compute_training_loss,
  normalize_score_mode,
  pair_checkpoint_selection_key,
  rank_indices_from_roles,
)
from setwise_triple_utils import (
  CANONICAL_ROLES,
  ROLE_COMPOSER,
  ROLE_DRAFT,
  ROLE_HUMAN,
  reconstruct_triples_from_pref_rows,
)
from train_pref_sentseq import prepare_sentence_data, resolve_device
from train_pref_setwise import SetwiseTrainConfig, encode_unique_sentences


STAGE_PAIR_DRAFT = "pair_draft"
STAGE_PAIR_HUMANTOP = "pair_humantop"
STAGE_PAIR_HUNK_THEN_SECTION = "pair_humantop_hunk_then_section"
STAGE_PAIR_HUNK_REPLAY = "pair_humantop_hunk_replay"
STAGE_JOINT_HUNK_REPLAY = "joint_humantop_hunk_replay"
TRAIN_STAGES = (
  STAGE_PAIR_DRAFT,
  STAGE_PAIR_HUMANTOP,
  STAGE_PAIR_HUNK_THEN_SECTION,
  STAGE_PAIR_HUNK_REPLAY,
  STAGE_JOINT_HUNK_REPLAY,
)


def stage_score_mode(stage: str) -> str:
  if stage == STAGE_JOINT_HUNK_REPLAY:
    return SCORE_MODE_JOINT
  return SCORE_MODE_INDEPENDENT


def stage_include_composer(stage: str) -> bool:
  return stage != STAGE_PAIR_DRAFT


def stage_uses_hunk(stage: str) -> bool:
  return stage in {
    STAGE_PAIR_HUNK_THEN_SECTION,
    STAGE_PAIR_HUNK_REPLAY,
    STAGE_JOINT_HUNK_REPLAY,
  }


def stage_replay_hunk(stage: str) -> bool:
  return stage in {STAGE_PAIR_HUNK_REPLAY, STAGE_JOINT_HUNK_REPLAY}


def stage_artifact_kind(stage: str) -> str:
  if stage == STAGE_JOINT_HUNK_REPLAY:
    return ARTIFACT_KIND_JOINT_HUNK_REPLAY
  return ARTIFACT_KIND_PAIR_INDEPENDENT


def _role_tensors(
  roles_batch: list[list[str]],
  *,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  human_idx: list[int] = []
  composer_idx: list[int] = []
  draft_idx: list[int] = []
  rank_rows: list[list[int]] = []
  for roles in roles_batch:
    role_to_idx = {role: idx for idx, role in enumerate(roles)}
    human_idx.append(role_to_idx[ROLE_HUMAN])
    composer_idx.append(role_to_idx.get(ROLE_COMPOSER, role_to_idx[ROLE_HUMAN]))
    draft_idx.append(role_to_idx.get(ROLE_DRAFT, role_to_idx[ROLE_HUMAN]))
    if ROLE_COMPOSER in role_to_idx and ROLE_DRAFT in role_to_idx:
      rank_rows.append(rank_indices_from_roles(roles, best_first_roles=CANONICAL_ROLES))
    else:
      rank_rows.append(
        rank_indices_from_roles(roles, best_first_roles=(ROLE_HUMAN, ROLE_DRAFT))
      )
  return (
    torch.tensor(human_idx, dtype=torch.long, device=device),
    torch.tensor(composer_idx, dtype=torch.long, device=device),
    torch.tensor(draft_idx, dtype=torch.long, device=device),
    torch.tensor(rank_rows, dtype=torch.long, device=device),
  )


def _permute_examples(batch: list[PairExample]) -> tuple[list[str], list[list[str]], list[list[str]]]:
  sources: list[str] = []
  candidate_texts: list[list[str]] = []
  candidate_roles: list[list[str]] = []
  for ex in batch:
    perm = list(range(len(ex.candidates)))
    random.shuffle(perm)
    sources.append(ex.source_text)
    candidate_texts.append([ex.candidates[i] for i in perm])
    candidate_roles.append([ex.roles[i] for i in perm])
  return sources, candidate_texts, candidate_roles


def _upsample_to_len(items: list, n: int, rng: random.Random) -> list:
  if not items:
    raise ValueError("cannot upsample an empty list")
  if len(items) >= n:
    out = list(items)
    rng.shuffle(out)
    return out[:n]
  reps = list(items)
  while len(reps) < n:
    extra = list(items)
    rng.shuffle(extra)
    reps.extend(extra)
  rng.shuffle(reps)
  return reps[:n]


def _interleaved_equal_batches(
  section_examples: list[PairExample],
  hunk_examples: list[PairExample],
  *,
  batch_size: int,
  rng: random.Random,
) -> list[list[PairExample]]:
  n_each = max(len(section_examples), len(hunk_examples))
  section = _upsample_to_len(section_examples, n_each, rng)
  hunk = _upsample_to_len(hunk_examples, n_each, rng)
  batches: list[list[PairExample]] = []
  for start in range(0, n_each, batch_size):
    batches.append(section[start : start + batch_size])
    batches.append(hunk[start : start + batch_size])
  return batches


@torch.no_grad()
def eval_examples(
  model: SetwiseRankingModel,
  examples: list[PairExample],
  prepared,
  *,
  device: torch.device,
  batch_size: int,
  score_mode: str,
) -> dict[str, float]:
  model.eval()
  if not examples:
    raise ValueError("eval examples must not be empty")
  all_logits: list[torch.Tensor] = []
  all_rank: list[torch.Tensor] = []
  all_human: list[torch.Tensor] = []
  all_composer: list[torch.Tensor] = []
  all_draft: list[torch.Tensor] = []
  for start in range(0, len(examples), batch_size):
    batch = examples[start : start + batch_size]
    sources = [ex.source_text for ex in batch]
    candidate_texts = [list(ex.candidates) for ex in batch]
    candidate_roles = [list(ex.roles) for ex in batch]
    tensors = build_setwise_batch_tensors(
      source_texts=sources,
      candidate_texts=candidate_texts,
      prepared=prepared,
      embed_dim=model.embed_dim,
      device=device,
    )
    logits = candidate_logits(model, tensors, score_mode=score_mode)
    human_idx, composer_idx, draft_idx, rank_idx = _role_tensors(
      candidate_roles,
      device=device,
    )
    all_logits.append(logits)
    all_rank.append(rank_idx)
    all_human.append(human_idx)
    all_composer.append(composer_idx)
    all_draft.append(draft_idx)
  return compute_setwise_metrics(
    torch.cat(all_logits, dim=0),
    torch.cat(all_rank, dim=0),
    human_index=torch.cat(all_human, dim=0),
    composer_index=torch.cat(all_composer, dim=0),
    draft_index=torch.cat(all_draft, dim=0),
  )


def _run_epoch(
  model: SetwiseRankingModel,
  batches: list[list[PairExample]],
  prepared,
  optimizer: torch.optim.Optimizer,
  *,
  device: torch.device,
  score_mode: str,
) -> dict[str, float]:
  model.train()
  epoch_loss = 0.0
  n_batches = 0
  train_logits: list[torch.Tensor] = []
  train_rank: list[torch.Tensor] = []
  train_human: list[torch.Tensor] = []
  train_composer: list[torch.Tensor] = []
  train_draft: list[torch.Tensor] = []
  for batch in batches:
    if not batch:
      continue
    sources, candidate_texts, candidate_roles = _permute_examples(batch)
    tensors = build_setwise_batch_tensors(
      source_texts=sources,
      candidate_texts=candidate_texts,
      prepared=prepared,
      embed_dim=model.embed_dim,
      device=device,
    )
    logits = candidate_logits(model, tensors, score_mode=score_mode)
    human_idx, composer_idx, draft_idx, rank_idx = _role_tensors(
      candidate_roles,
      device=device,
    )
    loss = compute_training_loss(
      logits,
      rank_idx,
      human_idx,
      loss_mode=LOSS_MODE_HUMAN_TOP,
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
  if train_logits:
    by_k: dict[int, dict[str, list[torch.Tensor]]] = {}
    for logits, rank, human, composer, draft in zip(
      train_logits,
      train_rank,
      train_human,
      train_composer,
      train_draft,
    ):
      k = logits.shape[-1]
      bucket = by_k.setdefault(
        k,
        {"logits": [], "rank": [], "human": [], "composer": [], "draft": []},
      )
      bucket["logits"].append(logits)
      bucket["rank"].append(rank)
      bucket["human"].append(human)
      bucket["composer"].append(composer)
      bucket["draft"].append(draft)
    pick_k = 3 if 3 in by_k else next(iter(by_k))
    picked = by_k[pick_k]
    metrics = compute_setwise_metrics(
      torch.cat(picked["logits"], dim=0),
      torch.cat(picked["rank"], dim=0),
      human_index=torch.cat(picked["human"], dim=0),
      composer_index=torch.cat(picked["composer"], dim=0),
      draft_index=torch.cat(picked["draft"], dim=0),
    )
  else:
    metrics = {}
  metrics["train_objective_loss"] = epoch_loss / max(n_batches, 1)
  metrics["train_human_top_loss"] = metrics["train_objective_loss"]
  return metrics


def _example_batches(
  examples: list[PairExample],
  *,
  batch_size: int,
  rng: random.Random,
) -> list[list[PairExample]]:
  order = list(examples)
  rng.shuffle(order)
  return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def train_multigranular_model(
  section_train: list[PairExample],
  hunk_train: list[PairExample],
  valid_examples: list[PairExample],
  cfg: SetwiseTrainConfig,
  *,
  device: torch.device,
  score_mode: str,
  phase1_epochs: int,
  phase2_epochs: int,
  replay_hunk: bool,
  log_prefix: str = "",
  precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[SetwiseRankingModel, dict[str, float], dict[str, float]]:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)
  rng = random.Random(cfg.seed)

  all_examples = section_train + hunk_train + valid_examples
  all_texts = unique_texts_from_examples(all_examples)
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

  phases: list[tuple[str, int, str]] = []
  if phase1_epochs > 0:
    if not hunk_train:
      raise ValueError("phase1_epochs>0 requires hunk examples")
    phases.append(("hunk", phase1_epochs, "phase1"))
  if phase2_epochs > 0:
    phases.append(("section", phase2_epochs, "phase2"))

  best_key: tuple[float, float, float] | None = None
  best_epoch = -1
  best_state: dict[str, torch.Tensor] | None = None
  best_train_metrics: dict[str, float] = {}
  best_valid_metrics: dict[str, float] = {}
  global_epoch = 0

  print(
    f"{log_prefix}train: section={len(section_train)} hunk={len(hunk_train)} "
    f"valid={len(valid_examples)} score_mode={score_mode} "
    f"phase1={phase1_epochs} phase2={phase2_epochs} replay_hunk={replay_hunk} "
    f"device={device.type} embed_dim={embed_dim}",
    flush=True,
  )

  for phase_kind, n_epochs, phase_name in phases:
    for _ in range(n_epochs):
      global_epoch += 1
      if phase_kind == "hunk":
        batches = _example_batches(hunk_train, batch_size=cfg.batch_size, rng=rng)
      elif replay_hunk:
        batches = _interleaved_equal_batches(
          section_train,
          hunk_train,
          batch_size=cfg.batch_size,
          rng=rng,
        )
      else:
        batches = _example_batches(section_train, batch_size=cfg.batch_size, rng=rng)
      train_metrics = _run_epoch(
        model,
        batches,
        prepared,
        optimizer,
        device=device,
        score_mode=score_mode,
      )
      valid_metrics = eval_examples(
        model,
        valid_examples,
        prepared,
        device=device,
        batch_size=cfg.batch_size,
        score_mode=score_mode,
      )
      print(
        f"{log_prefix}{phase_name} epoch {global_epoch} "
        f"train_loss={train_metrics['train_objective_loss']:.4f} "
        f"valid_human_among_top={valid_metrics['human_among_top']:.4f} "
        f"valid_human_over_composer={valid_metrics['human_over_composer']:.4f} "
        f"valid_human_over_draft={valid_metrics['human_over_draft']:.4f}",
        flush=True,
      )
      key = pair_checkpoint_selection_key(valid_metrics)
      if best_key is None or key > best_key:
        best_key = key
        best_epoch = global_epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_train_metrics = dict(train_metrics)
        best_valid_metrics = {f"valid_{k}": v for k, v in valid_metrics.items()}

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()
  best_train_metrics["best_epoch"] = float(best_epoch)
  print(
    f"{log_prefix}best epoch: {best_epoch} "
    f"valid_human_among_top={best_valid_metrics.get('valid_human_among_top', 0.0):.4f}",
    flush=True,
  )
  return model, best_train_metrics, best_valid_metrics


def save_multigranular_model(
  model: SetwiseRankingModel,
  cfg: SetwiseTrainConfig,
  *,
  embed_dim: int,
  output_dir: Path,
  stage: str,
  score_mode: str,
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
    loss_mode=LOSS_MODE_HUMAN_TOP,
    score_mode=score_mode,
    kind=stage_artifact_kind(stage),
  )
  config["stage"] = stage
  artifact = {
    "kind": config["kind"],
    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "config": config,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  torch.save(artifact, output_dir / "model.pt")
  (output_dir / "meta.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def _limit(items: list, n: int) -> list:
  if n <= 0:
    return items
  return items[:n]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", required=True, choices=TRAIN_STAGES)
  parser.add_argument("--section-train-file", default="data/section_middle/pref_train.jsonl")
  parser.add_argument("--section-eval-file", default="data/section_middle/pref_valid.jsonl")
  parser.add_argument("--hunk-train-file", default="data/pref_keep_split_hunk/train.jsonl")
  parser.add_argument("--hunk-eval-file", default="data/pref_keep_split_hunk/valid.jsonl")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--local-num-layers", type=int, default=1)
  parser.add_argument("--joint-num-layers", type=int, default=1)
  parser.add_argument("--num-layers", type=int, default=0)
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--phase1-epochs", type=int, default=-1)
  parser.add_argument("--phase2-epochs", type=int, default=-1)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  parser.add_argument("--max-train", type=int, default=0)
  parser.add_argument("--max-valid", type=int, default=0)
  args = parser.parse_args()

  local_layers = args.local_num_layers
  joint_layers = args.joint_num_layers
  if args.num_layers > 0:
    local_layers = 1
    joint_layers = max(1, args.num_layers - 1)

  stage = args.stage
  score_mode = stage_score_mode(stage)
  include_composer = stage_include_composer(stage)
  uses_hunk = stage_uses_hunk(stage)
  replay = stage_replay_hunk(stage)
  if args.phase1_epochs >= 0:
    phase1_epochs = args.phase1_epochs
  else:
    phase1_epochs = 10 if uses_hunk else 0
  if args.phase2_epochs >= 0:
    phase2_epochs = args.phase2_epochs
  else:
    phase2_epochs = args.epochs
  if not uses_hunk:
    phase1_epochs = 0

  train_triples = reconstruct_triples_from_pref_rows(load_jsonl(args.section_train_file))
  valid_triples = reconstruct_triples_from_pref_rows(load_jsonl(args.section_eval_file))
  section_train = _limit(
    examples_from_triples(train_triples, include_composer=include_composer),
    args.max_train,
  )
  valid_examples = _limit(
    examples_from_triples(valid_triples, include_composer=True),
    args.max_valid,
  )
  hunk_train: list[PairExample] = []
  if uses_hunk:
    hunk_train = _limit(examples_from_hunk_rows(load_jsonl(args.hunk_train_file)), args.max_train)

  cfg = SetwiseTrainConfig(
    sentence_model_name=args.model,
    truncate_dim=normalize_truncate_dim(args.truncate_dim),
    text_prefix=args.text_prefix,
    max_seq_length=args.max_seq_length,
    encode_batch_size=args.encode_batch_size,
    d_model=args.d_model,
    local_num_layers=local_layers,
    joint_num_layers=joint_layers,
    max_sents=args.max_sents,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
    loss_mode=LOSS_MODE_HUMAN_TOP,
  )
  device = resolve_device(args.device)
  model, train_metrics, valid_metrics = train_multigranular_model(
    section_train,
    hunk_train,
    valid_examples,
    cfg,
    device=device,
    score_mode=score_mode,
    phase1_epochs=phase1_epochs,
    phase2_epochs=phase2_epochs,
    replay_hunk=replay,
    log_prefix="",
  )
  save_multigranular_model(
    model,
    cfg,
    embed_dim=model.embed_dim,
    output_dir=Path(args.output_dir),
    stage=stage,
    score_mode=normalize_score_mode(score_mode),
  )
  metrics_path = Path(args.output_dir) / "metrics.json"
  metrics_path.write_text(
    json.dumps(
      {
        "stage": stage,
        "score_mode": score_mode,
        "seed": args.seed,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "replay_hunk": replay,
        **train_metrics,
        **valid_metrics,
      },
      ensure_ascii=False,
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  print(f"saved {args.output_dir}", flush=True)


if __name__ == "__main__":
  main()
