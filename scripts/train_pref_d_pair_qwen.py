#!/usr/bin/env python3
"""D ペア台帳で Qwen3-8B を BT または GPM として 1 fold 学習する。"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pref_d_causal_reward import (
  CausalRewardConfig,
  build_causal_reward_model,
  forward_bt_pair_deltas,
  forward_gpm_pair_vectors,
  save_causal_reward_model,
  predict_f_delta,
  predict_vectors,
)
from pref_d_interval_utils import load_d_rows
from pref_d_pair_utils import (
  filter_ledger_by_items,
  finite_scalar,
  group_ledger_by_item,
  group_rows_by_item,
  pairwise_bt_loss,
  pairwise_gpm_loss,
  rank_pair_metrics_gpm,
  rank_pair_metrics_scalar,
)


@dataclass
class TrainConfig:
  mode: str
  base_model: str
  train_file: Path
  valid_file: Path
  ledger_file: Path
  output_dir: Path
  fold: int
  epochs: int
  head_dim: int = 32
  batch_items: int = 8
  pair_batch_size: int = 2
  learning_rate: float = 1e-4
  weight_decay: float = 0.01
  warmup_ratio: float = 0.1
  max_length: int = 2048
  seed: int = 0
  lora_r: int = 16
  lora_alpha: int = 32
  base_quantization: str = "8bit"


def resolve_device() -> torch.device:
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for pref-d-pair-qwen training")
  return torch.device("cuda")


def item_ids_from_rows(rows: list[dict]) -> set[str]:
  return {str(r["item_id"]) for r in rows}


@torch.no_grad()
def validate_bt(model, tokenizer, valid_rows, *, cfg: TrainConfig, device) -> tuple[dict, list[dict]]:
  f_values = predict_f_delta(
    model,
    tokenizer,
    [r["draft"] for r in valid_rows],
    [r["y"] for r in valid_rows],
    max_length=cfg.max_length,
    device=device,
    batch_size=cfg.pair_batch_size,
  )
  pred_rows = []
  scores_by_row_id: dict[str, float | None] = {}
  for row, f in zip(valid_rows, f_values, strict=True):
    score = float(f) if finite_scalar(float(f)) else None
    scores_by_row_id[row["row_id"]] = score
    pred_rows.append({**row, "delta": score})
  metrics = rank_pair_metrics_scalar(group_rows_by_item(valid_rows), scores_by_row_id)
  return metrics, pred_rows


@torch.no_grad()
def validate_gpm(model, tokenizer, valid_rows, *, cfg: TrainConfig, device) -> tuple[dict, list[dict]]:
  vectors = predict_vectors(
    model,
    tokenizer,
    [r["draft"] for r in valid_rows],
    [r["y"] for r in valid_rows],
    max_length=cfg.max_length,
    device=device,
    batch_size=cfg.pair_batch_size,
  )
  pred_rows = []
  vectors_by_row_id: dict[str, list[float] | None] = {}
  for row, vec in zip(valid_rows, vectors, strict=True):
    if isinstance(vec, float):
      vec_list = [vec]
    else:
      vec_list = [float(x) for x in vec]
    ok = len(vec_list) == cfg.head_dim and all(math.isfinite(x) for x in vec_list)
    vectors_by_row_id[row["row_id"]] = vec_list if ok else None
    pred_rows.append({**row, "vector": vec_list if ok else None})
  metrics = rank_pair_metrics_gpm(group_rows_by_item(valid_rows), vectors_by_row_id)
  return metrics, pred_rows


def train_one_mode(cfg: TrainConfig) -> dict:
  from transformers import get_cosine_schedule_with_warmup

  device = resolve_device()
  train_rows = load_d_rows(cfg.train_file)
  valid_rows = load_d_rows(cfg.valid_file)
  train_ids = item_ids_from_rows(train_rows)
  valid_ids = item_ids_from_rows(valid_rows)

  from pref_d_pair_utils import load_pair_ledger

  all_ledger = load_pair_ledger(cfg.ledger_file)
  train_ledger = filter_ledger_by_items(all_ledger, train_ids)
  valid_ledger = filter_ledger_by_items(all_ledger, valid_ids)
  pairs_by_item = group_ledger_by_item(train_ledger)
  train_item_ids = sorted(pairs_by_item)

  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  head_dim = 1 if cfg.mode == "bt" else cfg.head_dim
  reward_cfg = CausalRewardConfig(
    base_model=cfg.base_model,
    max_length=cfg.max_length,
    head_dim=head_dim,
    lora_r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    base_quantization=cfg.base_quantization,
  )
  model, tokenizer = build_causal_reward_model(reward_cfg)
  model.train()

  trainable = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
  steps_per_epoch = max(1, math.ceil(len(train_item_ids) / cfg.batch_items))
  total_steps = steps_per_epoch * cfg.epochs
  scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * cfg.warmup_ratio),
    num_training_steps=total_steps,
  )
  autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

  log_path = cfg.output_dir / "train_log.jsonl"
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  best_a_micro = -1.0
  best_epoch = 0
  rng = random.Random(cfg.seed)

  for epoch in range(1, cfg.epochs + 1):
    order = list(train_item_ids)
    rng.shuffle(order)
    epoch_loss = 0.0
    n_batches = 0
    model.train()
    for start in range(0, len(order), cfg.batch_items):
      batch_item_ids = order[start : start + cfg.batch_items]
      if not batch_item_ids:
        continue

      optimizer.zero_grad(set_to_none=True)
      batch_loss = 0.0
      n_items = 0
      for item_id in batch_item_ids:
        pairs = pairs_by_item.get(item_id, [])
        if not pairs:
          continue
        chunk_losses: list[torch.Tensor] = []
        for chunk_start in range(0, len(pairs), cfg.pair_batch_size):
          chunk = pairs[chunk_start : chunk_start + cfg.pair_batch_size]
          drafts_w = [p["draft"] for p in chunk]
          ys_w = [p["winner_y"] for p in chunk]
          ys_l = [p["loser_y"] for p in chunk]

          with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            if cfg.mode == "bt":
              delta_w, delta_l = forward_bt_pair_deltas(
                model,
                tokenizer,
                drafts_w,
                ys_w,
                ys_l,
                max_length=cfg.max_length,
                device=device,
              )
              losses = pairwise_bt_loss(delta_w, delta_l)
            else:
              vec_w, vec_l = forward_gpm_pair_vectors(
                model,
                tokenizer,
                drafts_w,
                ys_w,
                ys_l,
                max_length=cfg.max_length,
                device=device,
              )
              losses = pairwise_gpm_loss(vec_w, vec_l)
          chunk_losses.extend(list(losses))

        item_loss = torch.stack(chunk_losses).mean() / len(batch_item_ids)
        item_loss.backward()
        batch_loss += float(item_loss.item()) * len(batch_item_ids)
        n_items += 1

      if n_items == 0:
        continue
      torch.nn.utils.clip_grad_norm_(trainable, 1.0)
      optimizer.step()
      scheduler.step()
      epoch_loss += batch_loss / n_items
      n_batches += 1

    model.eval()
    if cfg.mode == "bt":
      valid_metrics, pred_rows = validate_bt(model, tokenizer, valid_rows, cfg=cfg, device=device)
    else:
      valid_metrics, pred_rows = validate_gpm(model, tokenizer, valid_rows, cfg=cfg, device=device)

    a_micro = float(valid_metrics.get("A_micro") or 0.0)
    row = {
      "epoch": epoch,
      "train_loss": epoch_loss / max(n_batches, 1),
      "valid_A_micro": valid_metrics.get("A_micro"),
      "valid_A_macro": valid_metrics.get("A_macro"),
      "valid_evaluated_pairs": valid_metrics.get("evaluated_pairs_total"),
    }
    with log_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)

    if valid_metrics.get("A_micro") is not None and a_micro > best_a_micro:
      best_a_micro = a_micro
      best_epoch = epoch
      meta = {
        "kind": f"pref-d-{cfg.mode}-qwen",
        "mode": cfg.mode,
        "backend": f"causal_lora_{cfg.base_quantization}",
        "base_model": cfg.base_model,
        "base_quantization": cfg.base_quantization,
        "max_length": cfg.max_length,
        "num_labels": head_dim,
        "head_dim": cfg.head_dim if cfg.mode == "gpm" else 1,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "fold": cfg.fold,
        "best_epoch": best_epoch,
        "seed": cfg.seed,
      }
      save_causal_reward_model(model, tokenizer, cfg.output_dir / "best", meta=meta)
      (cfg.output_dir / "best_valid_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in pred_rows) + ("\n" if pred_rows else ""),
        encoding="utf-8",
      )

  summary = {
    "mode": cfg.mode,
    "backend": f"causal_lora_{cfg.base_quantization}",
    "fold": cfg.fold,
    "epochs": cfg.epochs,
    "best_epoch": best_epoch,
    "best_valid_A_micro": best_a_micro if best_epoch else None,
    "train_rows": len(train_rows),
    "valid_rows": len(valid_rows),
    "train_pairs": len(train_ledger),
    "head_dim": cfg.head_dim if cfg.mode == "gpm" else 1,
  }
  (cfg.output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=["bt", "gpm"], required=True)
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--train-file", type=Path, required=True)
  parser.add_argument("--valid-file", type=Path, required=True)
  parser.add_argument("--ledger-file", type=Path, default=Path("data/d/pair_ledger.jsonl"))
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--fold", type=int, default=0)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--head-dim", type=int, default=32)
  parser.add_argument("--batch-items", type=int, default=8)
  parser.add_argument("--pair-batch-size", type=int, default=8)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--max-length", type=int, default=2048)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--lora-r", type=int, default=16)
  parser.add_argument("--lora-alpha", type=int, default=32)
  parser.add_argument("--base-quantization", choices=["8bit", "4bit"], default="8bit")
  args = parser.parse_args()

  if args.mode == "gpm" and args.head_dim % 2 != 0:
    raise SystemExit("--head-dim must be even for GPM")

  cfg = TrainConfig(
    mode=args.mode,
    base_model=args.base_model,
    train_file=args.train_file,
    valid_file=args.valid_file,
    ledger_file=args.ledger_file,
    output_dir=args.output_dir,
    fold=args.fold,
    epochs=args.epochs,
    head_dim=args.head_dim,
    batch_items=args.batch_items,
    pair_batch_size=args.pair_batch_size,
    learning_rate=args.learning_rate,
    max_length=args.max_length,
    seed=args.seed,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    base_quantization=args.base_quantization,
  )
  summary = train_one_mode(cfg)
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
