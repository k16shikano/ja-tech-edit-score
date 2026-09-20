#!/usr/bin/env python3
"""学習用データ D を 1 fold 分学習する（ModernBERT または ruri 凍結）。"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pref_d_cross_encoder import (
  CrossEncoderConfig,
  build_cross_encoder,
  forward_f_delta,
  param_groups,
  save_cross_encoder,
)
from pref_d_interval_utils import (
  IntervalLossConfig,
  eval_report,
  interval_zone_loss,
  load_d_rows,
  primary_a_sign_accuracy,
)
from pref_d_ruri_interval import (
  RuriIntervalConfig,
  RuriIntervalModel,
  build_text_map,
  forward_f_delta_batch,
  load_ruri_encoder,
  rows_to_features,
  rows_to_self_features,
  save_ruri_interval,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainConfig:
  backend: str
  base_model: str
  train_file: Path
  valid_file: Path
  output_dir: Path
  fold: int
  epochs: int
  batch_size: int = 16
  encoder_lr: float = 5e-6
  head_lr: float = 1e-4
  weight_decay: float = 0.01
  warmup_ratio: float = 0.1
  max_length: int = 1024
  seed: int = 0
  precision: str = "auto"


def resolve_device() -> torch.device:
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for pref-d-interval training")
  return torch.device("cuda")


def resolve_precision(precision: str, device: torch.device) -> str:
  if device.type != "cuda":
    return "fp32"
  if precision == "auto":
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
  return precision


def collect_texts(rows: list[dict]) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for row in rows:
    for key in ("draft", "y"):
      value = str(row[key])
      if value not in seen:
        seen.add(value)
        texts.append(value)
  return texts


@torch.no_grad()
def validate_modernbert(
  model,
  tokenizer,
  rows: list[dict],
  *,
  cfg: TrainConfig,
  device: torch.device,
) -> tuple[dict, list[float]]:
  from pref_d_cross_encoder import predict_f_delta

  f_values = predict_f_delta(
    model,
    tokenizer,
    [r["draft"] for r in rows],
    [r["y"] for r in rows],
    max_length=cfg.max_length,
    device=device,
    batch_size=cfg.batch_size,
  )
  return eval_report(rows, f_values), f_values


@torch.no_grad()
def validate_ruri(
  model: RuriIntervalModel,
  scaler: StandardScaler,
  rows: list[dict],
  text_to_emb: dict[str, np.ndarray],
  *,
  cfg: TrainConfig,
  device: torch.device,
) -> tuple[dict, list[float]]:
  from pref_d_ruri_interval import predict_f_delta

  f_values = predict_f_delta(
    model,
    scaler,
    rows,
    text_to_emb,
    device=device,
    batch_size=cfg.batch_size,
  )
  return eval_report(rows, f_values), f_values


def train_modernbert(cfg: TrainConfig) -> dict:
  from transformers import get_linear_schedule_with_warmup

  device = resolve_device()
  precision = resolve_precision(cfg.precision, device)
  train_rows = load_d_rows(cfg.train_file)
  valid_rows = load_d_rows(cfg.valid_file)
  loss_cfg = IntervalLossConfig()

  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  model, tokenizer = build_cross_encoder(CrossEncoderConfig(cfg.base_model, cfg.max_length))
  model.to(device)
  model.train()

  optimizer = torch.optim.AdamW(
    param_groups(model, encoder_lr=cfg.encoder_lr, head_lr=cfg.head_lr),
    weight_decay=cfg.weight_decay,
  )
  steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
  total_steps = steps_per_epoch * cfg.epochs
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * cfg.warmup_ratio),
    num_training_steps=total_steps,
  )
  autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

  log_path = cfg.output_dir / "train_log.jsonl"
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  best_primary = -1.0
  best_epoch = 0
  rng = random.Random(cfg.seed)

  for epoch in range(1, cfg.epochs + 1):
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    epoch_loss = 0.0
    n_batches = 0
    model.train()
    for start in range(0, len(train_rows), cfg.batch_size):
      idxs = order[start : start + cfg.batch_size]
      batch = [train_rows[i] for i in idxs]
      drafts = [r["draft"] for r in batch]
      ys = [r["y"] for r in batch]
      positions = [r["position"] for r in batch]

      ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast(device_type="cpu", enabled=False)
      )
      optimizer.zero_grad(set_to_none=True)
      with ctx:
        f_vals = forward_f_delta(
          model,
          tokenizer,
          drafts,
          ys,
          max_length=cfg.max_length,
          device=device,
        )
        loss = interval_zone_loss(f_vals, positions, cfg=loss_cfg)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
      epoch_loss += float(loss.item())
      n_batches += 1

    valid_metrics, valid_f = validate_modernbert(
      model, tokenizer, valid_rows, cfg=cfg, device=device
    )
    row = {
      "epoch": epoch,
      "train_loss": epoch_loss / max(n_batches, 1),
      "valid_primary_a_sign_acc": valid_metrics["primary_a_sign_acc"],
      "valid_primary_a_frac_lt_0": valid_metrics["primary_a_frac_lt_0"],
      "valid_primary_c_frac_gt_0": valid_metrics["primary_c_frac_gt_0"],
    }
    with log_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)

    if valid_metrics["primary_a_sign_acc"] > best_primary:
      best_primary = valid_metrics["primary_a_sign_acc"]
      best_epoch = epoch
      save_cross_encoder(
        model,
        tokenizer,
        cfg.output_dir / "best",
        meta={
          "kind": "pref-d-interval",
          "backend": "modernbert",
          "base_model": cfg.base_model,
          "max_length": cfg.max_length,
          "fold": cfg.fold,
          "best_epoch": best_epoch,
          "seed": cfg.seed,
        },
      )
      (cfg.output_dir / "best_valid_predictions.jsonl").write_text(
        "\n".join(
          json.dumps({**r, "f": fv}, ensure_ascii=False)
          for r, fv in zip(valid_rows, valid_f)
        )
        + ("\n" if valid_rows else ""),
        encoding="utf-8",
      )

  summary = {
    "backend": "modernbert",
    "fold": cfg.fold,
    "epochs": cfg.epochs,
    "best_epoch": best_epoch,
    "best_primary_a_sign_acc": best_primary,
    "train_rows": len(train_rows),
    "valid_rows": len(valid_rows),
  }
  (cfg.output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  return summary


def train_ruri(cfg: TrainConfig) -> dict:
  device = resolve_device()
  train_rows = load_d_rows(cfg.train_file)
  valid_rows = load_d_rows(cfg.valid_file)
  loss_cfg = IntervalLossConfig()
  ruri_cfg = RuriIntervalConfig(base_model=cfg.base_model, max_seq_length=cfg.max_length)

  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  encoder = load_ruri_encoder(ruri_cfg, device=str(device))
  all_texts = collect_texts(train_rows + valid_rows)
  text_to_emb = build_text_map(encoder, all_texts, cfg=ruri_cfg, batch_size=cfg.batch_size)

  xy_train = rows_to_features(train_rows, text_to_emb)
  xx_train = rows_to_self_features(train_rows, text_to_emb)
  scaler = StandardScaler()
  scaler.fit(np.vstack([xy_train, xx_train]))

  model = RuriIntervalModel(xy_train.shape[1]).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.head_lr,
    weight_decay=cfg.weight_decay,
  )

  log_path = cfg.output_dir / "train_log.jsonl"
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  best_primary = -1.0
  best_epoch = 0
  rng = random.Random(cfg.seed)

  for epoch in range(1, cfg.epochs + 1):
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    epoch_loss = 0.0
    n_batches = 0
    model.train()
    for start in range(0, len(train_rows), cfg.batch_size):
      idxs = order[start : start + cfg.batch_size]
      batch = [train_rows[i] for i in idxs]
      xy = rows_to_features(batch, text_to_emb)
      xx = rows_to_self_features(batch, text_to_emb)
      positions = [r["position"] for r in batch]

      optimizer.zero_grad(set_to_none=True)
      f_vals = forward_f_delta_batch(model, scaler, xy, xx, device=device)
      loss = interval_zone_loss(f_vals, positions, cfg=loss_cfg)
      loss.backward()
      optimizer.step()
      epoch_loss += float(loss.item())
      n_batches += 1

    valid_metrics, valid_f = validate_ruri(
      model, scaler, valid_rows, text_to_emb, cfg=cfg, device=device
    )
    row = {
      "epoch": epoch,
      "train_loss": epoch_loss / max(n_batches, 1),
      "valid_primary_a_sign_acc": valid_metrics["primary_a_sign_acc"],
    }
    with log_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)

    if valid_metrics["primary_a_sign_acc"] > best_primary:
      best_primary = valid_metrics["primary_a_sign_acc"]
      best_epoch = epoch
      save_ruri_interval(
        model,
        scaler,
        cfg.output_dir / "best",
        meta={
          "kind": "pref-d-interval",
          "backend": "ruri",
          "base_model": cfg.base_model,
          "max_seq_length": cfg.max_length,
          "truncate_dim": ruri_cfg.truncate_dim,
          "text_prefix": ruri_cfg.text_prefix,
          "fold": cfg.fold,
          "best_epoch": best_epoch,
          "seed": cfg.seed,
        },
      )
      (cfg.output_dir / "best_valid_predictions.jsonl").write_text(
        "\n".join(
          json.dumps({**r, "f": fv}, ensure_ascii=False)
          for r, fv in zip(valid_rows, valid_f)
        )
        + ("\n" if valid_rows else ""),
        encoding="utf-8",
      )

  summary = {
    "backend": "ruri",
    "fold": cfg.fold,
    "epochs": cfg.epochs,
    "best_epoch": best_epoch,
    "best_primary_a_sign_acc": best_primary,
    "train_rows": len(train_rows),
    "valid_rows": len(valid_rows),
  }
  (cfg.output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=["modernbert", "ruri"], required=True)
  parser.add_argument("--base-model", default="")
  parser.add_argument("--train-file", type=Path, required=True)
  parser.add_argument("--valid-file", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--fold", type=int, default=0)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--encoder-lr", type=float, default=5e-6)
  parser.add_argument("--head-lr", type=float, default=1e-4)
  parser.add_argument("--max-length", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  base_model = args.base_model or (
    "sbintuitions/modernbert-ja-310m"
    if args.backend == "modernbert"
    else "cl-nagoya/ruri-v3-30m"
  )
  cfg = TrainConfig(
    backend=args.backend,
    base_model=base_model,
    train_file=args.train_file,
    valid_file=args.valid_file,
    output_dir=args.output_dir,
    fold=args.fold,
    epochs=args.epochs,
    batch_size=args.batch_size,
    encoder_lr=args.encoder_lr,
    head_lr=args.head_lr,
    max_length=args.max_length,
    seed=args.seed,
  )

  if cfg.backend == "modernbert":
    summary = train_modernbert(cfg)
  else:
    summary = train_ruri(cfg)
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
