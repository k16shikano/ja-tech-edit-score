#!/usr/bin/env python3
"""Cross-encoder Bradley-Terry 報酬モデルの学習（段階2b、GPU 向け）。

(source, candidate) をひとつのエンコーダに連結入力し、スカラー s(source, candidate)
を出す。損失は P(A ≻ B) = σ(s(A) - s(B))。

凍結埋め込み＋線形ヘッド（pref-bt）と違い、エンコーダ本体を fine-tune するので、
下書きと候補の関係（何がどう直されたか）を内部で見られる。採否は LOPO
（eval_pref_ce_xproject.py）で pref-bt と比較して決める。

依存は torch / transformers / numpy のみ（sentence-transformers 不要）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_jsonl(path: str, limit: int = -1) -> list[dict]:
  rows: list[dict] = []
  with Path(path).open(encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
      if 0 < limit <= len(rows):
        break
  return rows


def unique_preference_pairs(rows: list[dict]) -> list[dict]:
  """swap 拡張を除き、candidate_a=chosen / candidate_b=rejected の行に絞る。

  train_pref_bt.unique_preference_pairs と同じ規約（依存を切るため再実装）。
  """
  selected: list[dict] = []
  for row in rows:
    meta = row.get("meta", {})
    if meta.get("pair_order", "chosen_first") != "chosen_first":
      continue
    if int(row["label"]) != 1:
      continue
    selected.append(row)
  return selected


@dataclass
class CeTrainConfig:
  base_model: str
  max_length: int = 512
  batch_size: int = 16
  epochs: int = 2
  lr: float = 3e-5
  weight_decay: float = 0.01
  warmup_ratio: float = 0.1
  grad_accum: int = 1
  max_grad_norm: float = 1.0
  seed: int = 0
  precision: str = "auto"  # auto | bf16 | fp16 | fp32
  gradient_checkpointing: bool = False


def resolve_device() -> torch.device:
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(precision: str, device: torch.device) -> str:
  if device.type != "cuda":
    return "fp32"
  if precision == "auto":
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
  return precision


def build_model(base_model: str, *, gradient_checkpointing: bool):
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(base_model)
  # 既存の分類ヘッド付きモデルでもスカラーヘッドを新規初期化できるようにする
  model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=1,
    ignore_mismatched_sizes=True,
  )
  if gradient_checkpointing:
    model.gradient_checkpointing_enable()
  return model, tokenizer


def forward_scores(
  model,
  tokenizer,
  sources: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> torch.Tensor:
  encoded = tokenizer(
    sources,
    candidates,
    padding=True,
    truncation="longest_first",
    max_length=max_length,
    return_tensors="pt",
  )
  encoded = {k: v.to(device) for k, v in encoded.items()}
  out = model(**encoded)
  return out.logits.squeeze(-1)


def train_ce_model(
  train_rows: list[dict],
  cfg: CeTrainConfig,
  *,
  log_prefix: str = "",
):
  """学習済み (model, tokenizer, train_metrics) を返す。"""
  from transformers import get_linear_schedule_with_warmup

  device = resolve_device()
  precision = resolve_precision(cfg.precision, device)

  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  model, tokenizer = build_model(
    cfg.base_model, gradient_checkpointing=cfg.gradient_checkpointing
  )
  model.to(device)
  model.train()

  n = len(train_rows)
  steps_per_epoch = math.ceil(n / cfg.batch_size)
  total_steps = math.ceil(steps_per_epoch * cfg.epochs / cfg.grad_accum)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * cfg.warmup_ratio),
    num_training_steps=total_steps,
  )
  scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
  autocast_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
  }.get(precision)

  print(
    f"{log_prefix}train: n={n} device={device.type} precision={precision} "
    f"epochs={cfg.epochs} bs={cfg.batch_size} accum={cfg.grad_accum} "
    f"lr={cfg.lr} steps={total_steps}",
    flush=True,
  )

  last_loss = 0.0
  rng = random.Random(cfg.seed)
  micro_step = 0
  for epoch in range(cfg.epochs):
    order = list(range(n))
    rng.shuffle(order)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, n, cfg.batch_size):
      batch = [train_rows[i] for i in order[start : start + cfg.batch_size]]
      sources = [r["source_text"] for r in batch]
      chosen = [r["candidate_a"] for r in batch]
      rejected = [r["candidate_b"] for r in batch]

      ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast(device_type="cpu", enabled=False)
      )
      with ctx:
        s_w = forward_scores(
          model, tokenizer, sources, chosen,
          max_length=cfg.max_length, device=device,
        )
        s_l = forward_scores(
          model, tokenizer, sources, rejected,
          max_length=cfg.max_length, device=device,
        )
        loss = torch.nn.functional.softplus(-(s_w - s_l)).mean() / cfg.grad_accum

      if scaler is not None:
        scaler.scale(loss).backward()
      else:
        loss.backward()
      micro_step += 1

      if micro_step % cfg.grad_accum == 0:
        if scaler is not None:
          scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if scaler is not None:
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      last_loss = float(loss.item()) * cfg.grad_accum
      epoch_loss += last_loss
      n_batches += 1
    print(
      f"{log_prefix}epoch {epoch + 1}/{cfg.epochs} "
      f"mean_bt_loss={epoch_loss / max(n_batches, 1):.4f}",
      flush=True,
    )

  model.eval()
  return model, tokenizer, {"train_last_bt_loss": last_loss}


@torch.no_grad()
def eval_ce_pairs(
  model,
  tokenizer,
  rows: list[dict],
  *,
  max_length: int,
  batch_size: int,
) -> dict[str, float]:
  device = next(model.parameters()).device
  margins: list[float] = []
  for start in range(0, len(rows), batch_size):
    batch = rows[start : start + batch_size]
    sources = [r["source_text"] for r in batch]
    s_w = forward_scores(
      model, tokenizer, sources, [r["candidate_a"] for r in batch],
      max_length=max_length, device=device,
    )
    s_l = forward_scores(
      model, tokenizer, sources, [r["candidate_b"] for r in batch],
      max_length=max_length, device=device,
    )
    margins.extend((s_w - s_l).float().cpu().tolist())
  m = torch.tensor(margins)
  return {
    "pair_accuracy": float((m > 0).float().mean().item()),
    "bt_loss": float(torch.nn.functional.softplus(-m).mean().item()),
    "mean_margin": float(m.mean().item()),
  }


def save_ce_model(model, tokenizer, cfg: CeTrainConfig, output_dir: Path) -> None:
  model_dir = output_dir / "model"
  model_dir.mkdir(parents=True, exist_ok=True)
  model.save_pretrained(model_dir)
  tokenizer.save_pretrained(model_dir)
  meta = {
    "kind": "pref-ce",
    "base_model": cfg.base_model,
    "max_length": cfg.max_length,
  }
  (output_dir / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-model", default="sbintuitions/modernbert-ja-130m")
  parser.add_argument("--train-file", required=True, help="preference jsonl (swap 込み可)")
  parser.add_argument("--eval-file", required=True, help="preference jsonl")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--lr", type=float, default=3e-5)
  parser.add_argument("--weight-decay", type=float, default=0.01)
  parser.add_argument("--warmup-ratio", type=float, default=0.1)
  parser.add_argument("--grad-accum", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"]
  )
  parser.add_argument("--gradient-checkpointing", action="store_true")
  parser.add_argument("--max-train-samples", type=int, default=-1)
  args = parser.parse_args()

  train_rows = unique_preference_pairs(load_jsonl(args.train_file))
  eval_rows = unique_preference_pairs(load_jsonl(args.eval_file))
  if 0 < args.max_train_samples < len(train_rows):
    train_rows = train_rows[: args.max_train_samples]
  if not train_rows or not eval_rows:
    raise SystemExit("train/eval preference pairs are empty after filtering swaps")

  cfg = CeTrainConfig(
    base_model=args.base_model,
    max_length=args.max_length,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    grad_accum=args.grad_accum,
    seed=args.seed,
    precision=args.precision,
    gradient_checkpointing=args.gradient_checkpointing,
  )

  model, tokenizer, train_metrics = train_ce_model(train_rows, cfg)
  eval_metrics = eval_ce_pairs(
    model, tokenizer, eval_rows,
    max_length=cfg.max_length, batch_size=cfg.batch_size,
  )

  output_dir = Path(args.output_dir)
  save_ce_model(model, tokenizer, cfg, output_dir)
  metrics = {
    "base_model": cfg.base_model,
    "max_length": cfg.max_length,
    "epochs": cfg.epochs,
    "lr": cfg.lr,
    "batch_size": cfg.batch_size,
    "train_pairs": len(train_rows),
    "eval_pairs": len(eval_rows),
    **train_metrics,
    **{f"eval_{k}": v for k, v in eval_metrics.items()},
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(f"saved: {output_dir}")
  print(f"eval_pair_accuracy: {eval_metrics['pair_accuracy']:.4f}")
  print(f"eval_bt_loss: {eval_metrics['bt_loss']:.4f}")
  print(f"eval_mean_margin: {eval_metrics['mean_margin']:.4f}")


if __name__ == "__main__":
  main()
