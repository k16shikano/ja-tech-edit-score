#!/usr/bin/env python3
"""ModernBERT クロスエンコーダで g(x,y) と f(x,y)=g(x,y)-g(x,x) を出す。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class CrossEncoderConfig:
  base_model: str = "sbintuitions/modernbert-ja-310m"
  max_length: int = 1024
  num_labels: int = 1


def build_cross_encoder(
  cfg: CrossEncoderConfig,
  *,
  gradient_checkpointing: bool = True,
):
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
  model = AutoModelForSequenceClassification.from_pretrained(
    cfg.base_model,
    num_labels=cfg.num_labels,
    ignore_mismatched_sizes=True,
  )
  if gradient_checkpointing:
    model.gradient_checkpointing_enable()
  return model, tokenizer


def forward_logits(
  model: nn.Module,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> torch.Tensor:
  encoded = tokenizer(
    drafts,
    candidates,
    padding=True,
    truncation="longest_first",
    max_length=max_length,
    return_tensors="pt",
  )
  encoded = {k: v.to(device) for k, v in encoded.items()}
  out = model(**encoded)
  logits = out.logits
  if logits.ndim == 1:
    return logits
  if logits.shape[-1] == 1:
    return logits.squeeze(-1)
  return logits


def forward_g(
  model: nn.Module,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> torch.Tensor:
  logits = forward_logits(
    model,
    tokenizer,
    drafts,
    candidates,
    max_length=max_length,
    device=device,
  )
  if logits.ndim != 1:
    raise ValueError("forward_g expects scalar logits")
  return logits


def forward_f_delta(
  model: nn.Module,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> torch.Tensor:
  both_drafts = drafts + drafts
  both_cands = candidates + drafts
  g_all = forward_g(
    model,
    tokenizer,
    both_drafts,
    both_cands,
    max_length=max_length,
    device=device,
  )
  n = len(drafts)
  return g_all[:n] - g_all[n:]


@torch.no_grad()
def predict_f_delta(
  model: nn.Module,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
  batch_size: int = 16,
) -> list[float]:
  model.eval()
  out: list[float] = []
  for start in range(0, len(drafts), batch_size):
    batch_d = drafts[start : start + batch_size]
    batch_y = candidates[start : start + batch_size]
    f = forward_f_delta(
      model,
      tokenizer,
      batch_d,
      batch_y,
      max_length=max_length,
      device=device,
    )
    out.extend(float(x) for x in f.detach().cpu().tolist())
  return out


def param_groups(model: nn.Module, *, encoder_lr: float, head_lr: float):
  head_params = list(model.classifier.parameters())
  head_ids = {id(p) for p in head_params}
  encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
  return [
    {"params": encoder_params, "lr": encoder_lr},
    {"params": head_params, "lr": head_lr},
  ]


def save_cross_encoder(
  model: nn.Module,
  tokenizer,
  output_dir: Path,
  *,
  meta: dict,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)
  (output_dir / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def load_cross_encoder(model_dir: Path, *, device: torch.device):
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  meta_path = model_dir / "meta.json"
  meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
  cfg = CrossEncoderConfig(
    base_model=meta.get("base_model", "sbintuitions/modernbert-ja-310m"),
    max_length=int(meta.get("max_length", 1024)),
    num_labels=int(meta.get("num_labels", 1)),
  )
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  model = AutoModelForSequenceClassification.from_pretrained(model_dir)
  model.to(device)
  model.eval()
  return model, tokenizer, cfg, meta


@torch.no_grad()
def predict_vectors(
  model: nn.Module,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
  batch_size: int = 16,
) -> list[list[float]]:
  model.eval()
  out: list[list[float]] = []
  for start in range(0, len(drafts), batch_size):
    batch_d = drafts[start : start + batch_size]
    batch_y = candidates[start : start + batch_size]
    logits = forward_logits(
      model,
      tokenizer,
      batch_d,
      batch_y,
      max_length=max_length,
      device=device,
    )
    out.extend(logits.detach().cpu().tolist())
  return out
