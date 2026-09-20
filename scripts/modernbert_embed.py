#!/usr/bin/env python3
"""ModernBERT を凍結 bi-encoder として文・文書ベクトル化する。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class ModernBertEmbedConfig:
  base_model: str = "sbintuitions/modernbert-ja-310m"
  checkpoint_dir: str = ""
  max_seq_length: int = 512
  batch_size: int = 16
  normalize_embeddings: bool = True


class ModernBertTextEncoder(nn.Module):
  """単文入力の mean pooling 埋め込み。重みは凍結。"""

  def __init__(self, cfg: ModernBertEmbedConfig, *, device: torch.device) -> None:
    super().__init__()
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    self.cfg = cfg
    self.device = device
    if cfg.checkpoint_dir:
      clf = AutoModelForSequenceClassification.from_pretrained(cfg.checkpoint_dir)
      self.backbone = clf.model
      tokenizer_src = cfg.checkpoint_dir
    else:
      self.backbone = AutoModel.from_pretrained(cfg.base_model)
      tokenizer_src = cfg.base_model
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    self.backbone.to(device)
    self.backbone.eval()
    for param in self.backbone.parameters():
      param.requires_grad = False
    self.hidden_size = int(self.backbone.config.hidden_size)

  @torch.no_grad()
  def encode_batch(self, texts: list[str]) -> np.ndarray:
    if not texts:
      return np.zeros((0, self.hidden_size), dtype=np.float32)
    encoded = self.tokenizer(
      texts,
      padding=True,
      truncation=True,
      max_length=self.cfg.max_seq_length,
      return_tensors="pt",
    )
    encoded = {k: v.to(self.device) for k, v in encoded.items()}
    out = self.backbone(**encoded)
    hidden = out.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    if self.cfg.normalize_embeddings:
      pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)

  @torch.no_grad()
  def encode_texts(
    self,
    texts: list[str],
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
  ) -> np.ndarray:
    bs = batch_size or self.cfg.batch_size
    chunks: list[np.ndarray] = []
    iterator = range(0, len(texts), bs)
    if show_progress_bar:
      try:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="modernbert encode", unit="batch")
      except ImportError:
        pass
    for start in iterator:
      batch = texts[start : start + bs]
      chunks.append(self.encode_batch(batch))
    if not chunks:
      return np.zeros((0, self.hidden_size), dtype=np.float32)
    return np.vstack(chunks)


def encode_text_map(
  encoder: ModernBertTextEncoder,
  texts: list[str],
  *,
  batch_size: int | None = None,
  show_progress_bar: bool = False,
) -> dict[str, np.ndarray]:
  unique: list[str] = []
  seen: set[str] = set()
  for text in texts:
    if text not in seen:
      seen.add(text)
      unique.append(text)
  embeddings = encoder.encode_texts(
    unique,
    batch_size=batch_size,
    show_progress_bar=show_progress_bar,
  )
  return {text: embeddings[i] for i, text in enumerate(unique)}


def build_encoder(
  *,
  base_model: str,
  checkpoint_dir: str = "",
  max_seq_length: int,
  device: torch.device,
) -> ModernBertTextEncoder:
  cfg = ModernBertEmbedConfig(
    base_model=base_model,
    checkpoint_dir=checkpoint_dir,
    max_seq_length=max_seq_length,
  )
  return ModernBertTextEncoder(cfg, device=device)


def load_embedder_from_config(config: dict, *, device: str = "cpu"):
  """BT / SentSeq artifact の config から凍結埋め込み器を返す。"""
  backend = str(config.get("embed_backend", "sentence-transformers"))
  if backend == "modernbert":
    max_seq_length = int(config.get("max_seq_length") or 512)
    return build_encoder(
      base_model=str(config.get("modernbert_base_model") or config.get("sentence_model_name")),
      checkpoint_dir=str(config.get("modernbert_checkpoint") or ""),
      max_seq_length=max_seq_length,
      device=torch.device(device),
    )
  from pref_static_utils import load_sentence_model_from_artifact

  return load_sentence_model_from_artifact(config, device=device)


def encode_with_embedder(
  embedder,
  texts: list[str],
  *,
  batch_size: int,
  normalize_embeddings: bool,
  text_prefix: str = "",
) -> np.ndarray:
  if isinstance(embedder, ModernBertTextEncoder):
    return embedder.encode_texts(texts, batch_size=batch_size)
  from pref_static_utils import encode_texts

  return encode_texts(
    embedder,
    texts,
    batch_size=batch_size,
    normalize_embeddings=normalize_embeddings,
    text_prefix=text_prefix,
  )
