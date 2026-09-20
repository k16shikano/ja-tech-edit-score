#!/usr/bin/env python3
"""ruri 凍結 bi-encoder + 線形層で f(x,y)=g(x,y)-g(x,x) を出す。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from torch import nn

from pref_static_utils import assemble_pointwise_feature_vector, encode_text_map, normalize_truncate_dim


@dataclass
class RuriIntervalConfig:
  base_model: str = "cl-nagoya/ruri-v3-30m"
  max_seq_length: int = 512
  text_prefix: str = "文章: "
  truncate_dim: int = 0


class RuriIntervalModel(nn.Module):
  def __init__(self, in_dim: int) -> None:
    super().__init__()
    self.head = nn.Linear(in_dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(x).squeeze(-1)


def feature_vector(
  emb_draft: np.ndarray,
  emb_cand: np.ndarray,
  *,
  len_draft: float,
  len_cand: float,
) -> np.ndarray:
  return assemble_pointwise_feature_vector(
    emb_draft,
    emb_cand,
    len_source=len_draft,
    len_candidate=len_cand,
  )


def build_text_map(
  encoder: SentenceTransformer,
  texts: list[str],
  *,
  cfg: RuriIntervalConfig,
  batch_size: int,
) -> dict[str, np.ndarray]:
  unique = sorted(set(texts))
  embs = encode_text_map(
    encoder,
    unique,
    batch_size=batch_size,
    normalize_embeddings=True,
    text_prefix=cfg.text_prefix,
  )
  return embs


def rows_to_features(
  rows: list[dict],
  text_to_emb: dict[str, np.ndarray],
) -> np.ndarray:
  feats = [
    feature_vector(
      text_to_emb[row["draft"]],
      text_to_emb[row["y"]],
      len_draft=float(len(row["draft"])),
      len_cand=float(len(row["y"])),
    )
    for row in rows
  ]
  return np.vstack(feats)


def rows_to_self_features(
  rows: list[dict],
  text_to_emb: dict[str, np.ndarray],
) -> np.ndarray:
  feats = [
    feature_vector(
      text_to_emb[row["draft"]],
      text_to_emb[row["draft"]],
      len_draft=float(len(row["draft"])),
      len_cand=float(len(row["draft"])),
    )
    for row in rows
  ]
  return np.vstack(feats)


def forward_f_delta_batch(
  model: RuriIntervalModel,
  scaler: StandardScaler,
  xy_feats: np.ndarray,
  xx_feats: np.ndarray,
  *,
  device: torch.device,
) -> torch.Tensor:
  g_xy = model(torch.tensor(scaler.transform(xy_feats), dtype=torch.float32, device=device))
  g_xx = model(torch.tensor(scaler.transform(xx_feats), dtype=torch.float32, device=device))
  return g_xy - g_xx


@torch.no_grad()
def predict_f_delta(
  model: RuriIntervalModel,
  scaler: StandardScaler,
  rows: list[dict],
  text_to_emb: dict[str, np.ndarray],
  *,
  device: torch.device,
  batch_size: int = 32,
) -> list[float]:
  model.eval()
  out: list[float] = []
  for start in range(0, len(rows), batch_size):
    batch = rows[start : start + batch_size]
    xy = rows_to_features(batch, text_to_emb)
    xx = rows_to_self_features(batch, text_to_emb)
    f = forward_f_delta_batch(model, scaler, xy, xx, device=device)
    out.extend(float(x) for x in f.cpu().tolist())
  return out


def load_ruri_encoder(cfg: RuriIntervalConfig, *, device: str) -> SentenceTransformer:
  truncate_dim = normalize_truncate_dim(cfg.truncate_dim)
  encoder = SentenceTransformer(cfg.base_model, device=device, truncate_dim=truncate_dim)
  if cfg.max_seq_length > 0:
    encoder.max_seq_length = cfg.max_seq_length
  for p in encoder.parameters():
    p.requires_grad = False
  encoder.eval()
  return encoder


def save_ruri_interval(
  model: RuriIntervalModel,
  scaler: StandardScaler,
  output_dir: Path,
  *,
  meta: dict,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "head_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
      "scaler": scaler,
      "input_dim": int(scaler.n_features_in_),
    },
    output_dir / "model.pt",
  )
  (output_dir / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def load_ruri_interval(model_dir: Path, *, device: torch.device):
  meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
  cfg = RuriIntervalConfig(
    base_model=meta.get("base_model", "cl-nagoya/ruri-v3-30m"),
    max_seq_length=int(meta.get("max_seq_length", 512)),
    text_prefix=meta.get("text_prefix", "文章: "),
    truncate_dim=int(meta.get("truncate_dim", 0)),
  )
  blob = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
  model = RuriIntervalModel(int(blob["input_dim"]))
  model.load_state_dict(blob["head_state_dict"])
  model.to(device)
  model.eval()
  return model, blob["scaler"], cfg, meta
