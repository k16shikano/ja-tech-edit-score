#!/usr/bin/env python3
"""D ペア BT/GPM の採点関数（C 評価・難試験共通）。"""
from __future__ import annotations

from pathlib import Path

import torch

from pref_d_cross_encoder import load_cross_encoder, predict_f_delta, predict_vectors
from pref_d_pair_utils import gpm_score


def resolve_model_dir(cv_dir: Path, fold: int) -> Path:
  best = cv_dir / f"fold{fold}" / "best"
  if not (best / "config.json").is_file():
    raise FileNotFoundError(best)
  return best


def gpm_rank_scores(vectors: list[list[float]]) -> list[float]:
  tensors = [torch.tensor(v, dtype=torch.float64) for v in vectors]
  scores: list[float] = []
  for i, vec_i in enumerate(tensors):
    total = 0.0
    for j, vec_j in enumerate(tensors):
      if i == j:
        continue
      total += float(gpm_score(vec_i, vec_j).item())
    scores.append(total)
  return scores


def make_score_fn(model_dir: Path, *, device: str):
  dev = torch.device(device)
  model, tokenizer, cfg, meta = load_cross_encoder(model_dir, device=dev)
  mode = str(meta.get("mode") or "bt")
  max_length = int(meta.get("max_length", cfg.max_length))
  batch_size = 8

  if mode == "bt":

    def score(source: str, candidates: list[str]) -> list[float]:
      vals = predict_f_delta(
        model,
        tokenizer,
        [source] * len(candidates),
        candidates,
        max_length=max_length,
        device=dev,
        batch_size=batch_size,
      )
      return [float(v) for v in vals]

    return score, {"mode": mode, "head_dim": int(meta.get("head_dim", 1))}

  if mode == "gpm":

    def score(source: str, candidates: list[str]) -> list[float]:
      vecs = predict_vectors(
        model,
        tokenizer,
        [source] * len(candidates),
        candidates,
        max_length=max_length,
        device=dev,
        batch_size=batch_size,
      )
      return gpm_rank_scores(vecs)

    return score, {"mode": mode, "head_dim": int(meta.get("head_dim", cfg.num_labels))}

  raise ValueError(mode)
