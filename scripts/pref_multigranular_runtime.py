#!/usr/bin/env python3
"""一対または三点同時の多段階評価器の読み込みと採点。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pref_static_utils import load_sentence_model_from_artifact
from pref_setwise_runtime import LoadedSetwiseModel, _prepare_for_texts
from setwise_model import (
  SCORE_MODE_INDEPENDENT,
  build_setwise_batch_tensors,
  candidate_logits,
  load_setwise_model_from_artifact,
  normalize_logits_for_duplicate_texts,
  normalize_score_mode,
  public_relative_scores,
  ranked_indices_from_logits,
  strict_top_indices,
)


@dataclass
class LoadedMultigranularModel:
  setwise: LoadedSetwiseModel
  score_mode: str
  stage: str


def load_multigranular_model(
  model_dir: Path,
  *,
  device: str = "",
  encode_batch_size: int = 64,
) -> LoadedMultigranularModel:
  artifact_path = model_dir / "model.pt"
  if not artifact_path.is_file():
    raise FileNotFoundError(f"model not found: {artifact_path}")
  artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
  config = artifact["config"]
  resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
  dev = torch.device(resolved)
  model = load_setwise_model_from_artifact(artifact, device=dev)
  sent_cfg = {
    "sentence_model_name": config["sentence_model_name"],
    "truncate_dim": config.get("truncate_dim"),
    "max_seq_length": config.get("max_seq_length"),
  }
  sentence_model = load_sentence_model_from_artifact(sent_cfg, device=str(dev))
  for param in sentence_model.parameters():
    param.requires_grad = False
  loaded = LoadedSetwiseModel(
    model=model,
    sentence_model=sentence_model,
    normalize_embeddings=bool(config.get("normalize_embeddings", True)),
    text_prefix=str(config.get("text_prefix", "")),
    max_sents=int(config.get("max_sents", 128)),
    encode_batch_size=encode_batch_size,
    device=dev,
    model_dir=str(model_dir),
  )
  return LoadedMultigranularModel(
    setwise=loaded,
    score_mode=normalize_score_mode(config.get("score_mode")),
    stage=str(config.get("stage") or ""),
  )


@torch.no_grad()
def score_candidates_multigranular(
  loaded: LoadedMultigranularModel,
  source_text: str,
  candidates: list[str],
) -> dict:
  if not candidates:
    raise ValueError("need at least 1 candidate")
  unique_texts: list[str] = [source_text]
  seen = {source_text}
  for text in candidates:
    if text not in seen:
      seen.add(text)
      unique_texts.append(text)
  prepared = _prepare_for_texts(loaded.setwise, unique_texts)
  tensors = build_setwise_batch_tensors(
    source_texts=[source_text],
    candidate_texts=[candidates],
    prepared=prepared,
    embed_dim=loaded.setwise.model.embed_dim,
    device=loaded.setwise.device,
  )
  raw = candidate_logits(
    loaded.setwise.model,
    tensors,
    score_mode=loaded.score_mode,
  ).squeeze(0)
  raw = normalize_logits_for_duplicate_texts(candidates, raw)
  self_tensors = build_setwise_batch_tensors(
    source_texts=[source_text],
    candidate_texts=[[source_text]],
    prepared=prepared,
    embed_dim=loaded.setwise.model.embed_dim,
    device=loaded.setwise.device,
  )
  self_logit = candidate_logits(
    loaded.setwise.model,
    self_tensors,
    score_mode=SCORE_MODE_INDEPENDENT
    if loaded.score_mode == SCORE_MODE_INDEPENDENT
    else loaded.score_mode,
  ).squeeze(0)[0]
  public = public_relative_scores(raw, self_logit)
  scores = [float(public[i].item()) for i in range(len(candidates))]
  ranked_indices = ranked_indices_from_logits(public)
  winners = strict_top_indices(public)[0]
  return {
    "logits": scores,
    "raw_logits": [float(raw[i].item()) for i in range(len(candidates))],
    "self_logit": float(self_logit.item()),
    "ranked_indices": ranked_indices,
    "ranked_texts": [candidates[i] for i in ranked_indices],
    "best_index": winners[0] if len(winners) == 1 else None,
    "top1_tie": len(winners) != 1,
    "strict_top_indices": winners,
  }
