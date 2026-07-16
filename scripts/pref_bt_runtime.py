#!/usr/bin/env python3
"""Bradley-Terry 報酬モデルの読み込みとスコアリング。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from joblib import load
from sentence_transformers import SentenceTransformer

from pref_static_utils import (
  assemble_pointwise_feature_vector,
  encode_texts,
  load_sentence_model_from_artifact,
)
from train_pref_bt import LinearRewardHead


@dataclass
class LoadedBtModel:
  head: LinearRewardHead
  scaler: object
  sentence_model: SentenceTransformer
  normalize_embeddings: bool
  text_prefix: str
  model_dir: str


def load_bt_model(model_dir: Path) -> LoadedBtModel:
  artifact_path = model_dir / "model.joblib"
  if not artifact_path.is_file():
    raise FileNotFoundError(f"model not found: {artifact_path}")
  artifact = load(artifact_path)
  if artifact.get("kind") != "pref-bt":
    raise ValueError(f"not a pref-bt artifact: {artifact_path}")
  head = LinearRewardHead(int(artifact["input_dim"]))
  head.load_state_dict(artifact["head_state_dict"])
  head.eval()
  return LoadedBtModel(
    head=head,
    scaler=artifact["scaler"],
    sentence_model=load_sentence_model_from_artifact(artifact, device="cpu"),
    normalize_embeddings=bool(artifact["normalize_embeddings"]),
    text_prefix=str(artifact.get("text_prefix", "")),
    model_dir=str(model_dir),
  )


def score_candidates_bt(
  loaded: LoadedBtModel,
  source_text: str,
  candidates: list[str],
  *,
  batch_size: int = 32,
) -> list[float]:
  """各候補について s(source, candidate) を返す。"""
  if not candidates:
    return []
  unique: list[str] = [source_text]
  seen = {source_text}
  for text in candidates:
    if text not in seen:
      seen.add(text)
      unique.append(text)
  embeddings = encode_texts(
    loaded.sentence_model,
    unique,
    batch_size=batch_size,
    normalize_embeddings=loaded.normalize_embeddings,
    text_prefix=loaded.text_prefix,
  )
  emb_map = {
    text: np.asarray(vec, dtype=np.float32)
    for text, vec in zip(unique, embeddings, strict=True)
  }
  v_src = emb_map[source_text]
  len_s = float(len(source_text))
  feats = np.vstack(
    [
      assemble_pointwise_feature_vector(
        v_src,
        emb_map[cand],
        len_source=len_s,
        len_candidate=float(len(cand)),
      )
      for cand in candidates
    ]
  )
  scaled = loaded.scaler.transform(feats)
  with torch.no_grad():
    scores = loaded.head(torch.tensor(scaled, dtype=torch.float32)).numpy()
  return [float(s) for s in scores]
