#!/usr/bin/env python3
"""setwise 順位モデルの読み込みと一括順位付け。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from pref_static_utils import encode_texts, load_sentence_model_from_artifact
from sentseq_utils import split_document_sentences, truncate_sentence_units
from setwise_model import (
  SUPPORTED_ARTIFACT_KINDS,
  SetwiseRankingModel,
  build_setwise_batch_tensors,
  load_setwise_model_from_artifact,
  normalize_logits_for_duplicate_texts,
  ranked_indices_from_logits,
  strict_top_indices,
)
from train_pref_sentseq import prepare_sentence_data


@dataclass
class LoadedSetwiseModel:
  model: SetwiseRankingModel
  sentence_model: SentenceTransformer
  normalize_embeddings: bool
  text_prefix: str
  max_sents: int
  encode_batch_size: int
  device: torch.device
  model_dir: str


def load_setwise_model(
  model_dir: Path,
  *,
  device: str = "",
  encode_batch_size: int = 64,
) -> LoadedSetwiseModel:
  artifact_path = model_dir / "model.pt"
  if not artifact_path.is_file():
    raise FileNotFoundError(f"model not found: {artifact_path}")
  artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
  if artifact.get("kind") not in SUPPORTED_ARTIFACT_KINDS:
    raise ValueError(
      f"not a supported setwise artifact (kind={artifact.get('kind')!r}): {artifact_path}"
    )

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

  return LoadedSetwiseModel(
    model=model,
    sentence_model=sentence_model,
    normalize_embeddings=bool(config.get("normalize_embeddings", True)),
    text_prefix=str(config.get("text_prefix", "")),
    max_sents=int(config.get("max_sents", 128)),
    encode_batch_size=encode_batch_size,
    device=dev,
    model_dir=str(model_dir),
  )


def _encode_unique_sentences(
  loaded: LoadedSetwiseModel,
  sentences: list[str],
) -> dict[str, np.ndarray]:
  if not sentences:
    return {}
  embeddings = encode_texts(
    loaded.sentence_model,
    sentences,
    batch_size=loaded.encode_batch_size,
    normalize_embeddings=loaded.normalize_embeddings,
    text_prefix=loaded.text_prefix,
  )
  return {
    sent: np.asarray(vec, dtype=np.float32)
    for sent, vec in zip(sentences, embeddings, strict=True)
  }


def _prepare_for_texts(
  loaded: LoadedSetwiseModel,
  texts: list[str],
):
  unique_sents: list[str] = []
  seen: set[str] = set()
  for text in texts:
    units = truncate_sentence_units(
      split_document_sentences(text),
      max_sents=loaded.max_sents,
    )
    for unit in units:
      if unit.text not in seen:
        seen.add(unit.text)
        unique_sents.append(unit.text)
  sent_to_embedding = _encode_unique_sentences(loaded, unique_sents)
  return prepare_sentence_data(
    texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=loaded.max_sents,
    device=loaded.device,
  )


@torch.no_grad()
def rank_candidates_setwise(
  loaded: LoadedSetwiseModel,
  source_text: str,
  candidates: list[str],
) -> dict:
  """source + candidates を 1 回で順位付けする。"""
  if len(candidates) < 2:
    raise ValueError("need at least 2 candidates")
  unique_texts: list[str] = [source_text]
  seen = {source_text}
  for text in candidates:
    if text not in seen:
      seen.add(text)
      unique_texts.append(text)
  prepared = _prepare_for_texts(loaded, unique_texts)
  tensors = build_setwise_batch_tensors(
    source_texts=[source_text],
    candidate_texts=[candidates],
    prepared=prepared,
    embed_dim=loaded.model.embed_dim,
    device=loaded.device,
  )
  logits = loaded.model(
    tensors.source_sent_emb,
    tensors.source_para_start_mask,
    tensors.source_seq_len,
    tensors.candidate_sent_emb,
    tensors.candidate_para_start_mask,
    tensors.candidate_seq_len,
  ).squeeze(0)
  logits = normalize_logits_for_duplicate_texts(candidates, logits)
  scores = [float(logits[i].item()) for i in range(len(candidates))]
  ranked_indices = ranked_indices_from_logits(logits)
  ranked_texts = [candidates[i] for i in ranked_indices]
  winners = strict_top_indices(logits)[0]
  top1_tie = len(winners) != 1
  best_index = winners[0] if len(winners) == 1 else None
  return {
    "logits": scores,
    "ranked_indices": ranked_indices,
    "ranked_texts": ranked_texts,
    "best_index": best_index,
    "top1_tie": top1_tie,
    "strict_top_indices": winners,
  }
