#!/usr/bin/env python3
"""文列 Transformer 報酬モデル（pref-sentseq）の読み込みとスコアリング。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from pref_static_utils import encode_texts, load_sentence_model_from_artifact
from sentseq_utils import split_document_sentences, truncate_sentence_units
from train_pref_sentseq import (
  SentSeqRewardModel,
  encode_texts_to_doc_vectors,
  load_sentseq_model_from_artifact,
  prepare_sentence_data,
)


@dataclass
class LoadedSentSeqModel:
  model: SentSeqRewardModel
  sentence_model: SentenceTransformer
  normalize_embeddings: bool
  text_prefix: str
  max_sents: int
  encode_batch_size: int
  device: torch.device
  model_dir: str


def load_sentseq_model(
  model_dir: Path,
  *,
  device: str = "",
  encode_batch_size: int = 64,
) -> LoadedSentSeqModel:
  artifact_path = model_dir / "model.pt"
  if not artifact_path.is_file():
    raise FileNotFoundError(f"model not found: {artifact_path}")
  artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
  if artifact.get("kind") != "pref-sentseq":
    raise ValueError(f"not a pref-sentseq artifact: {artifact_path}")

  config = artifact["config"]
  resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
  dev = torch.device(resolved)
  model = load_sentseq_model_from_artifact(artifact, device=dev)
  sent_cfg = {
    "sentence_model_name": config["sentence_model_name"],
    "truncate_dim": config.get("truncate_dim"),
    "max_seq_length": config.get("max_seq_length"),
  }
  sentence_model = load_sentence_model_from_artifact(sent_cfg, device=str(dev))
  for param in sentence_model.parameters():
    param.requires_grad = False

  return LoadedSentSeqModel(
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
  loaded: LoadedSentSeqModel,
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
  loaded: LoadedSentSeqModel,
  texts: list[str],
) -> PreparedSentSeqData:
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
def score_candidates_sentseq(
  loaded: LoadedSentSeqModel,
  source_text: str,
  candidates: list[str],
  *,
  batch_size: int = 16,
) -> list[float]:
  """各候補について s(source, candidate) を返す。"""
  if not candidates:
    return []

  unique_texts: list[str] = [source_text]
  seen = {source_text}
  for text in candidates:
    if text not in seen:
      seen.add(text)
      unique_texts.append(text)

  prepared = _prepare_for_texts(loaded, unique_texts)
  doc_vecs = encode_texts_to_doc_vectors(
    loaded.model,
    unique_texts,
    prepared,
    device=loaded.device,
  )
  text_to_vec = {
    text: doc_vecs[i]
    for i, text in enumerate(unique_texts)
  }
  v_src = text_to_vec[source_text]
  len_s = torch.tensor([float(len(source_text))], dtype=torch.float32, device=loaded.device)

  scores: list[float] = []
  for start in range(0, len(candidates), batch_size):
    batch = candidates[start : start + batch_size]
    v_cands = torch.stack([text_to_vec[c] for c in batch], dim=0)
    len_c = torch.tensor(
      [float(len(c)) for c in batch], dtype=torch.float32, device=loaded.device
    )
    v_src_batch = v_src.unsqueeze(0).expand(len(batch), -1)
    len_s_batch = len_s.expand(len(batch))
    s = loaded.model.score_from_doc_vectors_fast(
      v_src_batch,
      v_cands,
      len_source=len_s_batch,
      len_candidate=len_c,
    )
    scores.extend([float(x) for x in s.cpu().tolist()])
  return scores
