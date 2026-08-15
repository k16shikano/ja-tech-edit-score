#!/usr/bin/env python3
"""InfoNCE 評価器（pref-nce）の読み込みとスコアリング。

点は cos(q(draft), v(candidate))。q は下書きの文書ベクトルを写したクエリ。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from pref_static_utils import encode_texts, load_sentence_model_from_artifact
from sentseq_utils import split_document_sentences, truncate_sentence_units
from train_pref_nce import (
  PrefNceModel,
  encode_texts_with_encoder,
  load_nce_model_from_artifact,
)
from train_pref_sentseq import prepare_sentence_data


@dataclass
class LoadedNceModel:
  model: PrefNceModel
  sentence_model: SentenceTransformer
  normalize_embeddings: bool
  text_prefix: str
  max_sents: int
  encode_batch_size: int
  device: torch.device
  model_dir: str


def load_nce_model(
  model_dir: Path,
  *,
  device: str = "",
  encode_batch_size: int = 64,
) -> LoadedNceModel:
  artifact_path = model_dir / "model.pt"
  if not artifact_path.is_file():
    raise FileNotFoundError(f"model not found: {artifact_path}")
  artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
  if artifact.get("kind") != "pref-nce":
    raise ValueError(f"not a pref-nce artifact: {artifact_path}")

  config = artifact["config"]
  resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
  dev = torch.device(resolved)
  model = load_nce_model_from_artifact(artifact, device=dev)
  sent_cfg = {
    "sentence_model_name": config["sentence_model_name"],
    "truncate_dim": config.get("truncate_dim"),
    "max_seq_length": config.get("max_seq_length"),
  }
  sentence_model = load_sentence_model_from_artifact(sent_cfg, device=str(dev))
  for param in sentence_model.parameters():
    param.requires_grad = False

  return LoadedNceModel(
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
  loaded: LoadedNceModel,
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


def _prepare_for_texts(loaded: LoadedNceModel, texts: list[str]):
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
def score_candidates_nce(
  loaded: LoadedNceModel,
  source_text: str,
  candidates: list[str],
  *,
  batch_size: int = 16,
) -> list[float]:
  """各候補について cos(q(source), v(candidate)) を返す。"""
  if not candidates:
    return []

  unique_texts: list[str] = [source_text]
  seen = {source_text}
  for text in candidates:
    if text not in seen:
      seen.add(text)
      unique_texts.append(text)

  prepared = _prepare_for_texts(loaded, unique_texts)
  doc_vecs = encode_texts_with_encoder(
    loaded.model.encoder,
    unique_texts,
    prepared,
    device=loaded.device,
  )
  text_to_vec = {text: doc_vecs[i] for i, text in enumerate(unique_texts)}
  query = loaded.model.queries_from_draft(text_to_vec[source_text].unsqueeze(0))

  scores: list[float] = []
  for start in range(0, len(candidates), batch_size):
    batch = candidates[start : start + batch_size]
    v_cands = F.normalize(
      torch.stack([text_to_vec[c] for c in batch], dim=0),
      dim=-1,
    )
    q_batch = query.expand(len(batch), -1)
    s = (q_batch * v_cands).sum(dim=-1)
    scores.extend([float(x) for x in s.cpu().tolist()])
  return scores
