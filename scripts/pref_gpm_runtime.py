#!/usr/bin/env python3
"""GPM 選好モデル（pref-gpm）の読み込みと対スコアリング。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from pref_static_utils import encode_texts, load_sentence_model_from_artifact
from sentseq_utils import split_document_sentences, truncate_sentence_units
from train_pref_gpm import GpmRewardModel, load_gpm_model_from_artifact
from train_pref_sentseq import encode_texts_to_doc_vectors, prepare_sentence_data


@dataclass
class LoadedGpmModel:
  model: GpmRewardModel
  sentence_model: SentenceTransformer
  normalize_embeddings: bool
  text_prefix: str
  max_sents: int
  encode_batch_size: int
  device: torch.device
  model_dir: str


def load_gpm_model(
  model_dir: Path,
  *,
  device: str = "",
  encode_batch_size: int = 64,
) -> LoadedGpmModel:
  artifact_path = model_dir / "model.pt"
  if not artifact_path.is_file():
    raise FileNotFoundError(f"model not found: {artifact_path}")
  artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
  if artifact.get("kind") != "pref-gpm":
    raise ValueError(f"not a pref-gpm artifact: {artifact_path}")

  config = artifact["config"]
  resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
  dev = torch.device(resolved)
  model = load_gpm_model_from_artifact(artifact, device=dev)
  sent_cfg = {
    "sentence_model_name": config["sentence_model_name"],
    "truncate_dim": config.get("truncate_dim"),
    "max_seq_length": config.get("max_seq_length"),
  }
  sentence_model = load_sentence_model_from_artifact(sent_cfg, device=str(dev))
  for param in sentence_model.parameters():
    param.requires_grad = False

  return LoadedGpmModel(
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
  loaded: LoadedGpmModel,
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
  loaded: LoadedGpmModel,
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
def _doc_vectors_for_texts(
  loaded: LoadedGpmModel,
  texts: list[str],
) -> dict[str, torch.Tensor]:
  if not texts:
    return {}
  unique_texts: list[str] = []
  seen: set[str] = set()
  for text in texts:
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
  return {text: doc_vecs[i] for i, text in enumerate(unique_texts)}


@torch.no_grad()
def preference_scores(
  loaded: LoadedGpmModel,
  source_text: str,
  pairs: list[tuple[str, str]],
) -> list[float]:
  if not pairs:
    return []
  texts = [source_text]
  for a, b in pairs:
    for text in (a, b):
      if text not in texts:
        texts.append(text)
  vec_map = _doc_vectors_for_texts(loaded, texts)
  v_src = vec_map[source_text]
  len_s = torch.tensor([float(len(source_text))], dtype=torch.float32, device=loaded.device)
  scores: list[float] = []
  for a, b in pairs:
    v_a = vec_map[a]
    v_b = vec_map[b]
    len_a = torch.tensor([float(len(a))], dtype=torch.float32, device=loaded.device)
    len_b = torch.tensor([float(len(b))], dtype=torch.float32, device=loaded.device)
    emb_a = loaded.model.embed_from_doc_vectors(
      v_src.unsqueeze(0),
      v_a.unsqueeze(0),
      len_source=len_s,
      len_candidate=len_a,
    )
    emb_b = loaded.model.embed_from_doc_vectors(
      v_src.unsqueeze(0),
      v_b.unsqueeze(0),
      len_source=len_s,
      len_candidate=len_b,
    )
    s = loaded.model.pair_score(emb_a, emb_b)
    scores.append(float(s.item()))
  return scores


@torch.no_grad()
def pairwise_matrix(
  loaded: LoadedGpmModel,
  source_text: str,
  candidates: list[str],
) -> list[list[float]]:
  n = len(candidates)
  if n == 0:
    return []
  pairs = [(candidates[i], candidates[j]) for i in range(n) for j in range(n)]
  flat = preference_scores(loaded, source_text, pairs)
  matrix: list[list[float]] = []
  idx = 0
  for _i in range(n):
    row: list[float] = []
    for _j in range(n):
      row.append(flat[idx])
      idx += 1
    matrix.append(row)
  return matrix


@torch.no_grad()
def score_vs_draft(
  loaded: LoadedGpmModel,
  source_text: str,
  candidates: list[str],
) -> list[float]:
  if not candidates:
    return []
  draft = source_text
  return preference_scores(
    loaded,
    source_text,
    [(candidate, draft) for candidate in candidates],
  )
