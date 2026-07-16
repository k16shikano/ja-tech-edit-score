#!/usr/bin/env python3
"""Batch inference helpers and optional in-process model cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer

from pref_static_utils import (
  aggregate_symmetric_candidate_scores,
  assemble_preference_feature_vector,
  encode_texts,
  load_sentence_model_from_artifact,
  preference_ab_scores_from_predict_proba,
)


@dataclass
class LoadedPrefModel:
  classifier: object
  sentence_model: SentenceTransformer
  normalize_embeddings: bool
  text_prefix: str
  model_dir: str


def load_pref_model(model_dir: Path) -> LoadedPrefModel:
  artifact_path = model_dir / "model.joblib"
  if not artifact_path.is_file():
    raise FileNotFoundError(f"model not found: {artifact_path}")
  artifact = load(artifact_path)
  sentence_model = load_sentence_model_from_artifact(artifact, device="cpu")
  return LoadedPrefModel(
    classifier=artifact["classifier"],
    sentence_model=sentence_model,
    normalize_embeddings=bool(artifact["normalize_embeddings"]),
    text_prefix=str(artifact.get("text_prefix", "")),
    model_dir=str(model_dir),
  )


def _encode_unique(
  sentence_model: SentenceTransformer,
  texts: list[str],
  *,
  normalize_embeddings: bool,
  text_prefix: str,
  batch_size: int,
) -> dict[str, np.ndarray]:
  if not texts:
    return {}
  embeddings = encode_texts(
    sentence_model,
    texts,
    batch_size=batch_size,
    normalize_embeddings=normalize_embeddings,
    text_prefix=text_prefix,
  )
  return {
    text: np.asarray(vec, dtype=np.float32)
    for text, vec in zip(texts, embeddings, strict=True)
  }


def score_edit_vs_base_batch(
  pairs: list[tuple[str, str]],
  loaded: LoadedPrefModel,
  *,
  batch_size: int = 64,
) -> list[tuple[float, float, dict[str, float]]]:
  """Score many (source, edited) hunks with one embedding pass + one predict_proba."""
  if not pairs:
    return []

  unique: list[str] = []
  seen: set[str] = set()
  for source, edited in pairs:
    for text in (source, edited):
      if text not in seen:
        seen.add(text)
        unique.append(text)

  emb_map = _encode_unique(
    loaded.sentence_model,
    unique,
    normalize_embeddings=loaded.normalize_embeddings,
    text_prefix=loaded.text_prefix,
    batch_size=batch_size,
  )

  classes = np.asarray(loaded.classifier.classes_)
  feature_rows: list[np.ndarray] = []
  for source, edited in pairs:
    v_src = emb_map[source]
    v_edit = emb_map[edited]
    v_base = emb_map[source]
    len_s = float(len(source))
    len_e = float(len(edited))
    feature_rows.append(
      assemble_preference_feature_vector(
        v_src, v_edit, v_base, len_source=len_s, len_a=len_e, len_b=len_s
      )
    )
    feature_rows.append(
      assemble_preference_feature_vector(
        v_src, v_base, v_edit, len_source=len_s, len_a=len_s, len_b=len_e
      )
    )

  probs = loaded.classifier.predict_proba(np.vstack(feature_rows))
  results: list[tuple[float, float, dict[str, float]]] = []
  for i in range(len(pairs)):
    pair_ab = preference_ab_scores_from_predict_proba(classes, probs[i * 2])
    pair_ba = preference_ab_scores_from_predict_proba(classes, probs[i * 2 + 1])
    score_edit, score_base, detail = aggregate_symmetric_candidate_scores(pair_ab, pair_ba)
    results.append((float(score_edit), float(score_base), detail))
  return results
