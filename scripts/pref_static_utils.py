#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path: str, limit: int = -1) -> list[dict]:
  rows: list[dict] = []
  with Path(path).open(encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
      if 0 < limit <= len(rows):
        break
  return rows


def collect_unique_texts(rows: list[dict]) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for row in rows:
    for key in ("source_text", "candidate_a", "candidate_b"):
      value = str(row[key])
      if value not in seen:
        seen.add(value)
        texts.append(value)
  return texts


def normalize_truncate_dim(value) -> int | None:
  """0 / 負値 / None は切り捨てなし。"""
  if value is None:
    return None
  dim = int(value)
  return dim if dim > 0 else None


def load_sentence_model_from_artifact(artifact: dict, *, device: str = "cpu") -> SentenceTransformer:
  truncate_dim = normalize_truncate_dim(artifact.get("truncate_dim"))
  model = SentenceTransformer(
    artifact["sentence_model_name"],
    device=device,
    truncate_dim=truncate_dim,
  )
  max_seq_length = artifact.get("max_seq_length")
  if max_seq_length:
    model.max_seq_length = int(max_seq_length)
  return model


def encode_texts(
  model,
  texts: list[str],
  *,
  batch_size: int,
  normalize_embeddings: bool,
  text_prefix: str = "",
  show_progress_bar: bool = False,
) -> np.ndarray:
  if not texts:
    return np.zeros((0, 0), dtype=np.float32)
  payloads = [text_prefix + text for text in texts]
  embeddings = model.encode(
    payloads,
    batch_size=batch_size,
    convert_to_numpy=True,
    show_progress_bar=show_progress_bar,
    normalize_embeddings=normalize_embeddings,
  )
  return np.asarray(embeddings, dtype=np.float32)


def encode_text_map(
  model,
  texts: list[str],
  *,
  batch_size: int,
  normalize_embeddings: bool,
  text_prefix: str = "",
  show_progress_bar: bool = True,
) -> dict[str, np.ndarray]:
  """文面→埋め込み。**同一文字列が複数回出るときは語の並び順が失われ、最後のベクトルのみになる**ので、順序付きでの推論では `encode_texts` を使う。"""
  if not texts:
    return {}
  embeddings = encode_texts(
    model,
    texts,
    batch_size=batch_size,
    normalize_embeddings=normalize_embeddings,
    text_prefix=text_prefix,
    show_progress_bar=show_progress_bar,
  )
  return {text: np.asarray(emb, dtype=np.float32) for text, emb in zip(texts, embeddings, strict=True)}


def assemble_pointwise_feature_vector(
  source: np.ndarray,
  candidate: np.ndarray,
  *,
  len_source: float,
  len_candidate: float,
) -> np.ndarray:
  """候補 1 件ぶんのスコア特徴（BT 報酬モデル用）。"""
  diff = source - candidate
  abs_diff = np.abs(diff)
  cos = float(np.dot(source, candidate))
  numeric = np.asarray(
    [
      cos,
      len_source,
      len_candidate,
      len_candidate - len_source,
      abs(len_candidate - len_source),
    ],
    dtype=np.float32,
  )
  return np.concatenate([source, candidate, diff, abs_diff, numeric]).astype(np.float32, copy=False)


def assemble_preference_feature_vector(
  source: np.ndarray,
  cand_a: np.ndarray,
  cand_b: np.ndarray,
  *,
  len_source: float,
  len_a: float,
  len_b: float,
) -> np.ndarray:
  """1 行ぶんの静的選好特徴（`train_pref_static` と同じ式）。"""
  diff_ab = cand_a - cand_b
  abs_diff_ab = np.abs(diff_ab)
  diff_sa = source - cand_a
  diff_sb = source - cand_b
  abs_diff_sa = np.abs(diff_sa)
  abs_diff_sb = np.abs(diff_sb)

  cos_sa = float(np.dot(source, cand_a))
  cos_sb = float(np.dot(source, cand_b))
  cos_ab = float(np.dot(cand_a, cand_b))

  numeric = np.asarray(
    [
      cos_sa,
      cos_sb,
      cos_ab,
      cos_sa - cos_sb,
      len_source,
      len_a,
      len_b,
      len_a - len_b,
      abs(len_a - len_b),
    ],
    dtype=np.float32,
  )

  out = np.concatenate(
    [
      source,
      cand_a,
      cand_b,
      diff_ab,
      abs_diff_ab,
      diff_sa,
      diff_sb,
      abs_diff_sa,
      abs_diff_sb,
      numeric,
    ]
  )
  return out.astype(np.float32, copy=False)


def preference_ab_scores_from_predict_proba(classes: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
  """`classes_` の列順に依存せず、`label==1` を「先頭スロット側が良い」・`label==0` を「末尾スロット側が良い」として確率を取り出す。

  （学習データでは先頭=`candidate_a`・末尾=`candidate_b`。）
  """
  mapping = {int(c): float(p) for c, p in zip(np.ravel(classes).tolist(), np.ravel(probs).tolist(), strict=True)}
  return mapping.get(1, 0.0), mapping.get(0, 0.0)


def aggregate_symmetric_candidate_scores(
  p_first_when_ab_ordered: tuple[float, float],
  p_first_when_ba_ordered: tuple[float, float],
) -> tuple[float, float, dict[str, float]]:
  """(ソース, A, B) と (ソース, B, A) の2回の順序別予測から、特定のドラフト A/B に対応する合成スコアを返す。

  * `p_first_when_ab_ordered` は (ソース, cand_first=A, cand_second=B) での `(P(label=1), P(label=0))`。
  * `p_first_when_ba_ordered` は (ソース, cand_first=B, cand_second=A) での同種。

  Returns:
    `(score_for_original_a, score_for_original_b, detail)`
  """
  prob_a_wins_given_order_ab = float(p_first_when_ab_ordered[0])
  prob_b_wins_given_order_ab = float(p_first_when_ab_ordered[1])
  prob_b_wins_given_order_ba = float(p_first_when_ba_ordered[0])
  prob_a_wins_given_order_ba = float(p_first_when_ba_ordered[1])
  score_original_a = 0.5 * (prob_a_wins_given_order_ab + prob_a_wins_given_order_ba)
  score_original_b = 0.5 * (prob_b_wins_given_order_ab + prob_b_wins_given_order_ba)
  detail = {
    "prob_a_order_AB": prob_a_wins_given_order_ab,
    "prob_b_order_AB": prob_b_wins_given_order_ab,
    "prob_first_is_B_order_BAfirst": prob_b_wins_given_order_ba,
    "prob_second_is_A_order_BAfirst": prob_a_wins_given_order_ba,
  }
  return score_original_a, score_original_b, detail


def static_pair_preference_inference(
  classifier,
  sentence_model,
  source_text: str,
  candidate_a: str,
  candidate_b: str,
  *,
  normalize_embeddings: bool,
  symmetric: bool,
  encode_batch_size: int | None = None,
  text_prefix: str = "",
) -> tuple[float, float, dict[str, float]]:
  """`train_pref_static` で学習した分類器で、(candidate_a を先頭スロットとみなした) ペア単位スコア・対称合成スコアを返す。

  symmetric=False のときのスコアは学習時の並びそのまま。「先頭=`candidate_a` が良い」確率・「末尾=`candidate_b` が良い」確率。
  """
  clf = classifier
  classes = np.asarray(clf.classes_)
  len_s = float(len(source_text))
  len_a = float(len(candidate_a))
  len_b = float(len(candidate_b))

  if not symmetric:
    bs = encode_batch_size if encode_batch_size is not None else 3
    embeddings = encode_texts(
      sentence_model,
      [source_text, candidate_a, candidate_b],
      batch_size=bs,
      normalize_embeddings=normalize_embeddings,
      text_prefix=text_prefix,
    )
    v_src = embeddings[0]
    v_a = embeddings[1]
    v_b = embeddings[2]
    feat_row = assemble_preference_feature_vector(
      v_src, v_a, v_b, len_source=len_s, len_a=len_a, len_b=len_b
    ).reshape(1, -1)
    probs = clf.predict_proba(feat_row)[0]
    p1, p0 = preference_ab_scores_from_predict_proba(classes, probs)
    return float(p1), float(p0), {}

  bs = encode_batch_size if encode_batch_size is not None else 3
  embeddings = encode_texts(
    sentence_model,
    [source_text, candidate_a, candidate_b],
    batch_size=bs,
    normalize_embeddings=normalize_embeddings,
    text_prefix=text_prefix,
  )
  v_src = embeddings[0]
  v_a = embeddings[1]
  v_b = embeddings[2]
  feat_ab = assemble_preference_feature_vector(
    v_src, v_a, v_b, len_source=len_s, len_a=len_a, len_b=len_b
  ).reshape(1, -1)
  feat_ba = assemble_preference_feature_vector(
    v_src,
    v_b,
    v_a,
    len_source=len_s,
    len_a=len_b,
    len_b=len_a,
  ).reshape(1, -1)
  probs2 = clf.predict_proba(np.vstack([feat_ab, feat_ba]))
  pair_ab = preference_ab_scores_from_predict_proba(classes, probs2[0])
  pair_ba = preference_ab_scores_from_predict_proba(classes, probs2[1])
  score_a, score_b, detail = aggregate_symmetric_candidate_scores(pair_ab, pair_ba)
  return float(score_a), float(score_b), detail


def build_feature_matrix(rows: list[dict], text_to_embedding: dict[str, np.ndarray]) -> np.ndarray:
  features: list[np.ndarray] = []
  for row in rows:
    source = text_to_embedding[row["source_text"]]
    cand_a = text_to_embedding[row["candidate_a"]]
    cand_b = text_to_embedding[row["candidate_b"]]

    feature = assemble_preference_feature_vector(
      source,
      cand_a,
      cand_b,
      len_source=float(len(row["source_text"])),
      len_a=float(len(row["candidate_a"])),
      len_b=float(len(row["candidate_b"])),
    )
    features.append(feature)
  return np.vstack(features)


def build_label_array(rows: list[dict]) -> np.ndarray:
  return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
