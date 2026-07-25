#!/usr/bin/env python3
"""報酬モデル（pref-bt / pref-ce / pref-sentseq）の自動判別ローダ。

- meta.json に kind: pref-ce があれば cross-encoder
- model.pt があれば文列 Transformer（pref-sentseq）
- それ以外は従来の BT（凍結埋め込み + 線形ヘッド、model.joblib）
どれも score(source, candidates) -> list[float] の同じ形で返す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class LoadedScorer:
  kind: str  # "bt" | "ce" | "sentseq"
  model_dir: str
  score: Callable[..., list[float]]  # score(source, candidates, *, batch_size=...)


def detect_scorer_kind(model_dir: Path) -> str:
  meta_path = model_dir / "meta.json"
  if meta_path.is_file():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("kind") == "pref-ce":
      return "ce"
  if (model_dir / "model.pt").is_file():
    return "sentseq"
  return "bt"


def load_scorer(model_dir: Path) -> LoadedScorer:
  kind = detect_scorer_kind(model_dir)
  if kind == "ce":
    from pref_ce_runtime import load_ce_model, score_candidates_ce

    loaded = load_ce_model(model_dir)

    def score(source: str, candidates: list[str], *, batch_size: int = 16) -> list[float]:
      return score_candidates_ce(loaded, source, candidates, batch_size=batch_size)

  elif kind == "sentseq":
    from pref_sentseq_runtime import load_sentseq_model, score_candidates_sentseq

    loaded = load_sentseq_model(model_dir)

    def score(source: str, candidates: list[str], *, batch_size: int = 16) -> list[float]:
      return score_candidates_sentseq(loaded, source, candidates, batch_size=batch_size)

  else:
    from pref_bt_runtime import load_bt_model, score_candidates_bt

    loaded = load_bt_model(model_dir)

    def score(source: str, candidates: list[str], *, batch_size: int = 32) -> list[float]:
      return score_candidates_bt(loaded, source, candidates, batch_size=batch_size)

  return LoadedScorer(kind=kind, model_dir=str(model_dir), score=score)
