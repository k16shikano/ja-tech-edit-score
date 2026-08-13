#!/usr/bin/env python3
"""Cross-encoder 報酬モデル（pref-ce）の読み込みとスコアリング。

pref_bt_runtime と同じ関数形。CE が LOPO で pref-bt に勝ったら
rank / converge から差し替えて使う。CPU でも動く（130m なら数秒/候補程度）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LoadedCeModel:
  model: object
  tokenizer: object
  max_length: int
  model_dir: str


def load_ce_model(model_dir: Path, *, device: str = "") -> LoadedCeModel:
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  meta_path = model_dir / "meta.json"
  if not meta_path.is_file():
    raise FileNotFoundError(f"meta not found: {meta_path}")
  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  if meta.get("kind") != "pref-ce":
    raise ValueError(f"not a pref-ce artifact: {meta_path}")

  resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
  model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model")
  tokenizer = AutoTokenizer.from_pretrained(model_dir / "model")
  model.to(resolved)
  model.eval()
  return LoadedCeModel(
    model=model,
    tokenizer=tokenizer,
    max_length=int(meta.get("max_length", 512)),
    model_dir=str(model_dir),
  )


@torch.no_grad()
def score_candidates_ce(
  loaded: LoadedCeModel,
  source_text: str,
  candidates: list[str],
  *,
  batch_size: int = 16,
) -> list[float]:
  """各候補について s(source, candidate) を返す。"""
  if not candidates:
    return []
  device = next(loaded.model.parameters()).device
  scores: list[float] = []
  for start in range(0, len(candidates), batch_size):
    batch = candidates[start : start + batch_size]
    encoded = loaded.tokenizer(
      [source_text] * len(batch),
      batch,
      padding=True,
      truncation="longest_first",
      max_length=loaded.max_length,
      return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits = loaded.model(**encoded).logits.squeeze(-1)
    scores.extend([float(s) for s in logits.float().cpu().tolist()])
  return scores
