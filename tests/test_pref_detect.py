from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from pref_scorer import detect_scorer_kind
from setwise_triple_utils import SetwiseTriple
from train_pref_detect import (
  KIND,
  detect_objective,
  labeled_examples_from_triples,
  save_detect_model,
  train_detect_model,
)
from train_pref_sentseq import SentSeqTrainConfig, collect_unique_sentences


def _triples(n: int) -> list[SetwiseTriple]:
  out: list[SetwiseTriple] = []
  for i in range(n):
    draft = f"下書き{i}の文である。"
    human = f"人間{i}が直した文である。"
    composer = f"生成{i}が直した文である。"
    out.append(
      SetwiseTriple(
        item_id=f"id-{i}",
        source_text=draft,
        draft=draft,
        human=human,
        composer=composer,
        meta={},
      )
    )
  return out


def _styled_embeddings(triples: list[SetwiseTriple], *, dim: int = 16, seed: int = 0) -> dict[str, np.ndarray]:
  texts = [t.draft for t in triples] + [t.human for t in triples] + [t.composer for t in triples]
  sents = collect_unique_sentences(texts, max_sents=8)
  rng = np.random.default_rng(seed)
  style_h = rng.standard_normal(dim).astype(np.float32)
  style_c = rng.standard_normal(dim).astype(np.float32)
  embs: dict[str, np.ndarray] = {}
  for sent in sents:
    vec = rng.standard_normal(dim).astype(np.float32)
    if sent.startswith("人間"):
      vec = vec + 4.0 * style_h
    elif sent.startswith("生成"):
      vec = vec + 4.0 * style_c
    embs[sent] = (vec / np.linalg.norm(vec)).astype(np.float32)
  return embs


def test_labeled_examples_are_human_one_others_zero() -> None:
  triples = _triples(2)
  examples = labeled_examples_from_triples(triples)
  assert len(examples) == 6
  by_role = {(row["item_id"], row["role"]): row for row in examples}
  for triple in triples:
    human = by_role[(triple.item_id, "human")]
    draft = by_role[(triple.item_id, "draft")]
    composer = by_role[(triple.item_id, "composer")]
    assert human["label"] == 1.0
    assert draft["label"] == 0.0
    assert composer["label"] == 0.0
    assert human["source_text"] == triple.draft
    assert draft["source_text"] == triple.draft
    assert composer["source_text"] == triple.draft
    assert human["candidate"] == triple.human
    assert draft["candidate"] == triple.draft
    assert composer["candidate"] == triple.composer
  assert all(row["source_text"] == row["candidate"] or row["role"] != "draft" for row in examples)
  assert not any(row["source_text"] == row["candidate"] and row["role"] == "composer" for row in examples)


def test_bce_treats_human_as_positive() -> None:
  logits = torch.tensor([4.0, -1.0, -1.0])
  labels = torch.tensor([1.0, 0.0, 0.0])
  loss_pos = F.binary_cross_entropy_with_logits(logits, labels)
  loss_neg = F.binary_cross_entropy_with_logits(logits, torch.tensor([0.0, 1.0, 0.0]))
  assert float(loss_pos) < float(loss_neg)


def test_composer_over_draft_term_lowers_when_composer_is_above_draft() -> None:
  human = torch.tensor([3.0, 3.0])
  draft = torch.tensor([0.0, 0.0])
  composer_low = torch.tensor([-1.0, -1.0])
  composer_high = torch.tensor([1.0, 1.0])
  loss_low, bce_low, cd_low = detect_objective(
    human, draft, composer_low, composer_over_draft=True
  )
  loss_high, bce_high, cd_high = detect_objective(
    human, draft, composer_high, composer_over_draft=True
  )
  assert cd_low is not None and cd_high is not None
  assert float(cd_high) < float(cd_low)
  assert float(loss_high) < float(loss_low)
  loss_off, bce_off, cd_off = detect_objective(
    human, draft, composer_low, composer_over_draft=False
  )
  assert cd_off is None
  assert abs(float(loss_off) - float(bce_off)) < 1e-6
  assert abs(float(loss_low) - (float(bce_low) + float(cd_low))) < 1e-6
  assert abs(float(loss_high) - (float(bce_high) + float(cd_high))) < 1e-6


def test_detect_prefers_human_on_styled_embeddings(tmp_path: Path) -> None:
  train = _triples(8)
  valid = []
  for i in range(4):
    k = i + 80
    draft = f"下書き{k}の文である。"
    human = f"人間{k}が直した文である。"
    composer = f"生成{k}が直した文である。"
    valid.append(
      SetwiseTriple(
        item_id=f"id-{k}",
        source_text=draft,
        draft=draft,
        human=human,
        composer=composer,
        meta={},
      )
    )
  embs = _styled_embeddings(train + valid, dim=16, seed=1)
  cfg = SentSeqTrainConfig(
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=64,
    dropout=0.0,
    max_sents=8,
    batch_size=4,
    epochs=12,
    lr=1e-3,
    seed=0,
  )
  model, train_metrics, valid_metrics = train_detect_model(
    train,
    valid,
    cfg,
    device=torch.device("cpu"),
    precomputed_embeddings=embs,
  )
  assert train_metrics["best_epoch"] >= 1
  assert valid_metrics["n_draft_over_human"] == 0
  assert valid_metrics["n_composer_over_human"] == 0
  save_detect_model(model, cfg, embed_dim=16, output_dir=tmp_path)
  meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
  assert meta["kind"] == KIND
  assert meta["loss"] == "bce_human_one_draft_composer_zero"
  assert detect_scorer_kind(tmp_path) == "detect"


def test_detect_scorer_kind_reads_detect_before_model_pt(tmp_path: Path) -> None:
  (tmp_path / "model.pt").write_bytes(b"not-a-real-checkpoint")
  (tmp_path / "meta.json").write_text(
    json.dumps({"kind": "pref-detect"}, ensure_ascii=False),
    encoding="utf-8",
  )
  assert detect_scorer_kind(tmp_path) == "detect"
  sentseq_dir = tmp_path / "sentseq"
  sentseq_dir.mkdir()
  (sentseq_dir / "model.pt").write_bytes(b"x")
  assert detect_scorer_kind(sentseq_dir) == "sentseq"


def test_detect_with_composer_over_draft_keeps_human_top(tmp_path: Path) -> None:
  train = _triples(8)
  valid = []
  for i in range(4):
    k = i + 80
    draft = f"下書き{k}の文である。"
    human = f"人間{k}が直した文である。"
    composer = f"生成{k}が直した文である。"
    valid.append(
      SetwiseTriple(
        item_id=f"id-{k}",
        source_text=draft,
        draft=draft,
        human=human,
        composer=composer,
        meta={},
      )
    )
  embs = _styled_embeddings(train + valid, dim=16, seed=1)
  cfg = SentSeqTrainConfig(
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=64,
    dropout=0.0,
    max_sents=8,
    batch_size=4,
    epochs=12,
    lr=1e-3,
    seed=0,
  )
  _model, train_metrics, valid_metrics = train_detect_model(
    train,
    valid,
    cfg,
    device=torch.device("cpu"),
    precomputed_embeddings=embs,
    composer_over_draft=True,
  )
  assert train_metrics["best_epoch"] >= 1
  assert train_metrics["composer_over_draft"] is True
  assert train_metrics["train_loss"] > train_metrics["train_bce_loss"]
  assert valid_metrics["n_draft_over_human"] == 0
  save_detect_model(
    _model, cfg, embed_dim=16, output_dir=tmp_path, composer_over_draft=True
  )
  meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
  assert meta["loss"] == "bce_human_one_plus_composer_over_draft"
  assert meta["composer_over_draft"] is True
