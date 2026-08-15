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
from train_pref_nce import (
  KIND,
  NceTrainConfig,
  PrefNceModel,
  nce_loss,
  save_nce_model,
  train_nce_model,
)
from train_pref_sentseq import SentSeqEncoder, collect_unique_sentences


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


def test_query_is_not_raw_draft_embedding() -> None:
  torch.manual_seed(0)
  encoder = SentSeqEncoder(
    embed_dim=8,
    d_model=16,
    nhead=2,
    num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    max_sents=8,
  )
  model = PrefNceModel(encoder, tau=0.07)
  draft_vec = torch.randn(5, 16)
  query = model.queries_from_draft(draft_vec)
  raw = F.normalize(draft_vec, dim=-1)
  assert not torch.allclose(query, raw, atol=1e-3)
  cosine = (query * raw).sum(dim=-1)
  assert not torch.allclose(cosine, torch.ones_like(cosine), atol=1e-3)


def test_nce_loss_treats_human_as_positive() -> None:
  logits = torch.tensor(
    [
      [4.0, 0.1, 0.2],
      [3.0, 0.0, 1.0],
      [5.0, 1.0, 0.5],
    ]
  )
  loss_pos = nce_loss(logits)
  loss_neg = nce_loss(logits[:, [1, 0, 2]])
  assert float(loss_pos) < float(loss_neg)


def test_infonce_prefers_human_on_styled_embeddings(tmp_path: Path) -> None:
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
  cfg = NceTrainConfig(
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
    tau=0.07,
  )
  model, train_metrics, valid_metrics = train_nce_model(
    train,
    valid,
    cfg,
    device=torch.device("cpu"),
    precomputed_embeddings=embs,
  )
  assert train_metrics["best_epoch"] >= 1
  assert valid_metrics["n_draft_over_human"] == 0
  assert valid_metrics["n_composer_over_human"] == 0
  save_nce_model(model, cfg, embed_dim=16, output_dir=tmp_path)
  meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
  assert meta["kind"] == KIND
  assert meta["tau"] == 0.07
  assert detect_scorer_kind(tmp_path) == "nce"


def test_detect_scorer_kind_reads_nce_before_model_pt(tmp_path: Path) -> None:
  (tmp_path / "model.pt").write_bytes(b"not-a-real-checkpoint")
  (tmp_path / "meta.json").write_text(
    json.dumps({"kind": "pref-nce"}, ensure_ascii=False),
    encoding="utf-8",
  )
  assert detect_scorer_kind(tmp_path) == "nce"
  sentseq_dir = tmp_path / "sentseq"
  sentseq_dir.mkdir()
  (sentseq_dir / "model.pt").write_bytes(b"x")
  assert detect_scorer_kind(sentseq_dir) == "sentseq"
