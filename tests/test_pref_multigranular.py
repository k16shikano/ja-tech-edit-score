from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from sentseq_utils import split_document_sentences
from setwise_model import (
  LOSS_MODE_HUMAN_TOP,
  SCORE_MODE_INDEPENDENT,
  SCORE_MODE_JOINT,
  SetwiseRankingModel,
  build_setwise_batch_tensors,
  candidate_logits,
  compute_setwise_metrics,
  human_top_loss,
  pair_checkpoint_selection_key,
  public_relative_scores,
)
from setwise_triple_utils import ROLE_DRAFT, ROLE_HUMAN, SetwiseTriple
from train_pref_multigranular import (
  STAGE_PAIR_HUMANTOP,
  examples_from_triples,
  train_multigranular_model,
)
from train_pref_sentseq import prepare_sentence_data
from train_pref_setwise import SetwiseTrainConfig


def _prepared(texts: list[str], *, embed_dim: int = 8, max_sents: int = 8, seed: int = 0):
  torch.manual_seed(seed)
  sent_to_embedding: dict[str, object] = {}
  for text in texts:
    for unit in split_document_sentences(text):
      if unit.text not in sent_to_embedding:
        sent_to_embedding[unit.text] = torch.randn(embed_dim).numpy()
  prepared = prepare_sentence_data(
    texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=max_sents,
    device=torch.device("cpu"),
  )
  return prepared, sent_to_embedding


def test_independent_human_score_ignores_composer_text() -> None:
  source = "下書き一。下書き二。"
  human = "人間一。人間二。"
  draft = "下書き一。下書き二。"
  composer_a = "生成甲。"
  composer_b = "生成乙。生成乙二。"
  prepared, _ = _prepared([source, human, draft, composer_a, composer_b])
  model = SetwiseRankingModel(embed_dim=8, d_model=32, nhead=4, dropout=0.0, max_sents=8)
  model.eval()
  t_a = build_setwise_batch_tensors(
    source_texts=[source],
    candidate_texts=[[human, composer_a, draft]],
    prepared=prepared,
    embed_dim=8,
    device=torch.device("cpu"),
  )
  t_b = build_setwise_batch_tensors(
    source_texts=[source],
    candidate_texts=[[human, composer_b, draft]],
    prepared=prepared,
    embed_dim=8,
    device=torch.device("cpu"),
  )
  with torch.no_grad():
    ind_a = candidate_logits(model, t_a, score_mode=SCORE_MODE_INDEPENDENT)
    ind_b = candidate_logits(model, t_b, score_mode=SCORE_MODE_INDEPENDENT)
    joint_a = candidate_logits(model, t_a, score_mode=SCORE_MODE_JOINT)
    joint_b = candidate_logits(model, t_b, score_mode=SCORE_MODE_JOINT)
  assert torch.allclose(ind_a[0, 0], ind_b[0, 0], atol=1e-5)
  assert torch.allclose(ind_a[0, 2], ind_b[0, 2], atol=1e-5)
  assert not torch.allclose(joint_a[0, 0], joint_b[0, 0], atol=1e-5)


def test_build_batch_allows_single_candidate() -> None:
  texts = ["下書き。", "候補。"]
  prepared, _ = _prepared(texts)
  batch = build_setwise_batch_tensors(
    source_texts=["下書き。"],
    candidate_texts=[["候補。"]],
    prepared=prepared,
    embed_dim=8,
    device=torch.device("cpu"),
  )
  assert batch.candidate_sent_emb.shape[1] == 1


def test_public_relative_scores_preserve_order() -> None:
  raw = torch.tensor([[3.0, 1.0, 0.0]])
  self_logit = torch.tensor([0.5])
  rel = public_relative_scores(raw, self_logit)
  assert torch.equal(torch.argsort(raw, descending=True), torch.argsort(rel, descending=True))


def test_human_among_top_allows_tie() -> None:
  logits = torch.tensor([[2.0, 2.0, 0.0]])
  rank = torch.tensor([[0, 1, 2]])
  metrics = compute_setwise_metrics(
    logits,
    rank,
    human_index=torch.tensor([0]),
    composer_index=torch.tensor([1]),
    draft_index=torch.tensor([2]),
  )
  assert metrics["human_top1"] == 0.0
  assert metrics["human_among_top"] == 1.0
  assert metrics["human_over_composer"] == 0.0


def test_pair_checkpoint_key_uses_min_pair() -> None:
  weak_composer = {
    "human_over_draft": 1.0,
    "human_over_composer": 0.4,
    "human_among_top": 0.9,
  }
  weak_draft = {
    "human_over_draft": 0.4,
    "human_over_composer": 1.0,
    "human_among_top": 0.9,
  }
  assert pair_checkpoint_selection_key(weak_composer)[0] == 0.4
  assert pair_checkpoint_selection_key(weak_draft)[0] == 0.4
  better = {
    "human_over_draft": 0.8,
    "human_over_composer": 0.8,
    "human_among_top": 0.5,
  }
  assert pair_checkpoint_selection_key(better) > pair_checkpoint_selection_key(weak_composer)
  tied_min_higher_among = {
    "human_over_draft": 0.4,
    "human_over_composer": 0.4,
    "human_among_top": 1.0,
  }
  assert pair_checkpoint_selection_key(tied_min_higher_among) > pair_checkpoint_selection_key(
    weak_composer
  )


def test_human_top_loss_does_not_order_composer_and_draft() -> None:
  human = torch.tensor([0])
  assert float(human_top_loss(torch.tensor([[3.0, 1.0, 0.0]]), human).item()) == float(
    human_top_loss(torch.tensor([[3.0, 0.0, 1.0]]), human).item()
  )


def test_independent_overfit_puts_human_among_top() -> None:
  triple = SetwiseTriple(
    item_id="synthetic-pair",
    source_text="下書き一。下書き二。",
    draft="下書き一。下書き二。",
    human="人間一。人間二。人間三。",
    composer="生成一。生成二。",
    meta={},
  )
  texts = [triple.source_text, triple.human, triple.composer, triple.draft]
  _, sent_to_embedding = _prepared(texts, embed_dim=16, seed=7)
  examples = examples_from_triples([triple], include_composer=True)
  cfg = SetwiseTrainConfig(
    d_model=32,
    local_num_layers=1,
    joint_num_layers=1,
    dropout=0.0,
    epochs=40,
    batch_size=1,
    lr=5e-3,
    max_sents=8,
    seed=0,
    loss_mode=LOSS_MODE_HUMAN_TOP,
  )
  model, _train, valid = train_multigranular_model(
    examples,
    [],
    examples,
    cfg,
    device=torch.device("cpu"),
    score_mode=SCORE_MODE_INDEPENDENT,
    phase1_epochs=0,
    phase2_epochs=40,
    replay_hunk=False,
    precomputed_embeddings=sent_to_embedding,
  )
  assert valid["valid_human_among_top"] == 1.0
  assert STAGE_PAIR_HUMANTOP == "pair_humantop"


def test_freeze_8d_protocol_exists() -> None:
  protocol = json.loads((ROOT / "data/blind_eval/8d_protocol.json").read_text(encoding="utf-8"))
  assert protocol["n_items"] == 40
  assert len(protocol["candidates"]) == 2
  assert protocol["judgment"]["scorer_selection"] is False
  items_path = ROOT / protocol["items_file"]
  if items_path.is_file():
    n = sum(1 for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert n == 40
