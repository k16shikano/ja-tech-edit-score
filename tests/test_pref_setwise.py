from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from pref_static_utils import load_jsonl
from score_hard_eval_setwise import pairwise_agreement, pair_direction, validate_item
from setwise_model import (
  ARCHITECTURE,
  ARTIFACT_KIND,
  ARTIFACT_KIND_HUMAN_TOP,
  LOSS_MODE_FULL_ORDER,
  LOSS_MODE_HUMAN_TOP,
  STRICT_COMPARE_EPS,
  SetwiseRankingModel,
  build_model_config,
  build_setwise_batch_tensors,
  checkpoint_selection_key,
  compute_setwise_metrics,
  format_best_model,
  human_top_loss,
  listmle_loss,
  load_setwise_model_from_artifact,
  normalize_logits_for_duplicate_texts,
  rank_indices_from_roles,
  strict_pair_outcome,
  strict_top1_hit,
)
from setwise_triple_utils import (
  CANONICAL_ROLES,
  TripleReconstructionError,
  reconstruct_triples_from_pref_rows,
)
from train_pref_sentseq import prepare_sentence_data


def _synthetic_batch(*, bsz: int = 2, num_cands: int = 3, max_s: int = 4, embed_dim: int = 8):
  device = torch.device("cpu")
  source_sent_emb = torch.randn(bsz, max_s, embed_dim)
  source_para = torch.zeros(bsz, max_s, dtype=torch.bool)
  source_len = torch.full((bsz,), max_s, dtype=torch.long)
  candidate_sent_emb = torch.randn(bsz, num_cands, max_s, embed_dim)
  candidate_para = torch.zeros(bsz, num_cands, max_s, dtype=torch.bool)
  candidate_len = torch.full((bsz, num_cands), max_s, dtype=torch.long)
  return (
    device,
    source_sent_emb,
    source_para,
    source_len,
    candidate_sent_emb,
    candidate_para,
    candidate_len,
  )


def test_permutation_equivariance_all_permutations() -> None:
  torch.manual_seed(0)
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    max_sents=8,
  )
  model.eval()
  (
    _device,
    source_sent_emb,
    source_para,
    source_len,
    candidate_sent_emb,
    candidate_para,
    candidate_len,
  ) = _synthetic_batch(bsz=1, num_cands=3, max_s=3, embed_dim=8)

  with torch.no_grad():
    base_logits = model(
      source_sent_emb,
      source_para,
      source_len,
      candidate_sent_emb,
      candidate_para,
      candidate_len,
    ).squeeze(0)

  for perm in itertools.permutations(range(3)):
    permuted_emb = candidate_sent_emb[:, list(perm)].clone()
    permuted_para = candidate_para[:, list(perm)].clone()
    permuted_len = candidate_len[:, list(perm)].clone()
    with torch.no_grad():
      perm_logits = model(
        source_sent_emb,
        source_para,
        source_len,
        permuted_emb,
        permuted_para,
        permuted_len,
      ).squeeze(0)
    expected = base_logits[torch.tensor(list(perm), dtype=torch.long)]
    assert torch.allclose(perm_logits, expected, atol=1e-5), (perm, perm_logits, expected)


def test_listmle_prefers_correct_order() -> None:
  logits_good = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
  logits_bad = torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32)
  rank = torch.tensor([[0, 1, 2]], dtype=torch.long)
  assert float(listmle_loss(logits_good, rank).item()) < float(
    listmle_loss(logits_bad, rank).item()
  )


def test_human_top_loss_prefers_human_on_top() -> None:
  logits_good = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
  logits_bad = torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32)
  human = torch.tensor([0], dtype=torch.long)
  assert float(human_top_loss(logits_good, human).item()) < float(
    human_top_loss(logits_bad, human).item()
  )


def test_human_top_loss_invariant_to_composer_draft_swap() -> None:
  logits_a = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
  logits_b = torch.tensor([[3.0, 0.0, 1.0]], dtype=torch.float32)
  human = torch.tensor([0], dtype=torch.long)
  assert float(human_top_loss(logits_a, human).item()) == float(
    human_top_loss(logits_b, human).item()
  )


def test_human_top_loss_backward() -> None:
  logits = torch.tensor([[1.0, 0.5, 0.2]], dtype=torch.float32, requires_grad=True)
  human = torch.tensor([0], dtype=torch.long)
  loss = human_top_loss(logits, human)
  loss.backward()
  assert logits.grad is not None
  assert torch.isfinite(logits.grad).all()


def test_human_top_checkpoint_key_ignores_composer_draft_order() -> None:
  base = {
    "human_top1": 0.8,
    "human_over_composer": 0.9,
    "human_over_draft": 0.95,
    "human_top_loss": 0.4,
    "listwise_loss": 1.2,
    "exact_order": 0.1,
    "composer_over_draft": 0.2,
  }
  alt = dict(base)
  alt["exact_order"] = 0.9
  alt["composer_over_draft"] = 0.9
  alt["listwise_loss"] = 0.1
  key_base = checkpoint_selection_key(base, loss_mode=LOSS_MODE_HUMAN_TOP)
  key_alt = checkpoint_selection_key(alt, loss_mode=LOSS_MODE_HUMAN_TOP)
  assert key_base == key_alt


def test_full_order_checkpoint_key_uses_exact_order() -> None:
  base = {
    "human_top1": 0.8,
    "human_over_composer": 0.9,
    "human_over_draft": 0.95,
    "human_top_loss": 0.4,
    "listwise_loss": 1.2,
    "exact_order": 0.1,
    "composer_over_draft": 0.2,
  }
  better = dict(base)
  better["exact_order"] = 0.9
  better["listwise_loss"] = 0.1
  key_base = checkpoint_selection_key(base, loss_mode=LOSS_MODE_FULL_ORDER)
  key_better = checkpoint_selection_key(better, loss_mode=LOSS_MODE_FULL_ORDER)
  assert key_better > key_base


def test_load_artifact_without_loss_mode_defaults_full_order(tmp_path: Path) -> None:
  torch.manual_seed(2)
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.1,
    max_sents=8,
  )
  cfg = build_model_config(
    {
      "d_model": 16,
      "nhead": 2,
      "local_num_layers": 1,
      "joint_num_layers": 1,
      "dim_feedforward": 32,
      "dropout": 0.1,
      "max_sents": 8,
      "sentence_model_name": "mock-ruri",
      "truncate_dim": None,
      "text_prefix": "文章: ",
      "max_seq_length": 64,
      "normalize_embeddings": True,
      "sent_split_version": "v1",
      "para_boundary_mode": "para_start_embedding",
    },
    embed_dim=8,
  )
  cfg.pop("loss_mode", None)
  artifact = {
    "kind": ARTIFACT_KIND,
    "model_state_dict": model.state_dict(),
    "config": cfg,
  }
  loaded = load_setwise_model_from_artifact(artifact, device=torch.device("cpu"))
  assert loaded.embed_dim == 8


def test_load_human_top_artifact(tmp_path: Path) -> None:
  torch.manual_seed(3)
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.1,
    max_sents=8,
  )
  cfg = build_model_config(
    {
      "d_model": 16,
      "nhead": 2,
      "local_num_layers": 1,
      "joint_num_layers": 1,
      "dim_feedforward": 32,
      "dropout": 0.1,
      "max_sents": 8,
      "sentence_model_name": "mock-ruri",
      "truncate_dim": None,
      "text_prefix": "文章: ",
      "max_seq_length": 64,
      "normalize_embeddings": True,
      "sent_split_version": "v1",
      "para_boundary_mode": "para_start_embedding",
    },
    embed_dim=8,
    loss_mode=LOSS_MODE_HUMAN_TOP,
  )
  artifact = {
    "kind": ARTIFACT_KIND_HUMAN_TOP,
    "model_state_dict": model.state_dict(),
    "config": cfg,
  }
  loaded = load_setwise_model_from_artifact(artifact, device=torch.device("cpu"))
  assert loaded.embed_dim == 8
  assert cfg["loss_mode"] == LOSS_MODE_HUMAN_TOP


def test_reconstruct_triples_from_real_pref_train() -> None:
  train_rows = load_jsonl(str(ROOT / "data/section_middle/pref_train.jsonl"))
  valid_rows = load_jsonl(str(ROOT / "data/section_middle/pref_valid.jsonl"))
  train_triples = reconstruct_triples_from_pref_rows(train_rows)
  valid_triples = reconstruct_triples_from_pref_rows(valid_rows)
  assert len(train_triples) == 328
  assert len(valid_triples) == 50
  one = next(t for t in train_triples if t.item_id.endswith("57b4fed572e9"))
  assert one.draft == one.source_text
  assert one.human != one.draft
  assert one.composer != one.draft
  assert one.human != one.composer


def test_reconstruct_rejects_missing_pair() -> None:
  rows = [
    {
      "schema": "section_triple_v1",
      "pair_kind": "gold_vs_draft",
      "label": 1,
      "preference": "a",
      "source_text": "下書き。",
      "candidate_a": "人間。",
      "candidate_b": "下書き。",
      "meta": {"item_id": "x"},
    }
  ]
  try:
    reconstruct_triples_from_pref_rows(rows)
    raise AssertionError("expected TripleReconstructionError")
  except TripleReconstructionError as exc:
    assert "missing pair_kind" in str(exc)


def test_reconstruct_rejects_inconsistent_source() -> None:
  base = {
    "schema": "section_triple_v1",
    "label": 1,
    "preference": "a",
    "meta": {"item_id": "x"},
  }
  rows = [
    {**base, "pair_kind": "gold_vs_draft", "source_text": "甲", "candidate_a": "人間", "candidate_b": "甲"},
    {**base, "pair_kind": "gold_vs_gen", "source_text": "乙", "candidate_a": "人間", "candidate_b": "生成"},
    {
      **base,
      "pair_kind": "gen_vs_draft",
      "source_text": "甲",
      "candidate_a": "生成",
      "candidate_b": "甲",
    },
  ]
  try:
    reconstruct_triples_from_pref_rows(rows)
    raise AssertionError("expected TripleReconstructionError")
  except TripleReconstructionError as exc:
    assert "inconsistent source_text" in str(exc)


def test_reconstruct_rejects_contradictory_role() -> None:
  base = {
    "schema": "section_triple_v1",
    "source_text": "下書き。",
    "label": 1,
    "preference": "a",
    "meta": {"item_id": "x"},
  }
  rows = [
    {**base, "pair_kind": "gold_vs_draft", "candidate_a": "人間A", "candidate_b": "下書き。"},
    {**base, "pair_kind": "gold_vs_gen", "candidate_a": "人間B", "candidate_b": "生成。"},
    {**base, "pair_kind": "gen_vs_draft", "candidate_a": "生成。", "candidate_b": "下書き。"},
  ]
  try:
    reconstruct_triples_from_pref_rows(rows)
    raise AssertionError("expected TripleReconstructionError")
  except TripleReconstructionError as exc:
    assert "contradictory human role" in str(exc)


def test_reconstruct_rejects_duplicate_role_text() -> None:
  base = {
    "schema": "section_triple_v1",
    "source_text": "下書き。",
    "label": 1,
    "preference": "a",
    "meta": {"item_id": "x"},
  }
  rows = [
    {**base, "pair_kind": "gold_vs_draft", "candidate_a": "人間。", "candidate_b": "下書き。"},
    {**base, "pair_kind": "gold_vs_gen", "candidate_a": "人間。", "candidate_b": "人間。"},
    {**base, "pair_kind": "gen_vs_draft", "candidate_a": "人間。", "candidate_b": "下書き。"},
  ]
  try:
    reconstruct_triples_from_pref_rows(rows)
    raise AssertionError("expected TripleReconstructionError")
  except TripleReconstructionError as exc:
    assert "must be distinct" in str(exc)


def test_save_load_artifact_contract(tmp_path: Path) -> None:
  torch.manual_seed(1)
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.1,
    max_sents=8,
  )
  cfg = build_model_config(
    {
      "d_model": 16,
      "nhead": 2,
      "local_num_layers": 1,
      "joint_num_layers": 1,
      "dim_feedforward": 32,
      "dropout": 0.1,
      "max_sents": 8,
      "sentence_model_name": "mock-ruri",
      "truncate_dim": None,
      "text_prefix": "文章: ",
      "max_seq_length": 64,
      "normalize_embeddings": True,
      "sent_split_version": "v1",
      "para_boundary_mode": "para_start_embedding",
    },
    embed_dim=8,
  )
  artifact = {
    "kind": ARTIFACT_KIND,
    "model_state_dict": model.state_dict(),
    "config": cfg,
  }
  out = tmp_path / "model.pt"
  torch.save(artifact, out)
  meta = tmp_path / "meta.json"
  meta.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
  loaded = load_setwise_model_from_artifact(
    torch.load(out, map_location="cpu", weights_only=False),
    device=torch.device("cpu"),
  )
  assert loaded.embed_dim == 8
  meta_cfg = json.loads(meta.read_text(encoding="utf-8"))
  assert meta_cfg["kind"] == ARTIFACT_KIND
  assert meta_cfg["architecture"] == ARCHITECTURE


def test_build_setwise_batch_tensors_shape() -> None:
  texts = ["下書き。", "人間。", "生成。"]
  sent_to_embedding = {t: torch.randn(8).numpy() for t in texts}
  prepared = prepare_sentence_data(
    texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=8,
    device=torch.device("cpu"),
  )
  batch = build_setwise_batch_tensors(
    source_texts=["下書き。"],
    candidate_texts=[["人間。", "生成。", "下書き。"]],
    prepared=prepared,
    embed_dim=8,
    device=torch.device("cpu"),
  )
  assert batch.candidate_sent_emb.shape[1] == 3
  assert batch.candidate_sent_emb.shape[2] == batch.source_sent_emb.shape[1]


def test_build_setwise_batch_tensors_rejects_length_mismatch() -> None:
  texts = ["下書き。", "人間。", "生成。"]
  sent_to_embedding = {t: torch.randn(8).numpy() for t in texts}
  prepared = prepare_sentence_data(
    texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=8,
    device=torch.device("cpu"),
  )
  try:
    build_setwise_batch_tensors(
      source_texts=["下書き。", "下書き二。"],
      candidate_texts=[["人間。", "生成。", "下書き。"]],
      prepared=prepared,
      embed_dim=8,
      device=torch.device("cpu"),
    )
    raise AssertionError("expected ValueError")
  except ValueError as exc:
    assert "must match source_texts" in str(exc)


def test_build_setwise_batch_handles_longer_candidates() -> None:
  from sentseq_utils import split_document_sentences

  source = "下書き一文。"
  cand_short = "候補甲一。候補甲二。"
  cand_long = "候補乙一。候補乙二。候補乙三。候補乙四。"
  cand_one = "候補丙一。"
  all_sents = []
  for text in (source, cand_short, cand_long, cand_one):
    all_sents.extend(u.text for u in split_document_sentences(text))
  sent_to_embedding = {s: torch.randn(8).numpy() for s in all_sents}
  prepared = prepare_sentence_data(
    [source, cand_short, cand_long, cand_one],
    sent_to_embedding=sent_to_embedding,
    max_sents=8,
    device=torch.device("cpu"),
  )
  batch = build_setwise_batch_tensors(
    source_texts=[source],
    candidate_texts=[[cand_short, cand_long, cand_one]],
    prepared=prepared,
    embed_dim=8,
    device=torch.device("cpu"),
  )
  assert int(batch.source_seq_len[0].item()) == 1
  assert int(batch.candidate_seq_len[0, 0].item()) == 2
  assert int(batch.candidate_seq_len[0, 1].item()) == 4
  assert int(batch.candidate_seq_len[0, 2].item()) == 1
  assert batch.candidate_sent_emb.shape[2] >= 4
  assert batch.source_sent_emb.shape[1] == batch.candidate_sent_emb.shape[2]

  short_units = split_document_sentences(cand_short)
  long_units = split_document_sentences(cand_long)
  for idx, unit in enumerate(short_units):
    src_idx = prepared.sent_to_idx[unit.text]
    assert torch.allclose(
      batch.candidate_sent_emb[0, 0, idx],
      prepared.sent_emb[src_idx],
    )
  for idx, unit in enumerate(long_units):
    src_idx = prepared.sent_to_idx[unit.text]
    assert torch.allclose(
      batch.candidate_sent_emb[0, 1, idx],
      prepared.sent_emb[src_idx],
    )

  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    max_sents=8,
  )
  model.eval()
  logits = model(
    batch.source_sent_emb,
    batch.source_para_start_mask,
    batch.source_seq_len,
    batch.candidate_sent_emb,
    batch.candidate_para_start_mask,
    batch.candidate_seq_len,
  )
  assert logits.shape == (1, 3)


def test_load_rejects_old_architecture(tmp_path: Path) -> None:
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    max_sents=8,
  )
  cfg = build_model_config(
    {
      "d_model": 16,
      "nhead": 2,
      "local_num_layers": 1,
      "joint_num_layers": 1,
      "dim_feedforward": 32,
      "dropout": 0.1,
      "max_sents": 8,
      "sentence_model_name": "mock-ruri",
      "truncate_dim": None,
      "text_prefix": "文章: ",
      "max_seq_length": 64,
      "normalize_embeddings": True,
      "sent_split_version": "v1",
      "para_boundary_mode": "para_start_embedding",
    },
    embed_dim=8,
  )
  cfg["architecture"] = "joint_transformer_on_sentence_tokens"
  artifact = {"kind": ARTIFACT_KIND, "model_state_dict": model.state_dict(), "config": cfg}
  try:
    load_setwise_model_from_artifact(artifact, device=torch.device("cpu"))
    raise AssertionError("expected ValueError")
  except ValueError as exc:
    assert "architecture" in str(exc)


def test_local_then_joint_architecture() -> None:
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    max_sents=8,
  )
  assert hasattr(model, "local_transformer")
  assert hasattr(model, "joint_transformer")
  forbidden = ("stream_encoder", "candidate_self_attn", "candidate_cross_attn")
  assert all(not hasattr(model, name) for name in forbidden)


def test_untrained_model_logits_are_not_all_equal() -> None:
  torch.manual_seed(1)
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    max_sents=8,
  )
  model.eval()
  (
    _device,
    source_sent_emb,
    source_para,
    source_len,
    candidate_sent_emb,
    candidate_para,
    candidate_len,
  ) = _synthetic_batch(bsz=1, num_cands=3, max_s=3, embed_dim=8)
  candidate_sent_emb[0, 0] = torch.randn(3, 8)
  candidate_sent_emb[0, 1] = torch.randn(3, 8) + 2.0
  candidate_sent_emb[0, 2] = torch.randn(3, 8) - 2.0
  with torch.no_grad():
    logits = model(
      source_sent_emb,
      source_para,
      source_len,
      candidate_sent_emb,
      candidate_para,
      candidate_len,
    ).squeeze(0)
  assert not torch.allclose(logits[0], logits[1], atol=1e-5)
  assert not torch.allclose(logits[0], logits[2], atol=1e-5)
  assert not torch.allclose(logits[1], logits[2], atol=1e-5)


def test_changing_one_candidate_does_not_shift_all_logits_uniformly() -> None:
  torch.manual_seed(0)
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    max_sents=8,
  )
  model.eval()
  (
    _device,
    source_sent_emb,
    source_para,
    source_len,
    candidate_sent_emb,
    candidate_para,
    candidate_len,
  ) = _synthetic_batch(bsz=1, num_cands=3, max_s=3, embed_dim=8)
  with torch.no_grad():
    base_logits = model(
      source_sent_emb,
      source_para,
      source_len,
      candidate_sent_emb,
      candidate_para,
      candidate_len,
    ).squeeze(0)
  perturbed = candidate_sent_emb.clone()
  perturbed[0, 1, 1] += 3.0
  with torch.no_grad():
    changed_logits = model(
      source_sent_emb,
      source_para,
      source_len,
      perturbed,
      candidate_para,
      candidate_len,
    ).squeeze(0)
  delta = changed_logits - base_logits
  assert not torch.allclose(base_logits, changed_logits, atol=1e-5)
  assert not torch.allclose(delta[0], delta[1], atol=1e-5) or not torch.allclose(
    delta[1], delta[2], atol=1e-5
  )


def test_uniform_logits_metrics_are_zero() -> None:
  logits = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
  rank = torch.tensor([[0, 1, 2]], dtype=torch.long)
  metrics = compute_setwise_metrics(
    logits,
    rank,
    human_index=torch.tensor([0]),
    composer_index=torch.tensor([1]),
    draft_index=torch.tensor([2]),
  )
  assert metrics["human_top1"] == 0.0
  assert metrics["exact_order"] == 0.0
  assert metrics["human_over_composer"] == 0.0
  assert metrics["human_over_draft"] == 0.0
  assert metrics["composer_over_draft"] == 0.0
  assert metrics["human_top_loss"] > 0.0
  assert checkpoint_selection_key(metrics, loss_mode=LOSS_MODE_FULL_ORDER)[0] == 0.0
  assert checkpoint_selection_key(metrics, loss_mode=LOSS_MODE_HUMAN_TOP)[0] == 0.0


def test_single_triple_overfit() -> None:
  from sentseq_utils import split_document_sentences
  from setwise_triple_utils import SetwiseTriple
  from train_pref_setwise import SetwiseTrainConfig, train_setwise_model

  triple = SetwiseTriple(
    item_id="synthetic-overfit",
    source_text="下書き一。下書き二。",
    draft="下書き一。下書き二。",
    human="人間一。人間二。人間三。",
    composer="生成一。生成二。",
    meta={},
  )
  embed_dim = 16
  sent_to_embedding: dict[str, object] = {}
  torch.manual_seed(7)
  for text in (triple.source_text, triple.human, triple.composer, triple.draft):
    for unit in split_document_sentences(text):
      if unit.text not in sent_to_embedding:
        sent_to_embedding[unit.text] = torch.randn(embed_dim).numpy()

  cfg = SetwiseTrainConfig(
    d_model=32,
    local_num_layers=1,
    joint_num_layers=1,
    dropout=0.0,
    epochs=60,
    batch_size=1,
    lr=5e-3,
    max_sents=8,
    seed=0,
  )
  _model, train_metrics, valid_metrics, _prepared = train_setwise_model(
    [triple],
    [triple],
    cfg,
    device=torch.device("cpu"),
    precomputed_embeddings=sent_to_embedding,
  )
  assert train_metrics["train_objective_loss"] < 1.2
  assert valid_metrics["valid_listwise_loss"] < 1.2
  assert valid_metrics["valid_human_top1"] == 1.0
  assert valid_metrics["valid_exact_order"] == 1.0


def test_hard_eval_tie_fixture_not_counted_as_hit() -> None:
  scores = {"human": 1.0, "mid": 1.0, "copy": 1.0}
  assert not strict_top1_hit(scores, "human", ["human", "mid", "copy"])
  assert format_best_model(scores, ["human", "mid", "copy"]).startswith("tie:")


def test_normalize_duplicate_text_logits() -> None:
  logits = torch.tensor([1.0, 2.0, 1.5])
  texts = ["same", "other", "same"]
  out = normalize_logits_for_duplicate_texts(texts, logits)
  assert out[0].item() == out[2].item()
  assert abs(out[0].item() - 1.25) < 1e-6
  assert abs(out[1].item() - 2.0) < 1e-6


def test_duplicate_text_pair_is_tie_after_normalization() -> None:
  logits = torch.tensor([-2.5848002433776855, -3.0, -2.5847995281219482])
  texts = ["human text", "mid text", "human text"]
  out = normalize_logits_for_duplicate_texts(texts, logits)
  scores = {"human": float(out[0].item()), "mid": float(out[1].item()), "copy": float(out[2].item())}
  assert scores["human"] == scores["copy"]
  assert strict_pair_outcome(scores["human"], scores["copy"]) == "tie"
  assert not strict_top1_hit(scores, "human", ["human", "mid", "copy"])


def test_micro_diff_logits_count_as_tie() -> None:
  scores = {
    "human": -2.5848002433776855,
    "copy": -2.5847995281219482,
    "mid": -3.0,
  }
  assert abs(scores["human"] - scores["copy"]) < STRICT_COMPARE_EPS
  assert strict_pair_outcome(scores["human"], scores["copy"]) == "tie"
  assert not strict_top1_hit(scores, "human", list(scores.keys()))
  assert pair_direction(scores, "human", "mid")
  assert not pair_direction(scores, "human", "copy")


def test_joint_transformer_reads_sentence_tokens_before_doc_pooling() -> None:
  torch.manual_seed(0)
  model = SetwiseRankingModel(
    embed_dim=8,
    d_model=16,
    nhead=2,
    local_num_layers=1,
    joint_num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    max_sents=8,
  )
  model.eval()
  (
    _device,
    source_sent_emb,
    source_para,
    source_len,
    candidate_sent_emb,
    candidate_para,
    candidate_len,
  ) = _synthetic_batch(bsz=1, num_cands=3, max_s=3, embed_dim=8)
  with torch.no_grad():
    base_logits = model(
      source_sent_emb,
      source_para,
      source_len,
      candidate_sent_emb,
      candidate_para,
      candidate_len,
    )
  perturbed_source = source_sent_emb.clone()
  perturbed_source[0, 1] += 1.0
  with torch.no_grad():
    changed_by_source = model(
      perturbed_source,
      source_para,
      source_len,
      candidate_sent_emb,
      candidate_para,
      candidate_len,
    )
  assert not torch.allclose(base_logits, changed_by_source, atol=1e-5)

  perturbed_candidate = candidate_sent_emb.clone()
  perturbed_candidate[0, 1, 1] += 1.0
  with torch.no_grad():
    changed_by_other_candidate = model(
      source_sent_emb,
      source_para,
      source_len,
      perturbed_candidate,
      candidate_para,
      candidate_len,
    )
  assert not torch.allclose(base_logits[0, 0], changed_by_other_candidate[0, 0], atol=1e-5)


def test_hard_eval_pairwise_fixture() -> None:
  item = {
    "id": "he-fixture",
    "status": "labeled",
    "base_text": "draft",
    "candidates": [
      {"id": "human", "text": "human"},
      {"id": "mid", "text": "mid"},
      {"id": "copy", "text": "copy"},
    ],
    "human": {"best_id": "human", "rank": ["human", "mid", "copy"]},
  }
  assert validate_item(item) is None
  scores = {"human": 3.0, "mid": 2.0, "copy": 1.0}
  agree, total = pairwise_agreement(item["human"]["rank"], scores)
  assert agree == 3 and total == 3
  assert pair_direction(scores, "human", "mid")
  assert pair_direction(scores, "human", "copy")
  assert pair_direction(scores, "mid", "copy")


def test_rank_indices_follow_permutation() -> None:
  roles = ["composer", "human", "draft"]
  rank = rank_indices_from_roles(roles, best_first_roles=CANONICAL_ROLES)
  assert rank == [1, 0, 2]
