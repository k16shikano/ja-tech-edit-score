#!/usr/bin/env python3
"""凍結 ruri 文埋め込み + local stream encoder + joint token encoder の setwise 順位付けモデル。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sentseq_utils import split_document_sentences, truncate_sentence_units
from train_pref_sentseq import PreparedSentSeqData, build_document_batch_tensors

STREAM_SOURCE = 0
STREAM_CANDIDATE = 1
ARTIFACT_KIND = "pref-setwise-section-triples"
ARTIFACT_KIND_HUMAN_TOP = "pref-setwise-section-human-top"
ARTIFACT_KIND_PAIR_INDEPENDENT = "pref-pair-independent"
ARTIFACT_KIND_JOINT_HUNK_REPLAY = "pref-joint-humantop-hunk-replay"
SUPPORTED_ARTIFACT_KINDS = frozenset(
  {
    ARTIFACT_KIND,
    ARTIFACT_KIND_HUMAN_TOP,
    ARTIFACT_KIND_PAIR_INDEPENDENT,
    ARTIFACT_KIND_JOINT_HUNK_REPLAY,
  }
)
ARCHITECTURE = "local_stream_then_joint_tokens_v2"
STRICT_COMPARE_EPS = 1e-6
LOSS_MODE_FULL_ORDER = "full_order"
LOSS_MODE_HUMAN_TOP = "human_top"
VALID_LOSS_MODES = frozenset({LOSS_MODE_FULL_ORDER, LOSS_MODE_HUMAN_TOP})
SCORE_MODE_JOINT = "joint"
SCORE_MODE_INDEPENDENT = "independent"
VALID_SCORE_MODES = frozenset({SCORE_MODE_JOINT, SCORE_MODE_INDEPENDENT})


def normalize_loss_mode(loss_mode: str | None) -> str:
  if loss_mode is None or loss_mode == LOSS_MODE_FULL_ORDER:
    return LOSS_MODE_FULL_ORDER
  if loss_mode == LOSS_MODE_HUMAN_TOP:
    return LOSS_MODE_HUMAN_TOP
  raise ValueError(
    f"unsupported loss_mode {loss_mode!r}; expected one of {sorted(VALID_LOSS_MODES)}"
  )


def validate_loss_mode(loss_mode: str) -> str:
  return normalize_loss_mode(loss_mode)


def normalize_score_mode(score_mode: str | None) -> str:
  if score_mode is None or score_mode == SCORE_MODE_JOINT:
    return SCORE_MODE_JOINT
  if score_mode == SCORE_MODE_INDEPENDENT:
    return SCORE_MODE_INDEPENDENT
  raise ValueError(
    f"unsupported score_mode {score_mode!r}; expected one of {sorted(VALID_SCORE_MODES)}"
  )


def artifact_kind_for_loss_mode(loss_mode: str) -> str:
  if normalize_loss_mode(loss_mode) == LOSS_MODE_HUMAN_TOP:
    return ARTIFACT_KIND_HUMAN_TOP
  return ARTIFACT_KIND


class SetwiseRankingModel(nn.Module):
  """source + K 候補の [DOC]+文 token を block ごとに local 処理し、全文列を joint で読む。"""

  def __init__(
    self,
    *,
    embed_dim: int,
    d_model: int = 256,
    nhead: int = 4,
    local_num_layers: int = 1,
    joint_num_layers: int = 1,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    max_sents: int = 128,
  ) -> None:
    super().__init__()
    self.embed_dim = embed_dim
    self.d_model = d_model
    self.local_num_layers = local_num_layers
    self.joint_num_layers = joint_num_layers
    self.max_sents = max_sents
    self.source_doc_token = nn.Parameter(torch.zeros(1, 1, d_model))
    self.candidate_doc_token = nn.Parameter(torch.zeros(1, 1, d_model))
    self.sent_proj = nn.Linear(embed_dim, d_model)
    self.pos_emb = nn.Embedding(max_sents + 1, d_model)
    self.para_start_emb = nn.Embedding(1, d_model)
    self.stream_type_emb = nn.Embedding(2, d_model)
    local_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      batch_first=True,
      activation="gelu",
    )
    joint_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      batch_first=True,
      activation="gelu",
    )
    self.local_transformer = nn.TransformerEncoder(
      local_layer,
      num_layers=local_num_layers,
    )
    self.joint_transformer = nn.TransformerEncoder(
      joint_layer,
      num_layers=joint_num_layers,
    )
    self.score_head = nn.Linear(d_model, 1)

  def _embed_stream_block(
    self,
    sent_emb: torch.Tensor,
    *,
    para_start_mask: torch.Tensor,
    doc_token: torch.Tensor,
    stream_type: int,
  ) -> torch.Tensor:
    """[DOC] + 文 token 列。stream 内 position は 1..max_s でリセット。"""
    bsz, max_s, _ = sent_emb.shape
    doc = doc_token.expand(bsz, -1, -1)
    sent = self.sent_proj(sent_emb)
    pos_ids = torch.arange(1, max_s + 1, device=sent_emb.device).unsqueeze(0).expand(
      bsz, -1
    )
    sent = sent + self.pos_emb(pos_ids)
    para_bias = self.para_start_emb(
      torch.zeros(1, dtype=torch.long, device=sent_emb.device)
    )
    sent = sent + para_start_mask.unsqueeze(-1).float() * para_bias
    seq = torch.cat([doc, sent], dim=1)
    type_ids = torch.full(
      (bsz, seq.shape[1]),
      stream_type,
      dtype=torch.long,
      device=sent_emb.device,
    )
    return seq + self.stream_type_emb(type_ids)

  def _block_padding_mask(
    self,
    piece_lens: torch.Tensor,
    *,
    block_width: int,
  ) -> torch.Tensor:
    positions = torch.arange(block_width, device=piece_lens.device)
    return positions.unsqueeze(0) >= piece_lens.unsqueeze(-1)

  def _joint_padding_mask(
    self,
    piece_lens: torch.Tensor,
    *,
    block_width: int,
  ) -> torch.Tensor:
    bsz, num_blocks = piece_lens.shape
    positions = torch.arange(block_width, device=piece_lens.device)
    pad_blocks = positions.view(1, 1, block_width) >= piece_lens.unsqueeze(-1)
    return pad_blocks.reshape(bsz, num_blocks * block_width)

  def forward(
    self,
    source_sent_emb: torch.Tensor,
    source_para_start_mask: torch.Tensor,
    source_seq_len: torch.Tensor,
    candidate_sent_emb: torch.Tensor,
    candidate_para_start_mask: torch.Tensor,
    candidate_seq_len: torch.Tensor,
  ) -> torch.Tensor:
    bsz, num_cands, max_s, _ = candidate_sent_emb.shape
    source_block = self._embed_stream_block(
      source_sent_emb,
      para_start_mask=source_para_start_mask,
      doc_token=self.source_doc_token,
      stream_type=STREAM_SOURCE,
    )
    cand_blocks: list[torch.Tensor] = []
    for k in range(num_cands):
      cand_blocks.append(
        self._embed_stream_block(
          candidate_sent_emb[:, k],
          para_start_mask=candidate_para_start_mask[:, k],
          doc_token=self.candidate_doc_token,
          stream_type=STREAM_CANDIDATE,
        )
      )

    block_width = max_s + 1
    num_blocks = 1 + num_cands
    blocks = torch.stack([source_block] + cand_blocks, dim=1)
    piece_lens = torch.zeros((bsz, num_blocks), dtype=torch.long, device=source_sent_emb.device)
    piece_lens[:, 0] = source_seq_len + 1
    piece_lens[:, 1:] = candidate_seq_len + 1

    local_in = blocks.reshape(bsz * num_blocks, block_width, self.d_model)
    local_lens = piece_lens.reshape(bsz * num_blocks)
    local_pad = self._block_padding_mask(local_lens, block_width=block_width)
    local_out = self.local_transformer(local_in, src_key_padding_mask=local_pad)
    local_blocks = local_out.view(bsz, num_blocks, block_width, self.d_model)
    joint_seq = local_blocks.reshape(bsz, num_blocks * block_width, self.d_model)

    joint_pad = self._joint_padding_mask(piece_lens, block_width=block_width)
    hidden = self.joint_transformer(joint_seq, src_key_padding_mask=joint_pad)

    doc_indices = torch.zeros((bsz, num_cands), dtype=torch.long, device=source_sent_emb.device)
    for k in range(num_cands):
      doc_indices[:, k] = (k + 1) * block_width
    gather_idx = doc_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
    cand_doc_hidden = hidden.gather(1, gather_idx)
    return self.score_head(cand_doc_hidden).squeeze(-1)


def independent_candidate_logits(
  model: SetwiseRankingModel,
  tensors: "SetwiseBatchTensors",
) -> torch.Tensor:
  """各候補を、下書きとその候補だけの入力で点を出す。[B, K]。"""
  bsz, num_cands, max_s, embed_dim = tensors.candidate_sent_emb.shape
  src_emb = (
    tensors.source_sent_emb.unsqueeze(1)
    .expand(-1, num_cands, -1, -1)
    .reshape(bsz * num_cands, max_s, embed_dim)
  )
  src_para = (
    tensors.source_para_start_mask.unsqueeze(1)
    .expand(-1, num_cands, -1)
    .reshape(bsz * num_cands, max_s)
  )
  src_len = tensors.source_seq_len.unsqueeze(1).expand(-1, num_cands).reshape(bsz * num_cands)
  cand_emb = tensors.candidate_sent_emb.reshape(bsz * num_cands, 1, max_s, embed_dim)
  cand_para = tensors.candidate_para_start_mask.reshape(bsz * num_cands, 1, max_s)
  cand_len = tensors.candidate_seq_len.reshape(bsz * num_cands, 1)
  logits = model(src_emb, src_para, src_len, cand_emb, cand_para, cand_len)
  return logits.view(bsz, num_cands)


def candidate_logits(
  model: SetwiseRankingModel,
  tensors: "SetwiseBatchTensors",
  *,
  score_mode: str,
) -> torch.Tensor:
  mode = normalize_score_mode(score_mode)
  if mode == SCORE_MODE_INDEPENDENT:
    return independent_candidate_logits(model, tensors)
  return model(
    tensors.source_sent_emb,
    tensors.source_para_start_mask,
    tensors.source_seq_len,
    tensors.candidate_sent_emb,
    tensors.candidate_para_start_mask,
    tensors.candidate_seq_len,
  )


def public_relative_scores(
  candidate_logits: torch.Tensor,
  self_logit: torch.Tensor,
) -> torch.Tensor:
  """公開点: 下書きと採点対象の点から、下書きと下書きの点を引く。"""
  if candidate_logits.ndim == 1:
    return candidate_logits - self_logit
  return candidate_logits - self_logit.unsqueeze(-1)


def listmle_loss(logits: torch.Tensor, rank_indices: torch.Tensor) -> torch.Tensor:
  """Plackett-Luce / ListMLE。rank_indices は best-first の候補 index。"""
  if logits.ndim != 2 or rank_indices.ndim != 2:
    raise ValueError("logits and rank_indices must be [B, K]")
  bsz, k = logits.shape
  if rank_indices.shape != (bsz, k):
    raise ValueError("rank_indices shape must match logits")
  loss = torch.zeros(bsz, device=logits.device, dtype=logits.dtype)
  remaining = torch.ones(bsz, k, dtype=torch.bool, device=logits.device)
  for step in range(k - 1):
    idx = rank_indices[:, step]
    masked = logits.masked_fill(~remaining, float("-inf"))
    log_denom = torch.logsumexp(masked, dim=-1)
    chosen = logits.gather(1, idx.unsqueeze(1)).squeeze(1)
    loss = loss + (log_denom - chosen)
    remaining.scatter_(1, idx.unsqueeze(1), False)
  return loss.mean()


def human_top_loss(logits: torch.Tensor, human_indices: torch.Tensor) -> torch.Tensor:
  """Plackett-Luce の第1項: logsumexp(all) - s_human。Composer と draft の相対順位は課さない。"""
  if logits.ndim != 2 or human_indices.ndim != 1:
    raise ValueError("logits must be [B, K] and human_indices must be [B]")
  bsz, _ = logits.shape
  if human_indices.shape[0] != bsz:
    raise ValueError("human_indices length must match batch size")
  human_logit = _logit_at(logits, human_indices)
  log_denom = torch.logsumexp(logits, dim=-1)
  return (log_denom - human_logit).mean()


def compute_training_loss(
  logits: torch.Tensor,
  rank_indices: torch.Tensor,
  human_indices: torch.Tensor,
  *,
  loss_mode: str,
) -> torch.Tensor:
  mode = normalize_loss_mode(loss_mode)
  if mode == LOSS_MODE_HUMAN_TOP:
    return human_top_loss(logits, human_indices)
  return listmle_loss(logits, rank_indices)


def rank_indices_from_roles(
  candidate_roles: list[str],
  *,
  best_first_roles: tuple[str, ...],
) -> list[int]:
  role_to_idx = {role: idx for idx, role in enumerate(candidate_roles)}
  return [role_to_idx[role] for role in best_first_roles]


def _logit_at(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
  return logits.gather(1, index.unsqueeze(1)).squeeze(1)


def strictly_greater(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  return a > b + STRICT_COMPARE_EPS


def strict_pair_outcome(better_score: float, worse_score: float) -> str:
  """better が worse より上か。win / tie / loss。"""
  if better_score > worse_score + STRICT_COMPARE_EPS:
    return "win"
  if worse_score > better_score + STRICT_COMPARE_EPS:
    return "loss"
  return "tie"


def strict_top1_hit(scores: dict[str, float], best_id: str, candidate_ids: list[str]) -> bool:
  if best_id not in scores:
    return False
  best_score = scores[best_id]
  return all(
    best_score > scores[cid] + STRICT_COMPARE_EPS
    for cid in candidate_ids
    if cid != best_id
  )


def format_best_model(scores: dict[str, float], candidate_ids: list[str]) -> str:
  top_ids = [
    cid
    for cid in candidate_ids
    if not any(
      scores[other] > scores[cid] + STRICT_COMPARE_EPS
      for other in candidate_ids
      if other != cid
    )
  ]
  if len(top_ids) == 1:
    return top_ids[0]
  return "tie:" + ",".join(sorted(top_ids, key=lambda cid: (-scores[cid], cid)))


def normalize_logits_for_duplicate_texts(
  candidate_texts: list[str],
  logits: torch.Tensor,
) -> torch.Tensor:
  """完全一致する候補文字列の logits を群平均で同一値にする。"""
  if logits.ndim != 1:
    raise ValueError("normalize_logits_for_duplicate_texts expects 1D logits")
  out = logits.clone()
  groups: dict[str, list[int]] = {}
  for idx, text in enumerate(candidate_texts):
    groups.setdefault(text, []).append(idx)
  for indices in groups.values():
    if len(indices) < 2:
      continue
    group_mean = out[indices].mean()
    out[indices] = group_mean
  return out


def _strictly_beats_all(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
  own = _logit_at(logits, index)
  bsz, k = logits.shape
  mask = torch.ones(bsz, k, dtype=torch.bool, device=logits.device)
  mask.scatter_(1, index.unsqueeze(1), False)
  others_max = logits.masked_fill(~mask, float("-inf")).max(dim=-1).values
  return strictly_greater(own, others_max)


def compute_setwise_metrics(
  logits: torch.Tensor,
  rank_indices: torch.Tensor,
  *,
  human_index: torch.Tensor,
  composer_index: torch.Tensor,
  draft_index: torch.Tensor,
) -> dict[str, float]:
  _, k = logits.shape
  human_top1 = _strictly_beats_all(logits, human_index).float().mean().item()
  human_logit = _logit_at(logits, human_index)
  composer_logit = _logit_at(logits, composer_index)
  draft_logit = _logit_at(logits, draft_index)
  human_over_composer = strictly_greater(human_logit, composer_logit).float().mean().item()
  human_over_draft = strictly_greater(human_logit, draft_logit).float().mean().item()
  composer_over_draft = strictly_greater(composer_logit, draft_logit).float().mean().item()
  others_max = logits.masked_fill(
    torch.zeros_like(logits, dtype=torch.bool).scatter(
      1, human_index.unsqueeze(1), True
    ),
    float("-inf"),
  ).max(dim=-1).values
  human_among_top = (~strictly_greater(others_max, human_logit)).float().mean().item()

  order_checks: list[torch.Tensor] = []
  for step in range(k - 1):
    better = rank_indices[:, step]
    worse = rank_indices[:, step + 1]
    order_checks.append(strictly_greater(_logit_at(logits, better), _logit_at(logits, worse)))
  exact_order = torch.stack(order_checks, dim=1).all(dim=1).float().mean().item()

  return {
    "listwise_loss": float(listmle_loss(logits, rank_indices).item()),
    "human_top_loss": float(human_top_loss(logits, human_index).item()),
    "human_top1": human_top1,
    "human_among_top": human_among_top,
    "exact_order": exact_order,
    "human_over_composer": human_over_composer,
    "human_over_draft": human_over_draft,
    "composer_over_draft": composer_over_draft,
  }


def checkpoint_selection_key(
  metrics: dict[str, float],
  *,
  loss_mode: str = LOSS_MODE_FULL_ORDER,
) -> tuple[float, float, float, float]:
  mode = normalize_loss_mode(loss_mode)
  if mode == LOSS_MODE_HUMAN_TOP:
    return (
      float(metrics["human_top1"]),
      float(metrics["human_over_composer"]),
      float(metrics["human_over_draft"]),
      -float(metrics["human_top_loss"]),
    )
  return (
    float(metrics["human_top1"]),
    float(metrics["human_over_composer"]),
    float(metrics["exact_order"]),
    -float(metrics["listwise_loss"]),
  )


def pair_checkpoint_selection_key(metrics: dict[str, float]) -> tuple[float, float, float]:
  """検証50件: min(人間＞下書き, 人間＞別案)、同値なら最上、次に人間＞別案。"""
  human_over_draft = float(metrics["human_over_draft"])
  human_over_composer = float(metrics["human_over_composer"])
  return (
    min(human_over_draft, human_over_composer),
    float(metrics["human_among_top"]),
    human_over_composer,
  )


def strict_top_indices(logits: torch.Tensor) -> list[list[int]]:
  """各 batch 行で logit が他候補すべてより厳密に大きい index（同点なら空）。"""
  if logits.ndim == 1:
    logits = logits.unsqueeze(0)
  bsz, k = logits.shape
  out: list[list[int]] = []
  for row in range(bsz):
    winners: list[int] = []
    for idx in range(k):
      if _strictly_beats_all(logits[row : row + 1], torch.tensor([idx], device=logits.device))[0]:
        winners.append(idx)
    out.append(winners)
  return out


def ranked_indices_from_logits(logits: torch.Tensor) -> list[int]:
  """表示用の順位。同点は index 昇順で tie-break。"""
  if logits.ndim == 2:
    if logits.shape[0] != 1:
      raise ValueError("ranked_indices_from_logits expects a single row for now")
    logits = logits.squeeze(0)
  order = torch.argsort(logits, descending=True, stable=True)
  return [int(i) for i in order.tolist()]


@dataclass
class SetwiseBatchTensors:
  source_sent_emb: torch.Tensor
  source_para_start_mask: torch.Tensor
  source_seq_len: torch.Tensor
  candidate_sent_emb: torch.Tensor
  candidate_para_start_mask: torch.Tensor
  candidate_seq_len: torch.Tensor


def _pad_document_tensors(
  sent_emb: torch.Tensor,
  para_start_mask: torch.Tensor,
  seq_len: torch.Tensor,
  *,
  max_s: int,
  embed_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  bsz, cur_s, _ = sent_emb.shape
  if cur_s >= max_s:
    return sent_emb, para_start_mask, seq_len
  device = sent_emb.device
  padded_sent = torch.zeros((bsz, max_s, embed_dim), dtype=sent_emb.dtype, device=device)
  padded_para = torch.zeros((bsz, max_s), dtype=torch.bool, device=device)
  if cur_s > 0:
    padded_sent[:, :cur_s] = sent_emb
    padded_para[:, :cur_s] = para_start_mask
  return padded_sent, padded_para, seq_len


def build_setwise_batch_tensors(
  *,
  source_texts: list[str],
  candidate_texts: list[list[str]],
  prepared: PreparedSentSeqData,
  embed_dim: int,
  device: torch.device,
) -> SetwiseBatchTensors:
  if not source_texts:
    raise ValueError("source_texts must not be empty")
  if len(candidate_texts) != len(source_texts):
    raise ValueError(
      "candidate_texts length must match source_texts: "
      f"{len(candidate_texts)} != {len(source_texts)}"
    )
  if not candidate_texts:
    raise ValueError("candidate_texts must not be empty")
  bsz = len(source_texts)
  num_cands = len(candidate_texts[0])
  if num_cands < 1:
    raise ValueError(f"each item must have at least 1 candidate, got {num_cands}")
  if any(len(cands) != num_cands for cands in candidate_texts):
    raise ValueError("all items must have the same candidate count")

  src_sent, src_para, src_len = build_document_batch_tensors(
    source_texts,
    prepared,
    embed_dim=embed_dim,
    device=device,
  )
  flat_candidates = [text for cands in candidate_texts for text in cands]
  flat_sent, flat_para, flat_len = build_document_batch_tensors(
    flat_candidates,
    prepared,
    embed_dim=embed_dim,
    device=device,
  )
  max_s = max(int(src_sent.shape[1]), int(flat_sent.shape[1]))
  src_sent, src_para, src_len = _pad_document_tensors(
    src_sent,
    src_para,
    src_len,
    max_s=max_s,
    embed_dim=embed_dim,
  )
  flat_sent, flat_para, flat_len = _pad_document_tensors(
    flat_sent,
    flat_para,
    flat_len,
    max_s=max_s,
    embed_dim=embed_dim,
  )

  cand_sent = torch.zeros(
    (bsz, num_cands, max_s, embed_dim), dtype=torch.float32, device=device
  )
  cand_para = torch.zeros((bsz, num_cands, max_s), dtype=torch.bool, device=device)
  cand_len = torch.zeros((bsz, num_cands), dtype=torch.long, device=device)
  for i in range(bsz):
    for k in range(num_cands):
      j = i * num_cands + k
      slen = int(flat_len[j].item())
      cand_len[i, k] = slen
      if slen > 0:
        cand_sent[i, k, :slen] = flat_sent[j, :slen]
        cand_para[i, k, :slen] = flat_para[j, :slen]
  return SetwiseBatchTensors(
    source_sent_emb=src_sent,
    source_para_start_mask=src_para,
    source_seq_len=src_len,
    candidate_sent_emb=cand_sent,
    candidate_para_start_mask=cand_para,
    candidate_seq_len=cand_len,
  )


def collect_unique_texts_for_triples(
  triples: list,
  *,
  include_source: bool = True,
) -> list[str]:
  seen: set[str] = set()
  texts: list[str] = []
  for triple in triples:
    items = triple.canonical_candidates()
    if include_source and triple.source_text not in seen:
      seen.add(triple.source_text)
      texts.append(triple.source_text)
    for text in items:
      if text not in seen:
        seen.add(text)
        texts.append(text)
  return texts


def collect_unique_sentences(texts: list[str], *, max_sents: int) -> list[str]:
  unique: list[str] = []
  seen: set[str] = set()
  for text in texts:
    units = truncate_sentence_units(
      split_document_sentences(text),
      max_sents=max_sents,
    )
    for unit in units:
      if unit.text not in seen:
        seen.add(unit.text)
        unique.append(unit.text)
  return unique


def build_model_config(
  cfg: dict,
  *,
  embed_dim: int,
  loss_mode: str = LOSS_MODE_FULL_ORDER,
  score_mode: str = SCORE_MODE_JOINT,
  kind: str | None = None,
) -> dict:
  mode = normalize_loss_mode(loss_mode)
  resolved_score = normalize_score_mode(score_mode)
  resolved_kind = kind or artifact_kind_for_loss_mode(mode)
  return {
    "kind": resolved_kind,
    "architecture": ARCHITECTURE,
    "loss_mode": mode,
    "score_mode": resolved_score,
    "d_model": cfg["d_model"],
    "nhead": cfg["nhead"],
    "local_num_layers": cfg["local_num_layers"],
    "joint_num_layers": cfg["joint_num_layers"],
    "dim_feedforward": cfg["dim_feedforward"],
    "dropout": cfg["dropout"],
    "max_sents": cfg["max_sents"],
    "embed_dim": embed_dim,
    "sentence_model_name": cfg["sentence_model_name"],
    "truncate_dim": cfg.get("truncate_dim"),
    "text_prefix": cfg["text_prefix"],
    "max_seq_length": cfg.get("max_seq_length"),
    "normalize_embeddings": cfg.get("normalize_embeddings", True),
    "sent_split_version": cfg.get("sent_split_version"),
    "para_boundary_mode": cfg.get("para_boundary_mode"),
    "teacher_schema": "section_triple_v1",
    "teacher_order": "human>composer>draft",
  }


def load_setwise_model_from_artifact(
  artifact: dict,
  *,
  device: torch.device,
) -> SetwiseRankingModel:
  config = artifact["config"]
  kind = config.get("kind")
  if kind not in SUPPORTED_ARTIFACT_KINDS:
    raise ValueError(
      f"not a supported setwise artifact (kind={kind!r}); "
      f"expected one of {sorted(SUPPORTED_ARTIFACT_KINDS)}"
    )
  if config.get("architecture") != ARCHITECTURE:
    raise ValueError(
      f"unsupported setwise architecture {config.get('architecture')!r}; "
      f"expected {ARCHITECTURE!r}"
    )
  model = SetwiseRankingModel(
    embed_dim=int(config["embed_dim"]),
    d_model=int(config["d_model"]),
    nhead=int(config["nhead"]),
    local_num_layers=int(config["local_num_layers"]),
    joint_num_layers=int(config["joint_num_layers"]),
    dim_feedforward=int(config["dim_feedforward"]),
    dropout=float(config["dropout"]),
    max_sents=int(config["max_sents"]),
  )
  model.load_state_dict(artifact["model_state_dict"])
  model.to(device)
  model.eval()
  return model
