#!/usr/bin/env python3
"""Qwen 因果 LM + LoRA で D ペア BT/GPM の g(x,y) と v(y|x) を出す。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


PREF_DRAFT_TOK = "<|pref_draft|>"
PREF_CAND_TOK = "<|pref_cand|>"
PREF_SPECIAL_TOKENS = [PREF_DRAFT_TOK, PREF_CAND_TOK]


@dataclass
class CausalRewardConfig:
  base_model: str = "Qwen/Qwen3-8B"
  max_length: int = 2048
  head_dim: int = 1
  lora_r: int = 16
  lora_alpha: int = 32
  lora_dropout: float = 0.05
  base_quantization: str = "8bit"


def make_bnb_config(*, base_quantization: str, compute_dtype: torch.dtype):
  from transformers import BitsAndBytesConfig

  if base_quantization == "8bit":
    return BitsAndBytesConfig(load_in_8bit=True)
  if base_quantization == "4bit":
    return BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=compute_dtype,
    )
  raise ValueError(f"unsupported base_quantization: {base_quantization!r}")


class RewardHead(nn.Module):
  def __init__(self, hidden_size: int, out_dim: int) -> None:
    super().__init__()
    self.linear = nn.Linear(hidden_size, out_dim)

  def forward(self, hidden: torch.Tensor) -> torch.Tensor:
    return self.linear(hidden)


class CausalRewardModel(nn.Module):
  def __init__(self, backbone: nn.Module, head: RewardHead) -> None:
    super().__init__()
    self.backbone = backbone
    self.head = head

  def _decoder(self) -> nn.Module:
    if hasattr(self.backbone, "get_base_model"):
      return self.backbone.get_base_model()
    return self.backbone

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    last_indices: torch.Tensor,
  ) -> torch.Tensor:
    outputs = self._decoder()(
      input_ids=input_ids,
      attention_mask=attention_mask,
      use_cache=False,
    )
    hidden = outputs.last_hidden_state
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    picked = hidden[batch_idx, last_indices].to(dtype=self.head.linear.weight.dtype)
    out = self.head(picked)
    if out.shape[-1] == 1:
      return out.squeeze(-1)
    return out


def encode_pref_pair(
  tokenizer,
  draft: str,
  candidate: str,
  *,
  max_length: int,
) -> tuple[list[int], int]:
  draft_marker = tokenizer.encode(PREF_DRAFT_TOK, add_special_tokens=False)
  cand_marker = tokenizer.encode(PREF_CAND_TOK, add_special_tokens=False)
  draft_body = tokenizer.encode(draft, add_special_tokens=False)
  cand_body = tokenizer.encode(candidate, add_special_tokens=False)
  suffix = cand_marker + cand_body
  prefix = draft_marker + draft_body

  if len(suffix) > max_length:
    ids = suffix[-max_length:]
  elif len(prefix) + len(suffix) > max_length:
    budget = max_length - len(suffix)
    if budget <= len(draft_marker):
      prefix = draft_marker[-budget:] if budget > 0 else []
    else:
      body_budget = budget - len(draft_marker)
      prefix = draft_marker + draft_body[-body_budget:]
    ids = prefix + suffix
  else:
    ids = prefix + suffix

  if not ids:
    raise ValueError("empty encoded sequence")
  return ids, len(ids) - 1


def batch_encode_pref_pairs(
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if len(drafts) != len(candidates):
    raise ValueError("draft/candidate length mismatch")
  pad_id = tokenizer.pad_token_id
  if pad_id is None:
    raise ValueError("tokenizer.pad_token_id is required")
  encoded = [
    encode_pref_pair(tokenizer, d, y, max_length=max_length)
    for d, y in zip(drafts, candidates, strict=True)
  ]
  max_len = max(len(ids) for ids, _ in encoded)
  input_ids: list[list[int]] = []
  attention_mask: list[list[int]] = []
  last_indices: list[int] = []
  for ids, last_idx in encoded:
    pad = max_len - len(ids)
    input_ids.append(ids + [pad_id] * pad)
    attention_mask.append([1] * len(ids) + [0] * pad)
    last_indices.append(last_idx)
  return (
    torch.tensor(input_ids, dtype=torch.long, device=device),
    torch.tensor(attention_mask, dtype=torch.long, device=device),
    torch.tensor(last_indices, dtype=torch.long, device=device),
  )


def build_causal_reward_model(
  cfg: CausalRewardConfig,
  *,
  trust_remote_code: bool = True,
) -> tuple[CausalRewardModel, object]:
  from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
  from transformers import AutoModel, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=trust_remote_code)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  added = tokenizer.add_special_tokens({"additional_special_tokens": PREF_SPECIAL_TOKENS})

  compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
  bnb = make_bnb_config(base_quantization=cfg.base_quantization, compute_dtype=compute_dtype)
  backbone = AutoModel.from_pretrained(
    cfg.base_model,
    quantization_config=bnb,
    device_map={"": 0},
    trust_remote_code=trust_remote_code,
  )
  backbone = prepare_model_for_kbit_training(backbone)
  old_vocab = backbone.get_input_embeddings().weight.size(0)
  if added:
    backbone.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
      emb = backbone.get_input_embeddings().weight
      mean_emb = emb[:old_vocab].mean(dim=0)
      emb[old_vocab:] = mean_emb
  backbone.config.use_cache = False

  lora_cfg = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type="FEATURE_EXTRACTION",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  )
  backbone = get_peft_model(backbone, lora_cfg)
  hidden_size = backbone.config.hidden_size
  head = RewardHead(hidden_size, cfg.head_dim)
  head.to(device=backbone.device, dtype=compute_dtype)
  return CausalRewardModel(backbone, head), tokenizer


def forward_outputs(
  model: CausalRewardModel,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> torch.Tensor:
  input_ids, attention_mask, last_indices = batch_encode_pref_pairs(
    tokenizer,
    drafts,
    candidates,
    max_length=max_length,
    device=device,
  )
  return model(input_ids, attention_mask, last_indices)


def forward_f_delta(
  model: CausalRewardModel,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> torch.Tensor:
  both_drafts = drafts + drafts
  both_cands = candidates + drafts
  g_all = forward_outputs(
    model,
    tokenizer,
    both_drafts,
    both_cands,
    max_length=max_length,
    device=device,
  )
  n = len(drafts)
  return g_all[:n] - g_all[n:]


def forward_bt_pair_deltas(
  model: CausalRewardModel,
  tokenizer,
  drafts: list[str],
  winner_ys: list[str],
  loser_ys: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  n = len(drafts)
  if len(winner_ys) != n or len(loser_ys) != n:
    raise ValueError("draft/winner/loser length mismatch")
  all_drafts = drafts + drafts + drafts
  all_cands = winner_ys + loser_ys + drafts
  g_all = forward_outputs(
    model,
    tokenizer,
    all_drafts,
    all_cands,
    max_length=max_length,
    device=device,
  )
  g_w = g_all[:n]
  g_l = g_all[n : 2 * n]
  g_x = g_all[2 * n :]
  return g_w - g_x, g_l - g_x


def forward_gpm_pair_vectors(
  model: CausalRewardModel,
  tokenizer,
  drafts: list[str],
  winner_ys: list[str],
  loser_ys: list[str],
  *,
  max_length: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  n = len(drafts)
  if len(winner_ys) != n or len(loser_ys) != n:
    raise ValueError("draft/winner/loser length mismatch")
  g_all = forward_outputs(
    model,
    tokenizer,
    drafts + drafts,
    winner_ys + loser_ys,
    max_length=max_length,
    device=device,
  )
  return g_all[:n], g_all[n:]


@torch.no_grad()
def predict_f_delta(
  model: CausalRewardModel,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
  batch_size: int = 8,
) -> list[float]:
  model.eval()
  out: list[float] = []
  for start in range(0, len(drafts), batch_size):
    batch_d = drafts[start : start + batch_size]
    batch_y = candidates[start : start + batch_size]
    vals = forward_f_delta(
      model, tokenizer, batch_d, batch_y, max_length=max_length, device=device
    )
    out.extend(float(x) for x in vals.detach().float().cpu().tolist())
  return out


@torch.no_grad()
def predict_vectors(
  model: CausalRewardModel,
  tokenizer,
  drafts: list[str],
  candidates: list[str],
  *,
  max_length: int,
  device: torch.device,
  batch_size: int = 8,
) -> list[list[float]]:
  model.eval()
  out: list[list[float]] = []
  for start in range(0, len(drafts), batch_size):
    batch_d = drafts[start : start + batch_size]
    batch_y = candidates[start : start + batch_size]
    logits = forward_outputs(
      model, tokenizer, batch_d, batch_y, max_length=max_length, device=device
    )
    out.extend(logits.detach().float().cpu().tolist())
  return out


def save_causal_reward_model(
  model: CausalRewardModel,
  tokenizer,
  output_dir: Path,
  *,
  meta: dict,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  model.backbone.save_pretrained(output_dir)
  torch.save(model.head.state_dict(), output_dir / "reward_head.pt")
  tokenizer.save_pretrained(output_dir)
  (output_dir / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def load_causal_reward_model(model_dir: Path, *, device: torch.device) -> tuple[CausalRewardModel, object, CausalRewardConfig, dict]:
  from peft import PeftModel
  from transformers import AutoModel, AutoTokenizer

  meta_path = model_dir / "meta.json"
  meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
  cfg = CausalRewardConfig(
    base_model=str(meta.get("base_model", "Qwen/Qwen3-8B")),
    max_length=int(meta.get("max_length", 2048)),
    head_dim=int(meta.get("head_dim", meta.get("num_labels", 1))),
    lora_r=int(meta.get("lora_r", 16)),
    lora_alpha=int(meta.get("lora_alpha", 32)),
    lora_dropout=float(meta.get("lora_dropout", 0.05)),
    base_quantization=str(meta.get("base_quantization", "8bit")),
  )
  compute_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
  bnb = make_bnb_config(base_quantization=cfg.base_quantization, compute_dtype=compute_dtype)
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  base = AutoModel.from_pretrained(
    cfg.base_model,
    quantization_config=bnb,
    device_map={"": 0},
    trust_remote_code=True,
  )
  base.config.use_cache = False
  backbone = PeftModel.from_pretrained(base, model_dir)
  head = RewardHead(base.config.hidden_size, cfg.head_dim)
  head.load_state_dict(torch.load(model_dir / "reward_head.pt", map_location="cpu"))
  head.to(device=backbone.device, dtype=compute_dtype)
  model = CausalRewardModel(backbone, head)
  model.eval()
  return model, tokenizer, cfg, meta
