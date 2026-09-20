#!/usr/bin/env python3
"""Phase 2: 密度スコア s_den = log p_adapter(y|x) - log p_base(y|x)。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import draft_from_row, load_jsonl, normalized_levenshtein
from section_middle_utils import build_revision_prompt

try:
  from sklearn.linear_model import LinearRegression
except ImportError as exc:
  raise SystemExit(f"sklearn required: {exc}") from exc


@dataclass
class ResidModel:
  intercept: float
  coef_abs_y: float
  coef_delta: float

  def predict(self, abs_y: float, delta_chars: float) -> float:
    return self.intercept + self.coef_abs_y * abs_y + self.coef_delta * delta_chars

  def resid(self, s_sum: float, abs_y: float, delta_chars: float) -> float:
    return s_sum - self.predict(abs_y, delta_chars)

  def to_dict(self) -> dict[str, float]:
    return {
      "intercept": self.intercept,
      "coef_abs_y": self.coef_abs_y,
      "coef_delta": self.coef_delta,
    }


def template_text(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
  try:
    return tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=add_generation_prompt,
      enable_thinking=False,
    )
  except TypeError:
    return tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=add_generation_prompt,
    )


def assistant_token_ids(tokenizer, user_content: str, assistant_content: str) -> list[int]:
  full_text = template_text(
    tokenizer,
    [
      {"role": "user", "content": user_content},
      {"role": "assistant", "content": assistant_content},
    ],
    add_generation_prompt=False,
  )
  prompt_text = template_text(
    tokenizer,
    [{"role": "user", "content": user_content}],
    add_generation_prompt=True,
  )
  full_ids = tokenizer.encode(full_text, add_special_tokens=False)
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  if len(full_ids) <= len(prompt_ids):
    return []
  if full_ids[: len(prompt_ids)] != prompt_ids:
    return full_ids[-(len(full_ids) - len(prompt_ids)) :]
  return full_ids[len(prompt_ids) :]


def sequence_logprob(model, input_ids, n_ans: int) -> float | None:
  import torch
  from torch.nn import functional as F

  if n_ans <= 0:
    return None
  with torch.no_grad():
    logits = model(input_ids).logits
  start = input_ids.size(1) - n_ans
  target = input_ids[:, start:]
  pred = logits[:, start - 1 : -1, :]
  lp = F.log_softmax(pred, dim=-1)
  tok_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
  return float(tok_lp.sum().item())


def score_candidate(
  model,
  tokenizer,
  *,
  user_content: str,
  draft: str,
  candidate: str,
  device,
  max_seq_len: int,
) -> dict[str, Any] | None:
  ans_ids = assistant_token_ids(tokenizer, user_content, candidate)
  if not ans_ids:
    return None
  prompt_text = template_text(
    tokenizer,
    [{"role": "user", "content": user_content}],
    add_generation_prompt=True,
  )
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  seq = prompt_ids + ans_ids
  if len(seq) > max_seq_len:
    overflow = len(seq) - max_seq_len
    if overflow >= len(ans_ids):
      return None
    prompt_ids = prompt_ids[-(max_seq_len - len(ans_ids)) :]
    seq = prompt_ids + ans_ids

  import torch

  input_ids = torch.tensor([seq], device=device)
  with model.disable_adapter():
    logp_base = sequence_logprob(model, input_ids, len(ans_ids))
  logp_ad = sequence_logprob(model, input_ids, len(ans_ids))
  if logp_base is None or logp_ad is None:
    return None
  s_sum = logp_ad - logp_base
  n_tokens = len(ans_ids)
  n_chars_draft = len(draft)
  n_chars_cand = len(candidate)
  delta_chars = n_chars_cand - n_chars_draft
  return {
    "s_sum": s_sum,
    "s_mean": s_sum / n_tokens,
    "n_tokens": n_tokens,
    "n_chars_draft": n_chars_draft,
    "n_chars_cand": n_chars_cand,
    "delta_chars": delta_chars,
    "edit_distance_norm": normalized_levenshtein(draft, candidate),
    "logp_adapter": logp_ad,
    "logp_base": logp_base,
  }


def user_content_from_row(row: dict) -> tuple[str, str]:
  messages = row.get("messages")
  if isinstance(messages, list) and len(messages) >= 2:
    user = str(messages[0].get("content") or "")
    draft = draft_from_row(row) or ""
    return user, draft
  draft = str(row.get("draft") or row.get("context_draft") or row.get("source_text") or "")
  if row.get("user_prompt"):
    return str(row["user_prompt"]), draft
  if draft:
    return build_revision_prompt(draft), draft
  raise ValueError("row lacks messages or draft")


def candidates_from_row(row: dict) -> list[dict[str, str]]:
  if row.get("candidates"):
    return [
      {"source": str(c.get("source") or "cand"), "text": str(c["text"])}
      for c in row["candidates"]
      if c.get("text") is not None
    ]
  out: list[dict[str, str]] = []
  if row.get("a_text") is not None:
    out.append({"source": str(row.get("a_source") or "a"), "text": str(row["a_text"])})
  if row.get("b_text") is not None:
    out.append({"source": str(row.get("b_source") or "b"), "text": str(row["b_text"])})
  if not out and row.get("y") is not None:
    out.append({"source": str(row.get("source") or "y"), "text": str(row["y"])})
  return out


def fit_resid_model(rows: list[dict]) -> ResidModel:
  xs: list[list[float]] = []
  ys: list[float] = []
  for row in rows:
    s_sum = row.get("s_sum")
    if s_sum is None:
      continue
    xs.append([abs(row.get("n_chars_cand") or 0), float(row.get("delta_chars") or 0)])
    ys.append(float(s_sum))
  if len(xs) < 3:
    return ResidModel(0.0, 0.0, 0.0)
  reg = LinearRegression()
  reg.fit(np.asarray(xs), np.asarray(ys))
  return ResidModel(
    intercept=float(reg.intercept_),
    coef_abs_y=float(reg.coef_[0]),
    coef_delta=float(reg.coef_[1]),
  )


def apply_resid(rows: list[dict], resid: ResidModel) -> None:
  for row in rows:
    if row.get("s_sum") is None:
      continue
    row["s_resid"] = resid.resid(
      float(row["s_sum"]),
      abs(int(row.get("n_chars_cand") or 0)),
      float(row.get("delta_chars") or 0),
    )
    row["score"] = row["s_resid"]


def load_density_model(adapter_dir: Path, base_model: str, device: str, *, trust_remote_code: bool):
  import torch
  from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer

  tok_src = adapter_dir if (adapter_dir / "tokenizer.json").is_file() else base_model
  tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=trust_remote_code)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
  if device == "cpu":
    dtype = torch.float32

  model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else "cpu",
    attn_implementation="sdpa",
    trust_remote_code=trust_remote_code,
    low_cpu_mem_usage=True,
  )
  model = PeftModel.from_pretrained(model, str(adapter_dir))
  model.eval()
  dev = next(model.parameters()).device
  return model, tokenizer, dev


def score_rows(
  model,
  tokenizer,
  device,
  rows: list[dict],
  *,
  max_seq_len: int,
) -> list[dict]:
  scored: list[dict] = []
  for row in rows:
    user, draft = user_content_from_row(row)
    item_id = str(
      row.get("item_id")
      or row.get("pair_id")
      or (row.get("meta") or {}).get("id")
      or row.get("id")
      or ""
    )
    for cand in candidates_from_row(row):
      metrics = score_candidate(
        model,
        tokenizer,
        user_content=user,
        draft=draft,
        candidate=cand["text"],
        device=device,
        max_seq_len=max_seq_len,
      )
      if metrics is None:
        continue
      scored.append(
        {
          "item_id": item_id,
          "source": cand["source"],
          **metrics,
        }
      )
  return scored


def score_fit_rows_from_train(train_rows: list[dict], model, tokenizer, device, *, max_seq_len: int) -> list[dict]:
  fit_rows: list[dict] = []
  for row in train_rows:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
      continue
    user = str(messages[0].get("content") or "")
    draft = draft_from_row(row) or ""
    assistant = str(messages[1].get("content") or "")
    metrics = score_candidate(
      model,
      tokenizer,
      user_content=user,
      draft=draft,
      candidate=assistant,
      device=device,
      max_seq_len=max_seq_len,
    )
    if metrics is None:
      continue
    fit_rows.append({"item_id": str((row.get("meta") or {}).get("id") or ""), "source": "human", **metrics})
  return fit_rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--adapter-dir", type=Path, required=True)
  parser.add_argument("--fit-train-jsonl", type=Path, required=True)
  parser.add_argument("--score-jsonl", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch

  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

  adapter_dir = args.adapter_dir.resolve()
  if not (adapter_dir / "adapter_config.json").is_file() and (adapter_dir / "adapter").is_dir():
    adapter_dir = adapter_dir / "adapter"

  model, tokenizer, dev = load_density_model(
    adapter_dir, args.base_model, device, trust_remote_code=args.trust_remote_code
  )

  train_rows = load_jsonl(args.fit_train_jsonl)
  fit_scored = score_fit_rows_from_train(
    train_rows, model, tokenizer, dev, max_seq_len=args.max_seq_length
  )
  resid = fit_resid_model(fit_scored)

  score_input = load_jsonl(args.score_jsonl)
  scored = score_rows(
    model, tokenizer, dev, score_input, max_seq_len=args.max_seq_length
  )
  apply_resid(scored, resid)

  out_path = args.out.resolve()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  meta = {
    "adapter_dir": str(args.adapter_dir),
    "fit_train_jsonl": str(args.fit_train_jsonl),
    "score_jsonl": str(args.score_jsonl),
    "n_fit": len(fit_scored),
    "n_scored": len(scored),
    "resid_model": resid.to_dict(),
  }
  with out_path.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
    for row in scored:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(json.dumps({"wrote": str(out_path), "n_scored": len(scored), "resid": resid.to_dict()}, ensure_ascii=False))


if __name__ == "__main__":
  main()
