#!/usr/bin/env python3
"""条件付き B 生成実験の共通処理（prompt、分割、長さ計測）。"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONDITIONS = ("x", "y", "xy")
SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
SPLIT_HOLDOUT = "holdout"
MAX_TOTAL_TOKENS = 32768
MAX_NEW_TOKENS = 8192
IM_END_TOKEN = "<|im_end|>"


@dataclass(frozen=True)
class ModelLockSpec:
  model_id: str
  model_revision: str
  tokenizer_revision: str
  eos_token_id: int
  pad_token_id: int
  enable_thinking: bool
  system_text: str


def load_yaml_config(path: Path) -> dict[str, Any]:
  return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
  return sha256_bytes(text.encode("utf-8"))


def group_id_from_source_ids(source_ids: list[str]) -> tuple[str, str]:
  if len(source_ids) == 1:
    return f"source:{source_ids[0]}", "source_id"
  payload = json.dumps(sorted(source_ids), ensure_ascii=False, separators=(",", ":"))
  digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
  return f"group:{digest}", "known_overlap"


def dev_group_rank(group_id: str, *, split_seed: int) -> str:
  return hashlib.sha256(f"{split_seed}\n{group_id}".encode("utf-8")).hexdigest()


def build_user_text(condition: str, draft: str, composer: str) -> str:
  if condition == "x":
    return f"【下書き】\n{draft}"
  if condition == "y":
    return f"【推敲候補】\n{composer}"
  if condition == "xy":
    return f"【下書き】\n{draft}\n\n【推敲候補】\n{composer}"
  raise ValueError(f"unknown condition {condition!r}")


def build_messages(condition: str, draft: str, composer: str, *, system_text: str) -> list[dict[str, str]]:
  return [
    {"role": "system", "content": system_text},
    {"role": "user", "content": build_user_text(condition, draft, composer)},
  ]


def apply_prompt_ids(tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool) -> list[int]:
  out = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=enable_thinking,
  )
  if hasattr(out, "get"):
    ids = out.get("input_ids")
    if ids is not None:
      return list(ids[0] if ids and isinstance(ids[0], list) else ids)
  if isinstance(out, list):
    return out
  raise TypeError(f"unexpected apply_chat_template output: {type(out)!r}")


def build_train_example(
  tokenizer,
  *,
  condition: str,
  draft: str,
  composer: str,
  human: str,
  system_text: str,
  enable_thinking: bool,
  eos_token_id: int,
) -> dict[str, Any]:
  messages = build_messages(condition, draft, composer, system_text=system_text)
  prompt_ids = apply_prompt_ids(tokenizer, messages, enable_thinking=enable_thinking)
  end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
  if end_id != eos_token_id:
    raise ValueError(f"eos mismatch: template={end_id} expected={eos_token_id}")
  target_raw = tokenizer(human, add_special_tokens=False)["input_ids"]
  target_ids = list(target_raw[0] if target_raw and isinstance(target_raw[0], list) else target_raw) + [end_id]
  input_ids = prompt_ids + target_ids
  labels = [-100] * len(prompt_ids) + target_ids
  attention_mask = [1] * len(input_ids)
  return {
    "input_ids": input_ids,
    "labels": labels,
    "attention_mask": attention_mask,
    "prompt_tokens": len(prompt_ids),
    "target_tokens": len(target_ids),
    "train_tokens": len(input_ids),
  }


def compute_length_fields(
  tokenizer,
  *,
  draft: str,
  composer: str,
  human: str,
  system_text: str,
  enable_thinking: bool,
  eos_token_id: int,
) -> dict[str, dict[str, int]]:
  out: dict[str, dict[str, int]] = {}
  for condition in CONDITIONS:
    ex = build_train_example(
      tokenizer,
      condition=condition,
      draft=draft,
      composer=composer,
      human=human,
      system_text=system_text,
      enable_thinking=enable_thinking,
      eos_token_id=eos_token_id,
    )
    prompt_tokens = ex["prompt_tokens"]
    target_tokens = ex["target_tokens"]
    train_tokens = ex["train_tokens"]
    out[condition] = {
      "prompt_tokens": prompt_tokens,
      "target_tokens": target_tokens,
      "train_tokens": train_tokens,
      "inference_total_tokens": prompt_tokens + MAX_NEW_TOKENS,
    }
  return out


def eligibility_from_lengths(lengths: dict[str, dict[str, int]]) -> tuple[bool, list[str]]:
  reasons: list[str] = []
  train_over = False
  infer_over = False
  for condition in CONDITIONS:
    row = lengths[condition]
    if row["train_tokens"] > MAX_TOTAL_TOKENS:
      train_over = True
    if row["inference_total_tokens"] > MAX_TOTAL_TOKENS:
      infer_over = True
  if train_over:
    reasons.append("train_total_over_32768")
  if infer_over:
    reasons.append("inference_budget_over_32768")
  return (not reasons), reasons


def manifest_sha256(manifest: dict[str, Any]) -> str:
  payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
  return sha256_text(payload)


def load_manifest(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> dict[str, dict[str, str]]:
  out: dict[str, dict[str, str]] = {}
  with path.open(encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      sid = str(row["source_id"])
      if sid in out:
        raise ValueError(f"duplicate source_id {sid!r}")
      out[sid] = row
  return out


def eligible_sources(manifest: dict[str, Any], *, split: str, condition: str | None = None) -> list[dict[str, Any]]:
  rows = []
  for src in manifest["sources"]:
    if src.get("split") != split:
      continue
    if not src.get("eligible"):
      continue
    rows.append(src)
  rows.sort(key=lambda r: r["source_id"])
  return rows


def select_dev_groups(
  train_groups: list[str],
  *,
  dev_fraction: float,
  split_seed: int,
) -> set[str]:
  if len(train_groups) < 2:
    raise ValueError(f"need at least 2 train groups for dev split, got {len(train_groups)}")
  n_dev = max(1, math.ceil(dev_fraction * len(train_groups)))
  ranked = sorted(train_groups, key=lambda gid: (dev_group_rank(gid, split_seed=split_seed), gid))
  return set(ranked[:n_dev])


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generation_id(*, condition: str, source_id: str, train_seed: int | None = None) -> str:
  if condition in ("composer", "xy-base"):
    return f"{condition}:shared:{source_id}"
  if train_seed is None:
    raise ValueError(f"train_seed required for condition {condition!r}")
  return f"{condition}:seed-{train_seed}:{source_id}"


def decode_generation_text(tokenizer, generated_ids: list[int], *, eos_token_id: int) -> tuple[str, str, bool]:
  """生成トークン列をデコードする。戻り値は (本文, finish_reason, truncated)。"""
  ids = list(generated_ids)
  truncated = False
  finish_reason = "eos"
  if ids and ids[-1] == eos_token_id:
    ids = ids[:-1]
  elif ids:
    truncated = True
    finish_reason = "max_tokens"
  if not ids:
    finish_reason = "empty" if not generated_ids else finish_reason
  text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
  return text, finish_reason, truncated
