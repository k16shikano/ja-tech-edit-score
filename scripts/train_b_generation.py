#!/usr/bin/env python3
"""条件付き B 生成の LoRA SFT（prompt/labels マスク、6節仕様）。"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from b_generation_common import (
  build_train_example,
  eligible_sources,
  load_manifest,
  load_records,
  load_yaml_config,
  write_json,
)
from b_generation_model import bf16_autocast, build_optimizer, forward_causal_loss, load_model_and_tokenizer


class TokenizedDataset(Dataset):
  def __init__(self, examples: list[dict]):
    self.examples = examples

  def __len__(self) -> int:
    return len(self.examples)

  def __getitem__(self, idx: int) -> dict:
    return self.examples[idx]


def teacher_token_count(labels: list[int]) -> int:
  return sum(1 for x in labels[1:] if x != -100)


def pad_batch(items: list[dict], *, pad_token_id: int) -> dict[str, torch.Tensor | int]:
  max_len = max(len(x["input_ids"]) for x in items)
  input_ids = []
  labels = []
  attention_mask = []
  weights = []
  prompt_tokens = int(items[0]["prompt_tokens"])
  for ex in items:
    pad = max_len - len(ex["input_ids"])
    input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
    labels.append(ex["labels"] + [-100] * pad)
    attention_mask.append(ex["attention_mask"] + [0] * pad)
    weights.append(float(ex["teacher_tokens"]))
  total = sum(weights)
  return {
    "input_ids": torch.tensor(input_ids, dtype=torch.long),
    "labels": torch.tensor(labels, dtype=torch.long),
    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    "teacher_tokens": torch.tensor(weights, dtype=torch.float32),
    "total_teacher_tokens": torch.tensor(total, dtype=torch.float32),
    "prompt_tokens": prompt_tokens,
  }


def microbatch_loss(model, batch: dict) -> tuple[torch.Tensor, float]:
  n_i = float(batch["teacher_tokens"].item())
  t = float(batch["total_teacher_tokens"].item())
  loss_mean = forward_causal_loss(
    model,
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
    prompt_tokens=int(batch["prompt_tokens"]),
  )
  weighted = float(loss_mean.detach().cpu()) * n_i / t
  return loss_mean * (n_i / t), weighted


@torch.no_grad()
def eval_nll(model, examples: list[dict], *, pad_token_id: int, device: torch.device, batch_size: int = 1) -> float:
  model.eval()
  total_loss = 0.0
  total_tokens = 0.0
  for start in range(0, len(examples), batch_size):
    batch_items = examples[start : start + batch_size]
    batch = pad_batch(batch_items, pad_token_id=pad_token_id)
    batch_tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with bf16_autocast(device):
      loss_val = float(
        forward_causal_loss(
          model,
          input_ids=batch_tensors["input_ids"],
          attention_mask=batch_tensors["attention_mask"],
          labels=batch_tensors["labels"],
          prompt_tokens=int(batch["prompt_tokens"]),
        ).item()
      )
    for ex in batch_items:
      n = ex["teacher_tokens"]
      total_loss += loss_val * n
      total_tokens += n
  return total_loss / total_tokens if total_tokens else math.nan


@dataclass
class TrainState:
  epoch: int = 0
  global_step: int = 0


def build_examples(
  records: dict[str, dict],
  sources: list[dict],
  *,
  condition: str,
  tokenizer,
  system_text: str,
  enable_thinking: bool,
  eos_token_id: int,
) -> list[dict]:
  examples = []
  for src in sources:
    rec = records[src["source_id"]]
    ex = build_train_example(
      tokenizer,
      condition=condition,
      draft=rec["draft"],
      composer=rec["composer"],
      human=rec["human"],
      system_text=system_text,
      enable_thinking=enable_thinking,
      eos_token_id=eos_token_id,
    )
    ex["source_id"] = src["source_id"]
    ex["teacher_tokens"] = teacher_token_count(ex["labels"])
    examples.append(ex)
  return examples


def train_loop(
  model,
  train_examples: list[dict],
  dev_examples: list[dict],
  *,
  cfg: dict,
  pad_token_id: int,
  device: torch.device,
  out_dir: Path,
  seed: int,
) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  grad_accum = int(cfg["gradient_accumulation_steps"])
  lr = float(cfg["learning_rate"])
  epochs = float(cfg["num_train_epochs"])
  max_grad_norm = float(cfg["max_grad_norm"])
  weight_decay = float(cfg["weight_decay"])

  optim = build_optimizer(
    model,
    cfg,
  )
  total_updates = max(1, math.ceil(len(train_examples) / grad_accum) * int(epochs))
  warmup = max(1, int(total_updates * float(cfg["warmup_ratio"])))

  def lr_at(step: int) -> float:
    if step < warmup:
      return lr * (step / warmup)
    progress = (step - warmup) / max(1, total_updates - warmup)
    return lr * (0.5 * (1.0 + math.cos(math.pi * progress)))

  scheduler_step = 0
  log_rows = []
  state = TrainState()

  for epoch in range(int(epochs)):
    state.epoch = epoch + 1
    order = list(range(len(train_examples)))
    random.shuffle(order)
    model.train()
    accum_loss = 0.0
    window_indices: list[int] = []

    def run_accum_window(idxs: list[int]) -> None:
      nonlocal accum_loss, scheduler_step, state
      chunk = [train_examples[i] for i in idxs]
      if not chunk:
        return
      t = float(sum(ex["teacher_tokens"] for ex in chunk))
      for ex in chunk:
        batch = pad_batch([ex], pad_token_id=pad_token_id)
        batch["total_teacher_tokens"] = torch.tensor(t, dtype=torch.float32)
        batch_tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch_tensors["prompt_tokens"] = int(batch["prompt_tokens"])
        batch_tensors["total_teacher_tokens"] = batch["total_teacher_tokens"].to(device)
        loss, weighted = microbatch_loss(model, batch_tensors)
        loss.backward()
        accum_loss += weighted
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      for pg in optim.param_groups:
        pg["lr"] = lr_at(scheduler_step)
      optim.step()
      optim.zero_grad(set_to_none=True)
      scheduler_step += 1
      state.global_step += 1
      log_rows.append(
        {
          "epoch": state.epoch,
          "global_step": state.global_step,
          "train_loss_weighted": accum_loss,
          "accum_teacher_tokens": t,
        }
      )
      accum_loss = 0.0

    pos = 0
    while pos < len(order):
      end = min(pos + grad_accum, len(order))
      run_accum_window(order[pos:end])
      pos = end

    dev_nll = eval_nll(model, dev_examples, pad_token_id=pad_token_id, device=device)
    ckpt_dir = out_dir / f"epoch-{state.epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    log_rows.append({"epoch": state.epoch, "dev_nll": dev_nll})
    print(f"epoch={state.epoch} dev_nll={dev_nll:.6f}", flush=True)

  best = min((r for r in log_rows if "dev_nll" in r), key=lambda r: r["dev_nll"], default=None)
  meta = {
    "condition": cfg.get("_condition"),
    "seed": seed,
    "train_examples": len(train_examples),
    "dev_examples": len(dev_examples),
    "epochs": epochs,
    "best_dev_nll": best,
    "log": log_rows,
  }
  write_json(out_dir / "train_meta.json", meta)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, default=Path("data/b_generation"))
  parser.add_argument("--condition", required=True, choices=["x", "y", "xy"])
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--config", type=Path, default=Path("configs/b_generation/common.yaml"))
  parser.add_argument("--out", type=Path, default="")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise SystemExit("CUDA required for train-b-generation")

  code_root = Path(__file__).resolve().parents[1]
  data_dir = code_root / args.data
  manifest = load_manifest(data_dir / "manifest.json")
  records = load_records(data_dir / manifest["records_file"])
  cfg = load_yaml_config(code_root / args.config)
  cfg["_condition"] = args.condition
  model_lock = json.loads((data_dir / manifest["model_lock_file"]).read_text(encoding="utf-8"))

  system_text = str(cfg.get("system_text", "")).strip()
  enable_thinking = bool(model_lock.get("enable_thinking", False))
  eos_token_id = int(model_lock["eos_token_id"])
  pad_token_id = int(model_lock["pad_token_id"])

  train_sources = eligible_sources(manifest, split="train")
  dev_sources = eligible_sources(manifest, split="dev")
  if not train_sources or not dev_sources:
    raise SystemExit(f"empty split train={len(train_sources)} dev={len(dev_sources)}")

  device = torch.device(str(cfg.get("device", "cuda:0")))
  model, tokenizer = load_model_and_tokenizer(cfg, device=device)

  train_examples = build_examples(
    records,
    train_sources,
    condition=args.condition,
    tokenizer=tokenizer,
    system_text=system_text,
    enable_thinking=enable_thinking,
    eos_token_id=eos_token_id,
  )
  dev_examples = build_examples(
    records,
    dev_sources,
    condition=args.condition,
    tokenizer=tokenizer,
    system_text=system_text,
    enable_thinking=enable_thinking,
    eos_token_id=eos_token_id,
  )

  out_dir = Path(args.out) if args.out else code_root / "outputs/b_generation" / args.condition / f"seed-{args.seed}"
  if not out_dir.is_absolute():
    out_dir = code_root / out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  train_loop(
    model,
    train_examples,
    dev_examples,
    cfg=cfg,
    pad_token_id=pad_token_id,
    device=device,
    out_dir=out_dir,
    seed=args.seed,
  )
  tokenizer.save_pretrained(str(out_dir / "tokenizer"))
  print(json.dumps({"wrote": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
  main()
