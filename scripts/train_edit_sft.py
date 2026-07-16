#!/usr/bin/env python3
"""編集モデルの QLoRA SFT（系統1フェーズ1）。

入力: data/edit_sft/train.jsonl（chat messages）
出力: LoRA アダプタと tokenizer 設定（原稿本文は成果物に含めない）
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset


def load_messages_jsonl(path: Path, *, limit: int = 0) -> Dataset:
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      messages = obj.get("messages")
      if not messages:
        continue
      # 成果物・学習ログに原稿由来の meta を残さない
      rows.append({"messages": messages})
      if limit > 0 and len(rows) >= limit:
        break
  if not rows:
    raise SystemExit(f"no rows in {path}")
  return Dataset.from_list(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", default="data/edit_sft/train.jsonl")
  parser.add_argument("--model", default="Qwen/Qwen3-8B")
  parser.add_argument("--out-dir", default="", help="default: outputs/edit-sft/<slug>")
  parser.add_argument("--max-seq-length", type=int, default=2048)
  parser.add_argument("--epochs", type=float, default=2.0)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--lora-r", type=int, default=16)
  parser.add_argument("--lora-alpha", type=int, default=32)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--grad-accum", type=int, default=8)
  parser.add_argument("--limit", type=int, default=0, help="train rows (0=all); smoke 用")
  parser.add_argument("--logging-steps", type=int, default=10)
  parser.add_argument("--save-steps", type=int, default=200)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch
  from peft import LoraConfig
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  from trl import SFTConfig, SFTTrainer

  from steering_utils import model_slug

  train_path = Path(args.train)
  if not train_path.is_file():
    raise SystemExit(f"missing {train_path}: run make edit-sft-data first")

  dataset = load_messages_jsonl(train_path, limit=args.limit)
  slug = model_slug(args.model)
  out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/edit-sft") / slug
  out_dir.mkdir(parents=True, exist_ok=True)

  print(
    f"model={args.model} n_train={len(dataset)} out={out_dir} "
    f"epochs={args.epochs} lora_r={args.lora_r}",
    flush=True,
  )

  tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=args.trust_remote_code,
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
  )
  model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=args.trust_remote_code,
  )
  model.config.use_cache = False

  peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  )

  sft_args = SFTConfig(
    output_dir=str(out_dir / "checkpoints"),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=2,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    max_length=args.max_seq_length,
    packing=False,
    seed=args.seed,
    report_to=[],
    gradient_checkpointing=True,
  )

  trainer_kwargs = {
    "model": model,
    "args": sft_args,
    "train_dataset": dataset,
    "peft_config": peft_config,
    "processing_class": tokenizer,
  }
  # TRL 版差: messages 列を chat template で読む経路
  try:
    trainer = SFTTrainer(**trainer_kwargs)
  except TypeError:
    trainer_kwargs.pop("processing_class", None)
    trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

  trainer.train()
  adapter_dir = out_dir / "adapter"
  trainer.save_model(str(adapter_dir))
  tokenizer.save_pretrained(str(adapter_dir))

  meta = {
    "base_model": args.model,
    "adapter_dir": str(adapter_dir),
    "n_train": len(dataset),
    "epochs": args.epochs,
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "max_seq_length": args.max_seq_length,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "grad_accum": args.grad_accum,
    "limit": args.limit,
    "note": "LoRA only; no source manuscript text in artifacts",
  }
  meta_path = out_dir / "train_meta.json"
  meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {adapter_dir}", flush=True)
  print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
  # DOK / 対話シェル双方で scripts から import できるよう
  import sys

  root = Path(__file__).resolve().parent
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  main()
