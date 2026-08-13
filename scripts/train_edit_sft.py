#!/usr/bin/env python3
"""編集モデルの QLoRA SFT。

入力: chat messages 形式の train.jsonl
出力: LoRA アダプタと tokenizer 設定（原稿本文は成果物に含めない）

TRL の版差（dataset_text_field / processing_class / tokenizer の置き場）は
inspect で検出して渡す。フォールバックの当て勘はしない。
"""
from __future__ import annotations

import argparse
import inspect
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


def _accepts(fn, name: str) -> bool:
  try:
    return name in inspect.signature(fn).parameters
  except (TypeError, ValueError):
    return False


def build_sft_config(SFTConfig, *, args, out_dir: Path, torch, n_train: int):
  """版差のあるキーだけ入れた SFTConfig を返す。"""
  kwargs: dict = {
    "output_dir": str(out_dir / "checkpoints"),
    "num_train_epochs": args.epochs,
    "per_device_train_batch_size": args.batch_size,
    "gradient_accumulation_steps": args.grad_accum,
    "learning_rate": args.learning_rate,
    "lr_scheduler_type": "cosine",
    "logging_steps": args.logging_steps,
    # エポック毎に全チェックポイントを残す（学習量とエポック数の関係を後から機械計測で選ぶ）
    "save_strategy": "epoch",
    "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    "packing": False,
    "seed": args.seed,
    "report_to": [],
    "gradient_checkpointing": True,
  }

  # 長さ上限: 新 API は max_length、旧は max_seq_length
  if _accepts(SFTConfig.__init__, "max_length"):
    kwargs["max_length"] = args.max_seq_length
  elif _accepts(SFTConfig.__init__, "max_seq_length"):
    kwargs["max_seq_length"] = args.max_seq_length

  # warmup: warmup_ratio は将来削除されるので steps を優先
  steps_per_epoch = max(
    1,
    (n_train + args.batch_size - 1) // args.batch_size,
  )
  updates_per_epoch = max(
    1,
    (steps_per_epoch + args.grad_accum - 1) // args.grad_accum,
  )
  total_updates = max(1, int(updates_per_epoch * args.epochs))
  warmup_steps = max(1, int(total_updates * 0.03))
  if _accepts(SFTConfig.__init__, "warmup_steps"):
    kwargs["warmup_steps"] = warmup_steps
  elif _accepts(SFTConfig.__init__, "warmup_ratio"):
    kwargs["warmup_ratio"] = 0.03

  # text 列名: 新 API は Config 側
  if _accepts(SFTConfig.__init__, "dataset_text_field"):
    kwargs["dataset_text_field"] = "text"

  return SFTConfig(**kwargs)


def build_trainer(SFTTrainer, *, model, sft_args, dataset, peft_config, tokenizer):
  """版差のある tokenizer 渡しだけ切り替えて SFTTrainer を作る。"""
  kwargs: dict = {
    "model": model,
    "args": sft_args,
    "train_dataset": dataset,
    "peft_config": peft_config,
  }

  if _accepts(SFTTrainer.__init__, "processing_class"):
    kwargs["processing_class"] = tokenizer
  elif _accepts(SFTTrainer.__init__, "tokenizer"):
    kwargs["tokenizer"] = tokenizer
  else:
    raise SystemExit(
      "この TRL の SFTTrainer は processing_class / tokenizer のどちらも受けない"
    )

  # 旧 API: dataset_text_field が Trainer 引数
  if (
    not _accepts(type(sft_args).__init__, "dataset_text_field")
    and _accepts(SFTTrainer.__init__, "dataset_text_field")
  ):
    kwargs["dataset_text_field"] = "text"

  return SFTTrainer(**kwargs)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", default="data/edit_sft/train.jsonl")
  parser.add_argument("--model", default="Qwen/Qwen3-8B")
  parser.add_argument("--out-dir", default="", help="default: outputs/edit-sft/<slug>")
  parser.add_argument("--max-seq-length", type=int, default=8192)
  parser.add_argument("--epochs", type=float, default=15.0)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--lora-r", type=int, default=16)
  parser.add_argument("--lora-alpha", type=int, default=32)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--grad-accum", type=int, default=8)
  parser.add_argument("--limit", type=int, default=0, help="train rows (0=all); smoke 用")
  parser.add_argument("--logging-steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch
  import trl
  from peft import LoraConfig
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  from trl import SFTConfig, SFTTrainer

  from steering_utils import model_slug

  print(f"trl={trl.__version__}", flush=True)

  train_path = Path(args.train)
  if not train_path.is_file():
    raise SystemExit(f"missing {train_path}: run make pairsplit-data first")

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
    target_modules=[
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj",
    ],
  )

  # Qwen3: 生成側は enable_thinking=False が既定。学習の直列化も揃える。
  def messages_to_text(messages: list) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
      return tokenizer.apply_chat_template(
        messages, enable_thinking=False, **kwargs
      )
    except TypeError:
      return tokenizer.apply_chat_template(messages, **kwargs)

  dataset = dataset.map(
    lambda ex: {"text": messages_to_text(ex["messages"])},
    remove_columns=[c for c in dataset.column_names if c != "text"],
  )
  sample = dataset[0]["text"]
  if "<think>" in sample and "</think>" not in sample[:200]:
    print(
      "WARNING: chat template may have left an open think block; "
      f"text_head={sample[:120]!r}",
      flush=True,
    )

  sft_args = build_sft_config(
    SFTConfig, args=args, out_dir=out_dir, torch=torch, n_train=len(dataset)
  )
  trainer = build_trainer(
    SFTTrainer,
    model=model,
    sft_args=sft_args,
    dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
  )

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
    "trl_version": trl.__version__,
    "note": "LoRA only; no source manuscript text in artifacts",
  }
  meta_path = out_dir / "train_meta.json"
  meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {adapter_dir}", flush=True)
  print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
  import sys

  root = Path(__file__).resolve().parent
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  main()
