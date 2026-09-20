#!/usr/bin/env python3
"""編集者尤度 R の QLoRA SFT 学習（A2 または edit_sft_all）。"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import ead_out, ead_work, load_jsonl, repo_root

A2_EXPECTED = 379
A2_TRAIN_REL_PATHS = (
  "data/edit_sft_section/train.jsonl",
  "data/edit_sft_section/heldout.jsonl",
)


def _accepts(fn, name: str) -> bool:
  try:
    return name in inspect.signature(fn).parameters
  except (TypeError, ValueError):
    return False


def load_exclude_ids(root: Path, variant: str) -> set[str]:
  if variant != "excl":
    return set()
  out: set[str] = set()
  for name in ("exclude_ids_D.json", "exclude_ids_B.json"):
    path = ead_work() / name
    if not path.is_file():
      path = root / "outputs/ead/work" / name
    if path.is_file():
      obj = json.loads(path.read_text(encoding="utf-8"))
      out.update(str(x) for x in obj.get("ids") or [])
  return out


def filter_train_rows(rows: list[dict], exclude: set[str]) -> list[dict]:
  kept: list[dict] = []
  for row in rows:
    meta = row.get("meta") or {}
    item_id = str(meta.get("id") or row.get("id") or "")
    if item_id and item_id in exclude:
      continue
    messages = row.get("messages")
    if not messages:
      continue
    kept.append(row)
  return kept


def load_a2_train_rows(root: Path) -> tuple[list[dict], list[str]]:
  rows: list[dict] = []
  sources: list[str] = []
  for rel in A2_TRAIN_REL_PATHS:
    path = root / rel
    if not path.is_file():
      raise SystemExit(f"missing {path}")
    part = load_jsonl(path)
    rows.extend(part)
    sources.append(str(path.relative_to(root)))
  rows = filter_train_rows(rows, set())
  return rows, sources


def resolve_training(root: Path, args) -> tuple[list[dict], list[str], str, int, str]:
  if args.corpus == "a2":
    rows, sources = load_a2_train_rows(root)
    if args.limit > 0:
      rows = rows[: args.limit]
    if args.limit == 0 and len(rows) != A2_EXPECTED:
      raise SystemExit(f"A2: n_train={len(rows)} expected {A2_EXPECTED}")
    return rows, sources, "ead-den-a2", 0, "a2"

  if not args.variant:
    raise SystemExit("--variant excl|full is required with --corpus all")

  train_src = root / "data/edit_sft_all/train.jsonl"
  if not train_src.is_file():
    raise SystemExit(f"missing {train_src}")

  exclude = load_exclude_ids(root, args.variant)
  rows = filter_train_rows(load_jsonl(train_src), exclude)
  if args.limit > 0:
    rows = rows[: args.limit]

  expected = 1516 if args.variant == "excl" else 1716
  if args.limit == 0 and len(rows) != expected:
    print(
      f"WARNING: n_train={len(rows)} expected {expected} for variant={args.variant}",
      flush=True,
    )
  return rows, [str(train_src.relative_to(root))], f"ead-den-a-{args.variant}", len(exclude), args.variant


def messages_to_text(tokenizer, messages: list) -> str:
  kwargs = {"tokenize": False, "add_generation_prompt": False}
  try:
    return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
  except TypeError:
    return tokenizer.apply_chat_template(messages, **kwargs)


def count_truncated(rows: list[dict], tokenizer, max_seq_len: int) -> dict:
  truncated = 0
  lengths: list[int] = []
  for row in rows:
    text = messages_to_text(tokenizer, row["messages"])
    n = len(tokenizer.encode(text, add_special_tokens=False))
    lengths.append(n)
    if n > max_seq_len:
      truncated += 1
  return {
    "n_rows": len(rows),
    "n_truncated": truncated,
    "truncated_frac": (truncated / len(rows)) if rows else 0.0,
    "max_token_len": max(lengths) if lengths else 0,
    "median_token_len": sorted(lengths)[len(lengths) // 2] if lengths else 0,
  }


def build_sft_config(SFTConfig, *, args, out_dir: Path, torch, n_train: int):
  kwargs: dict = {
    "output_dir": str(out_dir / "checkpoints"),
    "num_train_epochs": args.epochs,
    "per_device_train_batch_size": args.batch_size,
    "gradient_accumulation_steps": args.grad_accum,
    "learning_rate": args.learning_rate,
    "lr_scheduler_type": "cosine",
    "logging_steps": args.logging_steps,
    "save_strategy": "epoch",
    "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    "packing": False,
    "seed": args.seed,
    "report_to": [],
    "gradient_checkpointing": True,
  }
  if _accepts(SFTConfig.__init__, "max_length"):
    kwargs["max_length"] = args.max_seq_length
  elif _accepts(SFTConfig.__init__, "max_seq_length"):
    kwargs["max_seq_length"] = args.max_seq_length

  steps_per_epoch = max(1, (n_train + args.batch_size - 1) // args.batch_size)
  updates_per_epoch = max(1, (steps_per_epoch + args.grad_accum - 1) // args.grad_accum)
  total_updates = max(1, int(updates_per_epoch * args.epochs))
  warmup_steps = max(1, int(total_updates * 0.03))
  if _accepts(SFTConfig.__init__, "warmup_steps"):
    kwargs["warmup_steps"] = warmup_steps
  elif _accepts(SFTConfig.__init__, "warmup_ratio"):
    kwargs["warmup_ratio"] = 0.03
  if _accepts(SFTConfig.__init__, "dataset_text_field"):
    kwargs["dataset_text_field"] = "text"
  if _accepts(SFTConfig.__init__, "optim"):
    kwargs["optim"] = "adamw_8bit"
  return SFTConfig(**kwargs)


def build_trainer(SFTTrainer, *, model, sft_args, dataset, peft_config, tokenizer):
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
    raise SystemExit("SFTTrainer requires processing_class or tokenizer")
  if (
    not _accepts(type(sft_args).__init__, "dataset_text_field")
    and _accepts(SFTTrainer.__init__, "dataset_text_field")
  ):
    kwargs["dataset_text_field"] = "text"
  return SFTTrainer(**kwargs)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument(
    "--corpus",
    choices=("a2", "all"),
    default="all",
    help="a2=A2 379 節（ead-den-a2）。all=edit_sft_all（legacy excl/full）",
  )
  parser.add_argument("--variant", choices=("excl", "full"), default=None)
  parser.add_argument("--model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="", help="cuda device index or cpu")
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--epochs", type=float, default=3.0)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--lora-r", type=int, default=16)
  parser.add_argument("--lora-alpha", type=int, default=32)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--grad-accum", type=int, default=16)
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--logging-steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch
  import trl
  from peft import LoraConfig
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  from trl import SFTConfig, SFTTrainer

  root = args.root.resolve()
  if args.device:
    if args.device.isdigit():
      os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    elif args.device == "cpu":
      os.environ["CUDA_VISIBLE_DEVICES"] = ""

  rows, train_sources, adapter_name, n_excluded, run_label = resolve_training(root, args)

  out_dir = ead_out() / "adapters" / adapter_name
  out_dir.mkdir(parents=True, exist_ok=True)
  train_path = ead_work() / f"{adapter_name}-train.jsonl"
  train_path.parent.mkdir(parents=True, exist_ok=True)
  with train_path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

  tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=args.trust_remote_code,
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  truncate_stats = count_truncated(rows, tokenizer, args.max_seq_length)
  print(json.dumps({"truncate": truncate_stats, "n_train": len(rows)}, ensure_ascii=False), flush=True)

  dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
  dataset = dataset.map(
    lambda ex: {"text": messages_to_text(tokenizer, ex["messages"])},
    remove_columns=[c for c in dataset.column_names if c != "text"],
  )

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
    attn_implementation="sdpa",
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
    "corpus": args.corpus,
    "variant": run_label,
    "base_model": args.model,
    "adapter_dir": str(adapter_dir),
    "train_path": str(train_path.relative_to(root)),
    "train_sources": train_sources,
    "n_train": len(rows),
    "n_excluded": n_excluded,
    "epochs": args.epochs,
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "max_seq_length": args.max_seq_length,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "grad_accum": args.grad_accum,
    "limit": args.limit,
    "seed": args.seed,
    "truncate": truncate_stats,
    "trl_version": trl.__version__,
  }
  meta_path = out_dir / "train_meta.json"
  meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {adapter_dir}", flush=True)
  print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
  main()
