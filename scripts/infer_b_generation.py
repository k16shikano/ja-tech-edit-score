#!/usr/bin/env python3
"""holdout に対する B 条件付き生成推論（6節 prompt、8節 greedy）。"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from b_generation_common import (
  MAX_NEW_TOKENS,
  MAX_TOTAL_TOKENS,
  apply_prompt_ids,
  build_messages,
  decode_generation_text,
  eligible_sources,
  generation_id,
  load_manifest,
  load_records,
  load_yaml_config,
  sha256_text,
  write_json,
)
from b_generation_model import bf16_autocast, load_model_for_inference


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


TRAIN_CONDITIONS = ("x", "y", "xy")


def infer_condition_name(raw: str) -> str:
  if raw in TRAIN_CONDITIONS or raw == "xy-base":
    return raw
  raise ValueError(f"unknown condition {raw!r}")


def prompt_condition(raw: str) -> str:
  return "xy" if raw == "xy-base" else raw


def build_generation_config(cfg: dict, model_lock: dict) -> dict:
  return {
    "enable_thinking": bool(model_lock.get("enable_thinking", False)),
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "repetition_penalty": 1.0,
    "max_new_tokens": int(cfg.get("max_new_tokens", MAX_NEW_TOKENS)),
    "eos_token_id": int(model_lock["eos_token_id"]),
    "pad_token_id": int(model_lock["pad_token_id"]),
    "use_cache": True,
  }


@torch.no_grad()
def generate_one(
  model,
  tokenizer,
  *,
  prompt_ids: list[int],
  gen_cfg: dict,
) -> tuple[list[int], str | None]:
  device = next(model.parameters()).device
  input_ids = torch.tensor([prompt_ids], device=device)
  try:
    with bf16_autocast(device):
      out = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        do_sample=bool(gen_cfg["do_sample"]),
        num_beams=int(gen_cfg["num_beams"]),
        repetition_penalty=float(gen_cfg["repetition_penalty"]),
        eos_token_id=int(gen_cfg["eos_token_id"]),
        pad_token_id=int(gen_cfg["pad_token_id"]),
        use_cache=bool(gen_cfg["use_cache"]),
      )
  except Exception as exc:
    return [], str(exc)
  seq = out[0].tolist()
  return seq[len(prompt_ids) :], None


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=Path("data/b_generation/manifest.json"))
  parser.add_argument("--split", default="holdout")
  parser.add_argument("--condition", required=True)
  parser.add_argument("--seed", type=int, default=0, help="train seed（xy-base/composer では無視）")
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--config", type=Path, default=Path("configs/b_generation/common.yaml"))
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--resume", action="store_true", help="出力済み generation_id を飛ばす")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise SystemExit("CUDA required for infer-b-generation")

  condition = infer_condition_name(args.condition)
  code_root = Path(__file__).resolve().parents[1]
  manifest_path = code_root / args.manifest
  manifest = load_manifest(manifest_path)
  data_dir = manifest_path.parent
  records = load_records(data_dir / manifest["records_file"])
  cfg = load_yaml_config(code_root / args.config)
  model_lock = json.loads((data_dir / manifest["model_lock_file"]).read_text(encoding="utf-8"))
  system_text = str(cfg.get("system_text", "")).strip()
  enable_thinking = bool(model_lock.get("enable_thinking", False))
  eos_token_id = int(model_lock["eos_token_id"])

  gen_cfg = build_generation_config(cfg, model_lock)
  sources = eligible_sources(manifest, split=args.split)
  if not sources:
    raise SystemExit(f"no eligible sources for split={args.split!r}")

  adapter_path: Path | None
  train_seed: int | None
  if condition == "xy-base":
    adapter_path = None
    train_seed = None
    default_ckpt = "base-only"
  else:
    train_seed = int(args.seed)
    adapter_path = (
      args.checkpoint
      if args.checkpoint is not None
      else code_root / "outputs/b_generation" / condition / f"seed-{train_seed}" / "epoch-3"
    )
    default_ckpt = str(adapter_path)
    if not adapter_path.is_dir():
      raise SystemExit(f"missing checkpoint: {adapter_path}")

  if args.out is not None:
    out_path = args.out
  elif condition == "xy-base":
    out_path = code_root / "outputs/b_generation/predictions/xy-base.jsonl"
  else:
    out_path = code_root / "outputs/b_generation/predictions" / f"{condition}-seed-{train_seed}.jsonl"

  device = torch.device(str(cfg.get("device", "cuda:0")))
  model, tokenizer = load_model_for_inference(cfg, device=device, adapter_path=adapter_path)

  done_ids: set[str] = set()
  if args.resume and out_path.is_file():
    with out_path.open(encoding="utf-8") as f:
      for line in f:
        if line.strip():
          done_ids.add(str(json.loads(line)["generation_id"]))

  pcond = prompt_condition(condition)
  n_total = len(sources)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  n_written = 0
  n_skipped = 0
  for idx, src in enumerate(sources, start=1):
    sid = src["source_id"]
    gid = generation_id(condition=condition, source_id=sid, train_seed=train_seed)
    if gid in done_ids:
      n_skipped += 1
      continue

    rec = records[sid]
    prompt_len_expected = src["lengths"][pcond]["prompt_tokens"]
    infer_budget = src["lengths"][pcond]["inference_total_tokens"]
    if infer_budget > MAX_TOTAL_TOKENS:
      raise SystemExit(f"inference budget over 32768: {sid} {infer_budget}")

    messages = build_messages(pcond, rec["draft"], rec["composer"], system_text=system_text)
    prompt_ids = apply_prompt_ids(tokenizer, messages, enable_thinking=enable_thinking)
    if len(prompt_ids) != prompt_len_expected:
      raise SystemExit(
        f"prompt token mismatch {sid}: prepare={prompt_len_expected} infer={len(prompt_ids)}"
      )

    t0 = time.time()
    gen_ids, err = generate_one(model, tokenizer, prompt_ids=prompt_ids, gen_cfg=gen_cfg)
    elapsed = time.time() - t0
    if err:
      row = {
        "generation_id": generation_id(condition=condition, source_id=sid, train_seed=train_seed),
        "source_id": sid,
        "condition": condition,
        "train_seed": train_seed,
        "checkpoint": default_ckpt,
        "text": "",
        "text_sha256": sha256_text(""),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": 0,
        "finish_reason": "error",
        "truncated": False,
        "error": err,
        "elapsed_sec": elapsed,
        "generation_config": gen_cfg,
      }
    else:
      text, finish_reason, truncated = decode_generation_text(
        tokenizer, gen_ids, eos_token_id=eos_token_id
      )
      row = {
        "generation_id": generation_id(condition=condition, source_id=sid, train_seed=train_seed),
        "source_id": sid,
        "condition": condition,
        "train_seed": train_seed,
        "checkpoint": default_ckpt,
        "text": text,
        "text_sha256": sha256_text(text),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(gen_ids),
        "finish_reason": finish_reason,
        "truncated": truncated,
        "error": None,
        "elapsed_sec": elapsed,
        "generation_config": gen_cfg,
      }
    with out_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_written += 1
    print(
      f"infer {condition} seed={train_seed} {idx}/{n_total} {sid} "
      f"tokens={row['generated_tokens']} {row['finish_reason']} {elapsed:.1f}s",
      flush=True,
    )

  rows = load_jsonl(out_path) if out_path.is_file() else []

  meta = {
    "condition": condition,
    "train_seed": train_seed,
    "checkpoint": default_ckpt,
    "split": args.split,
    "n_sources": len(rows),
    "n_skipped": n_skipped,
    "n_written_this_run": n_written,
    "n_error": sum(1 for r in rows if r["finish_reason"] == "error"),
    "n_truncated": sum(1 for r in rows if r["truncated"]),
    "n_empty": sum(1 for r in rows if not r["text"]),
    "out": str(out_path),
  }
  write_json(out_path.with_suffix(".meta.json"), meta)
  del model
  gc.collect()
  torch.cuda.empty_cache()
  print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
  main()
