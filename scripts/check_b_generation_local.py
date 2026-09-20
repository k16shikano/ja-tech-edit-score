#!/usr/bin/env python3
"""Pro 4000 上で B 生成学習の事前実行（OOM 確認）。"""
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
  CONDITIONS,
  build_train_example,
  load_manifest,
  load_records,
  load_yaml_config,
  write_json,
)
from b_generation_model import bf16_autocast, build_optimizer, forward_causal_loss, load_model_and_tokenizer


def pick_stress_sources(manifest: dict) -> dict[str, dict]:
  eligible = [s for s in manifest["sources"] if s.get("eligible")]
  if not eligible:
    raise SystemExit("no eligible sources in manifest")

  def key_train(s):
    return max(s["lengths"][c]["train_tokens"] for c in CONDITIONS)

  def key_infer(s):
    return max(s["lengths"][c]["prompt_tokens"] for c in CONDITIONS)

  def key_target(s):
    return s["lengths"]["x"]["target_tokens"]

  return {
    "train_longest": max(eligible, key=key_train),
    "infer_prompt_longest": max(eligible, key=key_infer),
    "target_longest": max(eligible, key=key_target),
  }


def run_train_probe(model, example: dict, *, device: torch.device, cfg: dict) -> dict:
  input_ids = torch.tensor([example["input_ids"]], device=device)
  labels = torch.tensor([example["labels"]], device=device)
  attention_mask = torch.tensor([example["attention_mask"]], device=device)
  optim = build_optimizer(model, cfg)
  torch.cuda.reset_peak_memory_stats(device)
  t0 = time.time()
  model.train()
  loss = forward_causal_loss(
    model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    prompt_tokens=example["prompt_tokens"],
  )
  loss.backward()
  optim.step()
  optim.zero_grad(set_to_none=True)
  first_loss = float(loss.detach().cpu())
  del optim, loss
  gc.collect()
  torch.cuda.empty_cache()
  model.eval()
  with torch.no_grad():
    with bf16_autocast(device):
      out2 = forward_causal_loss(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_tokens=example["prompt_tokens"],
      )
      second_loss = float(out2.detach().cpu())
  elapsed = time.time() - t0
  return {
    "train_tokens": len(example["input_ids"]),
    "first_step_loss": first_loss,
    "second_forward_loss": second_loss,
    "elapsed_sec": elapsed,
    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    "status": "ok",
  }


def run_infer_probe(model, tokenizer, example: dict, *, device: torch.device, max_new_tokens: int) -> dict:
  prompt_ids = example["input_ids"][: example["prompt_tokens"]]
  input_ids = torch.tensor([prompt_ids], device=device)
  torch.cuda.reset_peak_memory_stats(device)
  t0 = time.time()
  model.eval()
  with torch.no_grad(), bf16_autocast(device):
    _ = model.generate(
      input_ids=input_ids,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      num_beams=1,
      eos_token_id=int(tokenizer.eos_token_id),
      pad_token_id=int(tokenizer.pad_token_id),
      use_cache=True,
    )
  elapsed = time.time() - t0
  gc.collect()
  torch.cuda.empty_cache()
  return {
    "prompt_tokens": int(input_ids.shape[1]),
    "max_new_tokens": max_new_tokens,
    "elapsed_sec": elapsed,
    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    "status": "ok",
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=Path("data/b_generation/manifest.json"))
  parser.add_argument("--config", type=Path, default=Path("configs/b_generation/common.yaml"))
  parser.add_argument("--out", type=Path, default=Path("outputs/b_generation/local_preflight.json"))
  parser.add_argument("--condition", default="xy", choices=list(CONDITIONS))
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise SystemExit("CUDA required for local preflight")

  code_root = Path(__file__).resolve().parents[1]
  manifest_path = code_root / args.manifest
  manifest = load_manifest(manifest_path)
  records_path = manifest_path.parent / manifest["records_file"]
  records = load_records(records_path)
  cfg = load_yaml_config(code_root / args.config)
  model_lock = json.loads((records_path.parent / manifest["model_lock_file"]).read_text(encoding="utf-8"))
  system_text = str(cfg.get("system_text", "")).strip()
  enable_thinking = bool(model_lock.get("enable_thinking", False))
  eos_token_id = int(model_lock["eos_token_id"])

  stress = pick_stress_sources(manifest)
  device = torch.device("cuda:0")
  gpu_name = torch.cuda.get_device_name(device)
  gpu_total = torch.cuda.get_device_properties(device).total_memory

  model, tokenizer = load_model_and_tokenizer(cfg, device=device)

  probes: dict = {}
  status = "ok"
  error = None
  probe_order = ("train_longest", "target_longest", "infer_prompt_longest")
  try:
    for name in probe_order:
      src = stress[name]
      rec = records[src["source_id"]]
      ex = build_train_example(
        tokenizer,
        condition=args.condition if name != "target_longest" else "x",
        draft=rec["draft"],
        composer=rec["composer"],
        human=rec["human"],
        system_text=system_text,
        enable_thinking=enable_thinking,
        eos_token_id=eos_token_id,
      )
      if name == "infer_prompt_longest":
        probes[name] = run_infer_probe(
          model,
          tokenizer,
          ex,
          device=device,
          max_new_tokens=int(cfg.get("max_new_tokens", 8192)),
        )
      else:
        probes[name] = run_train_probe(model, ex, device=device, cfg=cfg)
      gc.collect()
      torch.cuda.empty_cache()
  except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
      status = "oom"
      error = str(exc)
    else:
      raise
  finally:
    del model
    gc.collect()
    torch.cuda.empty_cache()

  report = {
    "schema_version": "b-local-preflight-v1",
    "status": status,
    "error": error,
    "gpu": {"name": gpu_name, "total_memory_bytes": int(gpu_total)},
    "condition": args.condition,
    "stress_sources": {k: v["source_id"] for k, v in stress.items()},
    "probes": probes,
  }
  out = code_root / args.out
  write_json(out, report)
  print(json.dumps({"wrote": str(out), "status": status, "probes": list(probes)}, ensure_ascii=False))
  if status != "ok":
    raise SystemExit(status)


if __name__ == "__main__":
  main()
