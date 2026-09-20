#!/usr/bin/env python3
"""E3 生成（Qwen3 base、§3.1 共通プロンプト）。下書きは評価用データ C の 60 件。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import repo_root
from ead.e_data import E2_E3_EXPECTED, e_data_dir, e_row_path, load_e2_e3_samples
from ead.e_prompt import PROMPT_TAG, build_e_user_content
from generate_edit_sft import (
  decode_generated,
  fit_chat_inputs,
  revision_max_new_tokens,
)


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def done_ids(path: Path) -> set[str]:
  if not path.is_file():
    return set()
  done: set[str] = set()
  with path.open(encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      item_id = str(row.get("item_id") or row.get("id") or "")
      generated = str(row.get("generated") or "").strip()
      if item_id and generated:
        done.add(item_id)
  return done


def append_row(path: Path, row: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=None)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--mode", default="base", choices=["adapter", "base"])
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument(
    "--adapter",
    default="outputs/Qwen__Qwen3-8B-pairsplit-v2/adapter",
  )
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--load-in-4bit", action="store_true", default=True)
  parser.add_argument("--no-4bit", action="store_true")
  parser.add_argument("--max-new-tokens", type=int, default=4096)
  parser.add_argument("--max-input-tokens", type=int, default=5120)
  parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  if args.no_4bit:
    args.load_in_4bit = False

  root = (args.root or repo_root()).resolve()
  e_kind = "E2" if args.mode == "adapter" else "E3"
  out_path = (args.out or e_row_path(e_data_dir(root), e_kind)).resolve()

  if args.mode == "adapter":
    raise SystemExit("E2 is imported from C (adapter_greedy); use build_e_data.py --build-e2-from-c")

  samples = load_e2_e3_samples(root)
  if len(samples) != E2_E3_EXPECTED:
    raise SystemExit(f"expected {E2_E3_EXPECTED} E2/E3 drafts, got {len(samples)}")

  done = done_ids(out_path)
  pending = [s for s in samples if s["item_id"] not in done]
  if args.limit > 0:
    pending = pending[: args.limit]
  if not pending:
    print(f"nothing to do: out={out_path} done={len(done)}")
    return

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    print("WARNING: cuda unavailable; falling back to cpu", file=sys.stderr)
    device = "cpu"

  if args.dtype == "auto":
    if device == "cuda" and torch.cuda.is_bf16_supported():
      dtype = torch.bfloat16
    elif device == "cuda":
      dtype = torch.float16
    else:
      dtype = torch.float32
  else:
    dtype = {
      "float16": torch.float16,
      "bfloat16": torch.bfloat16,
      "float32": torch.float32,
    }[args.dtype]
    if device == "cpu" and dtype != torch.float32:
      dtype = torch.float32

  load_in_4bit = bool(args.load_in_4bit) and device == "cuda"
  tok_src = args.base_model
  adapter_path = (root / args.adapter).resolve()
  if args.mode == "adapter":
    if not adapter_path.is_dir():
      raise SystemExit(f"missing adapter: {adapter_path}")
    has_tok = any((adapter_path / f).is_file() for f in ("tokenizer.json", "tokenizer_config.json"))
    if has_tok:
      tok_src = str(adapter_path)

  print(
    f"e_kind={e_kind} mode={args.mode} pending={len(pending)} done={len(done)} "
    f"out={out_path} 4bit={load_in_4bit} max_input={args.max_input_tokens}",
    flush=True,
  )

  tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model_kwargs: dict = {"trust_remote_code": args.trust_remote_code, "low_cpu_mem_usage": True}
  if load_in_4bit:
    from transformers import BitsAndBytesConfig

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=dtype,
    )
    model_kwargs["device_map"] = "auto"
  else:
    model_kwargs["torch_dtype"] = dtype
    model_kwargs["device_map"] = "cpu" if device == "cpu" else "auto"

  model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
  if args.mode == "adapter":
    from peft import PeftModel

    print(f"loading adapter {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))
  model.eval()

  adapter_ref = str(adapter_path.relative_to(root)) if args.mode == "adapter" else None

  for i, sample in enumerate(pending, start=1):
    draft = sample["draft"]
    user_content = build_e_user_content(draft)
    if user_content != sample["user_content"]:
      raise SystemExit(f"prompt mismatch for {sample['item_id']}")

    inputs, n_in = fit_chat_inputs(
      tokenizer,
      user_content,
      max_input_tokens=args.max_input_tokens,
      enable_thinking=False,
    )
    first_param = next(model.parameters())
    inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
    max_new = revision_max_new_tokens(
      tokenizer,
      draft,
      mode=args.mode,
      cap=args.max_new_tokens,
    )
    print(
      f"[{i}/{len(pending)}] {sample['item_id']} "
      f"draft_chars={len(draft)} input_tokens={n_in} max_new={max_new}",
      flush=True,
    )
    with torch.no_grad():
      out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        do_sample=False,
      )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    text = decode_generated(tokenizer, gen_ids)
    row = {
      "item_id": sample["item_id"],
      "project_id": sample["project_id"],
      "draft": draft,
      "generated": text,
      "e_kind": e_kind,
      "prompt_tag": PROMPT_TAG,
      "generator": args.base_model,
      "adapter": adapter_ref,
      "generation": "greedy",
      "input_tokens": n_in,
      "max_new_tokens": max_new,
      "enable_thinking": False,
      "created_at": utc_now_iso(),
    }
    append_row(out_path, row)
    if device == "cuda":
      del out_ids, inputs
      torch.cuda.empty_cache()

  print(f"done: wrote/updated {out_path}", flush=True)


if __name__ == "__main__":
  os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
  main()
