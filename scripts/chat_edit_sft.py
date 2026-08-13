#!/usr/bin/env python3
"""編集モデルを対話で試す（素のベース / LoRA）。

対話検証用。バッチ評価（generate_edit_sft）とは次が違う。

- 既定はサンプリング（温度付き）。同じ下書きでも回ごとに違う案が出る。
- 指示文は学習・バッチ生成と同一（export_edit_sft.INSTRUCTION）。

原稿本文は端末に出るだけなので、ログや成果物に残さないこと。

例:
  PYTHONPATH=scripts python scripts/chat_edit_sft.py --mode base --load-in-4bit
  PYTHONPATH=scripts python scripts/chat_edit_sft.py --mode adapter --load-in-4bit
  # バッチ評価と同じ貪欲デコードに戻すとき
  PYTHONPATH=scripts python scripts/chat_edit_sft.py --mode adapter --greedy --load-in-4bit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_edit_sft import INSTRUCTION  # noqa: E402

CHAT_INSTRUCTION = INSTRUCTION


def read_draft() -> str | None:
  """空行だけの行で入力終了。/quit で終了、/clear は無視（次の下書きへ）。"""
  print(
    "下書きを貼る（終了は空行のみの行。終了は /quit）:",
    flush=True,
  )
  lines: list[str] = []
  while True:
    try:
      line = input()
    except EOFError:
      return None
    if line.strip() == "/quit":
      return None
    if line.strip() == "/clear":
      lines.clear()
      print("(入力クリア)", flush=True)
      continue
    if line == "" and lines:
      break
    if line == "" and not lines:
      continue
    lines.append(line)
  return "\n".join(lines).strip() or None


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--adapter", default="/app/adapter")
  parser.add_argument("--mode", choices=["base", "adapter"], default="adapter")
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  parser.add_argument("--load-in-4bit", action="store_true")
  parser.add_argument("--max-new-tokens", type=int, default=1024)
  parser.add_argument("--max-input-tokens", type=int, default=3072)
  parser.add_argument("--trust-remote-code", action="store_true", default=True)
  parser.add_argument(
    "--enable-thinking",
    action="store_true",
    help="Qwen3 思考モード（既定は無効）",
  )
  parser.add_argument(
    "--greedy",
    action="store_true",
    help="貪欲デコード（バッチ評価と同じ。対話検証では使わない）",
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="サンプリング温度（--greedy 時は無視）",
  )
  parser.add_argument(
    "--top-p",
    type=float,
    default=0.9,
    help="nucleus sampling（--greedy 時は無視）",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="乱数シード（再現したいときだけ）",
  )
  args = parser.parse_args()

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

  root = Path(__file__).resolve().parent
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  from generate_edit_sft import decode_generated, fit_chat_inputs

  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    print("WARNING: cuda unavailable; falling back to cpu", file=sys.stderr)
    device = "cpu"

  if device == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
  elif device == "cuda":
    dtype = torch.float16
  else:
    dtype = torch.float32

  load_in_4bit = bool(args.load_in_4bit) and device == "cuda"
  tok_src = (
    args.adapter
    if args.mode == "adapter" and Path(args.adapter).is_dir()
    else args.base_model
  )
  instruction = CHAT_INSTRUCTION
  decode_desc = (
    "greedy"
    if args.greedy
    else f"sample T={args.temperature} top_p={args.top_p}"
  )
  print(
    f"loading mode={args.mode} model={args.base_model} "
    f"device={device} 4bit={load_in_4bit} decode={decode_desc} …",
    flush=True,
  )
  if args.seed is not None:
    set_seed(args.seed)

  tokenizer = AutoTokenizer.from_pretrained(
    tok_src, trust_remote_code=args.trust_remote_code
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model_kwargs: dict = {
    "trust_remote_code": args.trust_remote_code,
    "low_cpu_mem_usage": True,
  }
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

    if not Path(args.adapter).is_dir():
      raise SystemExit(f"missing adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
  model.eval()
  print(
    "ready. 同じ下書きを base / adapter で比べる。"
    " 既定はサンプリングなので、同じ下書きを再度貼ると別案が出る。",
    flush=True,
  )

  while True:
    draft = read_draft()
    if draft is None:
      print("bye", flush=True)
      break
    user = f"{instruction}\n\n{draft}"
    inputs, n_in = fit_chat_inputs(
      tokenizer,
      user,
      max_input_tokens=args.max_input_tokens,
      enable_thinking=bool(args.enable_thinking),
    )
    first_param = next(model.parameters())
    inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
    max_new = args.max_new_tokens
    print(f"(input_tokens≈{n_in}, generating…)", flush=True)
    gen_kwargs: dict = {
      "max_new_tokens": max_new,
      "pad_token_id": tokenizer.pad_token_id,
      "use_cache": True,
    }
    if args.greedy:
      gen_kwargs["do_sample"] = False
    else:
      gen_kwargs["do_sample"] = True
      gen_kwargs["temperature"] = float(args.temperature)
      gen_kwargs["top_p"] = float(args.top_p)
    with torch.inference_mode():
      out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[-1] :]
    text = decode_generated(tokenizer, gen)
    print("----- 推敲 -----", flush=True)
    print(text, flush=True)
    print("---------------", flush=True)


if __name__ == "__main__":
  main()
