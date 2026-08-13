#!/usr/bin/env python3
"""推敲対照ペアの層活性を抽出する。

フェーズ A（読み取り）。原稿本文は成果物に保存しない。
出力は activations.npz（draft/revised の活性）と meta.json。

prompt-mode:
  none    : 生テキストを流して全トークン平均（初回フェーズ A と同じ）
  reading : 「推敲済みか考えよ」プロンプトに包み、最終トークン活性を取る
  norms   : 文章規範全文を前置した判定プロンプトで、最終トークン活性を取る

reading / norms では、全サンプル共通の前置部分（指示＋規範）の KV キャッシュを
一度だけ計算して使い回す。draft と revised で前置は同一なので、対照差分では
前置の寄与は打ち消される。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from steering_utils import load_revision_pairs, model_slug

PROMPTS = {
  "reading": {
    "prefix": (
      "次の文章が、日本語の技術文書として推敲済みかどうかを考えよ。\n\n文章：\n"
    ),
    "suffix": "\n\nこの文章は推敲済みか。",
  },
  "norms": {
    "prefix": (
      "以下は日本語技術文書の文章規範である。\n\n{norms}\n\n"
      "次の文章が、この規範に沿って推敲済みかどうかを考えよ。\n\n文章：\n"
    ),
    "suffix": "\n\nこの文章は規範に沿って推敲済みか。",
  },
}


def require_transformers():
  try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
  except ImportError as exc:  # pragma: no cover
    raise SystemExit(
      "torch / transformers が必要です。pip install -r requirements-steering.txt"
    ) from exc
  return torch, AutoModelForCausalLM, AutoTokenizer


def stratified_limit(pairs: list[dict], limit: int) -> list[dict]:
  """書籍（project_id）横断のラウンドロビンで limit 件を選ぶ。"""
  if limit <= 0 or limit >= len(pairs):
    return pairs
  by_project: dict[str, list[dict]] = defaultdict(list)
  for p in pairs:
    by_project[p["project_id"]].append(p)
  order = sorted(by_project)
  out: list[dict] = []
  depth = 0
  while len(out) < limit:
    added = False
    for pid in order:
      bucket = by_project[pid]
      if depth < len(bucket):
        out.append(bucket[depth])
        added = True
        if len(out) >= limit:
          break
    if not added:
      break
    depth += 1
  return out


def mean_pool_hidden(hidden, attention_mask):
  """(batch, seq, dim) をマスク付き平均。"""
  mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
  summed = (hidden * mask).sum(dim=1)
  denom = mask.sum(dim=1).clamp(min=1e-6)
  return summed / denom


def encode_batch_mean(model, tokenizer, texts, *, device, max_length, torch):
  """prompt-mode=none: 全トークン平均。(batch, n_layers_incl_embed, dim)"""
  encoded = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=max_length,
  )
  encoded = {k: v.to(device) for k, v in encoded.items()}
  with torch.no_grad():
    out = model(**encoded, output_hidden_states=True, use_cache=False)
  pooled = []
  for h in out.hidden_states:
    pooled.append(mean_pool_hidden(h, encoded["attention_mask"]).float().cpu())
  return torch.stack(pooled, dim=1).numpy()


class PromptedEncoder:
  """共通前置の KV キャッシュを使い回し、最終トークン活性を取る。"""

  def __init__(self, model, tokenizer, *, prefix: str, suffix: str, device, max_length, torch):
    self.model = model
    self.tokenizer = tokenizer
    self.device = device
    self.max_length = max_length
    self.torch = torch
    self.suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"]
    self.prefix_ids = prefix_ids.to(device)
    self.prefix_len = int(prefix_ids.shape[1])

    with torch.no_grad():
      out = model(input_ids=self.prefix_ids, use_cache=True)
    cache = out.past_key_values
    if hasattr(cache, "crop"):
      self.cache = cache
      self.reusable = True
    else:  # 古い transformers 向けフォールバック（毎回前置を計算し直す）
      self.cache = None
      self.reusable = False

  def encode(self, text: str) -> np.ndarray:
    """1 テキスト分。(n_layers_incl_embed, dim)"""
    torch = self.torch
    body_ids = self.tokenizer(
      text,
      add_special_tokens=False,
      truncation=True,
      max_length=self.max_length,
    )["input_ids"]
    new_ids = torch.tensor([body_ids + self.suffix_ids], device=self.device)

    if self.reusable:
      attn = torch.ones(
        (1, self.prefix_len + new_ids.shape[1]), dtype=torch.long, device=self.device
      )
      with torch.no_grad():
        out = self.model(
          input_ids=new_ids,
          attention_mask=attn,
          past_key_values=self.cache,
          use_cache=True,
          output_hidden_states=True,
        )
      # 使い回すため、追記された本文・接尾辞ぶんを切り戻す
      self.cache.crop(self.prefix_len)
    else:
      full_ids = torch.cat([self.prefix_ids, new_ids], dim=1)
      with torch.no_grad():
        out = self.model(
          input_ids=full_ids,
          use_cache=False,
          output_hidden_states=True,
        )

    pooled = [h[0, -1, :].float().cpu() for h in out.hidden_states]
    return torch.stack(pooled, dim=0).numpy()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/revision_pairs.jsonl", help="revision pairs JSONL")
  parser.add_argument("--model", required=True, help="Hugging Face model id")
  parser.add_argument("--out-dir", default="", help="output dir (default: outputs/steering/<slug>[--mode])")
  parser.add_argument("--device", default="cuda", help="cuda / cpu / mps")
  parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--max-length", type=int, default=2048)
  parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="max pairs (0 = all)。書籍横断のラウンドロビンで選ぶ",
  )
  parser.add_argument(
    "--prompt-mode",
    default="none",
    choices=["none", "reading", "norms"],
    help="none=生テキスト平均 / reading=判定プロンプト最終トークン / norms=規範前置",
  )
  parser.add_argument(
    "--norms-file",
    default="data/tech-writing-norms.md",
    help="prompt-mode=norms で前置する規範ファイル",
  )
  parser.add_argument(
    "--trust-remote-code",
    action="store_true",
    help="pass trust_remote_code=True to from_pretrained",
  )
  args = parser.parse_args()

  torch, AutoModelForCausalLM, AutoTokenizer = require_transformers()

  pairs_path = Path(args.pairs)
  if not pairs_path.is_file():
    raise SystemExit(
      f"missing pairs: {pairs_path}\n先に make steering-pairs を実行してください。"
    )

  pairs = load_revision_pairs(pairs_path)
  pairs = stratified_limit(pairs, args.limit)
  if not pairs:
    raise SystemExit("no pairs")

  norms_text = ""
  norms_sha = ""
  if args.prompt_mode == "norms":
    norms_path = Path(args.norms_file)
    if not norms_path.is_file():
      raise SystemExit(f"missing norms file: {norms_path}")
    norms_text = norms_path.read_text(encoding="utf-8").strip()
    norms_sha = hashlib.sha256(norms_text.encode("utf-8")).hexdigest()[:16]

  slug = model_slug(args.model)
  dir_name = slug if args.prompt_mode == "none" else f"{slug}--{args.prompt_mode}"
  out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/steering") / dir_name
  out_dir.mkdir(parents=True, exist_ok=True)

  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    print("WARNING: cuda requested but unavailable; falling back to cpu", file=sys.stderr)
    device = "cpu"

  dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
  }
  torch_dtype = None if args.dtype == "auto" else dtype_map[args.dtype]

  print(f"loading model: {args.model} device={device}", flush=True)
  tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=args.trust_remote_code,
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model_kwargs = {"trust_remote_code": args.trust_remote_code}
  if torch_dtype is not None:
    model_kwargs["torch_dtype"] = torch_dtype
  elif device == "cuda":
    model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

  model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
  model.to(device)
  model.eval()

  n = len(pairs)
  draft_chunks: list[np.ndarray] = []
  revised_chunks: list[np.ndarray] = []

  if args.prompt_mode == "none":
    for start in range(0, n, args.batch_size):
      batch = pairs[start : start + args.batch_size]
      draft_chunks.append(
        encode_batch_mean(
          model,
          tokenizer,
          [p["draft"] for p in batch],
          device=device,
          max_length=args.max_length,
          torch=torch,
        )
      )
      revised_chunks.append(
        encode_batch_mean(
          model,
          tokenizer,
          [p["revised"] for p in batch],
          device=device,
          max_length=args.max_length,
          torch=torch,
        )
      )
      done = min(start + args.batch_size, n)
      print(f"encoded {done}/{n}", flush=True)
  else:
    spec = PROMPTS[args.prompt_mode]
    prefix = spec["prefix"].format(norms=norms_text) if args.prompt_mode == "norms" else spec["prefix"]
    encoder = PromptedEncoder(
      model,
      tokenizer,
      prefix=prefix,
      suffix=spec["suffix"],
      device=device,
      max_length=args.max_length,
      torch=torch,
    )
    print(
      f"prompt-mode={args.prompt_mode} prefix_tokens={encoder.prefix_len} "
      f"kv_reuse={encoder.reusable}",
      flush=True,
    )
    for i, p in enumerate(pairs, start=1):
      draft_chunks.append(encoder.encode(p["draft"])[None, :, :])
      revised_chunks.append(encoder.encode(p["revised"])[None, :, :])
      if i % 16 == 0 or i == n:
        print(f"encoded {i}/{n}", flush=True)

  draft_act = np.concatenate(draft_chunks, axis=0).astype(np.float16)
  revised_act = np.concatenate(revised_chunks, axis=0).astype(np.float16)
  n_layers = int(draft_act.shape[1])
  hidden = int(draft_act.shape[2])

  act_path = out_dir / "activations.npz"
  np.savez_compressed(
    act_path,
    draft=draft_act,
    revised=revised_act,
    project_id=np.asarray([p["project_id"] for p in pairs], dtype=object),
    pair_id=np.asarray([p["id"] for p in pairs], dtype=object),
  )

  meta = {
    "model": args.model,
    "slug": slug,
    "prompt_mode": args.prompt_mode,
    "pooling": "mean" if args.prompt_mode == "none" else "last_token",
    "norms_sha256_16": norms_sha,
    "n_pairs": n,
    "n_layers_including_embed": n_layers,
    "hidden_size": hidden,
    "device": device,
    "dtype_saved": "float16",
    "max_length": args.max_length,
    "batch_size": args.batch_size,
    "projects": sorted({p["project_id"] for p in pairs}),
    "note": "hidden_states[0]=embed, [1:]=transformer layers",
  }
  meta_path = out_dir / "meta.json"
  meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  print(f"wrote {act_path}")
  print(f"wrote {meta_path}")
  print(f"shape draft/revised: {draft_act.shape}")


if __name__ == "__main__":
  main()
