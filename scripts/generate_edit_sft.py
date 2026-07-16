#!/usr/bin/env python3
"""編集モデル評価用の推敲生成（系統1）。

mode:
  adapter    : ベース＋LoRA。SFT と同じ短い指示
  base       : ベースのみ。同じ短い指示（LoRA 効果の切り分け）
  base_norms : ベースのみ。規範全文を前置した指示（規範スキル対照）

CPU / CUDA 両対応。原稿本文を入出力に含むので公開配布しないこと。
V100 32GB では既定で 4bit 量子化読み込みを使う（規範前置の長文脈で OOM しやすいため）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def load_heldout(
  path: Path,
  *,
  limit: int,
  min_chars: int,
  max_chars: int,
) -> list[dict]:
  by_project: dict[str, list[dict]] = {}
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      user = obj["messages"][0]["content"]
      draft = user.split("\n\n", 1)[-1]
      if len(draft) < min_chars:
        continue
      if max_chars > 0 and len(draft) > max_chars:
        continue
      meta = obj.get("meta") or {}
      pid = str(meta.get("project_id") or "(unknown)")
      row = {
        "id": meta.get("id", ""),
        "project_id": pid,
        "user_sft": user,
        "draft": draft,
        "gold": obj["messages"][1]["content"],
      }
      by_project.setdefault(pid, []).append(row)

  if limit <= 0:
    out: list[dict] = []
    for pid in sorted(by_project):
      out.extend(by_project[pid])
    return out

  projects = sorted(by_project)
  rows: list[dict] = []
  depth = 0
  while len(rows) < limit:
    added = False
    for pid in projects:
      bucket = by_project[pid]
      if depth < len(bucket):
        rows.append(bucket[depth])
        added = True
        if len(rows) >= limit:
          break
    if not added:
      break
    depth += 1
  return rows


def build_user_content(sample: dict, *, mode: str, norms_text: str) -> str:
  if mode in ("adapter", "base"):
    return sample["user_sft"]
  if mode == "base_norms":
    # 規範対照は「推敲後本文のみ」を厳密に要求する。
    # 前置き・解説・膨張があると BT 比較の対照条件として使えない。
    return (
      "あなたは日本語技術文書の編集者である。役割は下書きを推敲することだけだ。\n\n"
      "【文章規範】\n"
      f"{norms_text}\n\n"
      "【作業】\n"
      "次の下書きを、上記の規範に沿って意味を保ったまま推敲する。\n\n"
      "【出力規則（厳守）】\n"
      "- 出力は推敲後の本文のみ。前置き・後書き・挨拶は禁止。\n"
      "- 「以下は推敲です」「推敲後の文」「解説」「根拠」「変更点」などのメタ文言は禁止。\n"
      "- 箇条書きの変更理由や、規範の復唱は禁止。\n"
      "- 下書きに無い話題・定義・段落を新たに足さない。\n"
      "- 見出しや短い下書きを、本文の解説へ膨らませない。"
      " 長さは下書きと同程度（目安として 0.7〜1.5 倍）に保つ。\n"
      "- Markdown の区切り線（---）で本文と解説を分けない。解説自体を出さない。\n\n"
      "【下書き】\n"
      f"{sample['draft']}\n\n"
      "【出力】推敲後の本文のみ："
    )
  raise SystemExit(f"unknown mode: {mode}")


def strip_thinking(text: str) -> str:
  """Qwen3 の思考ブロックを除去する。閉じタグ無しの途中出力にも対応。"""
  if not text:
    return text
  # 完了したブロック
  text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE)
  # 閉じられていないブロック（先頭から残り全部）
  text = re.sub(r"<think>[\s\S]*\Z", "", text, flags=re.IGNORECASE)
  # タグだけ残った場合
  text = text.replace("<think>", "").replace("</think>", "")
  return text.strip()


def strip_meta_revision(text: str) -> str:
  """プロンプト漏れのメタ前置き・解説を落とす（保険）。主対策はプロンプト側。"""
  if not text:
    return text
  t = text.strip()

  # 「**推敲後…**」見出しの直後から取る
  m = re.search(
    r"\*\*推敲後[^*]*\*\*[：:]*\s*\n+(.*)",
    t,
    flags=re.DOTALL,
  )
  if m:
    t = m.group(1).strip()

  # 先頭の「以下は…推敲…」前置きを --- または空行区切りまで落とす
  if re.match(r"以下は[^\n]{0,200}推敲", t):
    parts = re.split(r"\n---\n+", t, maxsplit=1)
    if len(parts) == 2:
      t = parts[1].strip()
    else:
      # 空行2つで本文が始まる場合
      parts = re.split(r"\n\s*\n", t, maxsplit=1)
      if len(parts) == 2:
        t = parts[1].strip()

  # 末尾の解説ブロックを切る
  cut_markers = (
    r"\n---\s*\n+(?:\#\#?\#?\s*)?(?:説明|解説|根拠|変更点|推敲の根拠)",
    r"\n\*\*(?:説明|解説|根拠|変更点|推敲の根拠)[^*]*\*\*",
    r"\n(?:\#\#?\#?\s*)?(?:説明|解説|根拠|変更点)[：:]",
  )
  for pat in cut_markers:
    m = re.search(pat, t)
    if m:
      t = t[: m.start()].strip()

  # 残った単独の「推敲後の文：」行
  t = re.sub(r"^\*\*推敲後[^*]*\*\*[：:]*\s*\n*", "", t)
  t = re.sub(r"^推敲後の(?:文|本文)[：:]\s*\n*", "", t)
  t = re.sub(r"(?:\n\s*)?---\s*$", "", t)
  return t.strip()


def decode_generated(tokenizer, gen_ids) -> str:
  """生成トークン列を本文だけにデコードする。

  Qwen3 の公式手順に合わせ、最後の ``</think>``（token 151668）以降を本文とする。
  トークン分割後も文字列側で残った think 断片を除去する。
  """
  ids = gen_ids.tolist() if hasattr(gen_ids, "tolist") else list(gen_ids)
  end_id = tokenizer.convert_tokens_to_ids("</think>")
  if end_id is None or end_id == getattr(tokenizer, "unk_token_id", None):
    end_id = 151668
  try:
    cut = len(ids) - ids[::-1].index(end_id)
  except ValueError:
    cut = 0
    start_id = tokenizer.convert_tokens_to_ids("<think>")
    if (
      start_id is not None
      and start_id != getattr(tokenizer, "unk_token_id", None)
      and ids
      and ids[0] == start_id
    ):
      # 思考が閉じずに終わった場合は本文無しとして捨てる
      return ""
  text = tokenizer.decode(ids[cut:], skip_special_tokens=True)
  return strip_meta_revision(strip_thinking(text))


def fit_chat_inputs(
  tokenizer,
  user_content: str,
  *,
  max_input_tokens: int,
  enable_thinking: bool,
):
  """プロンプトが長すぎるときは draft 側（末尾）を残して先頭を切る。

  base_norms では規範が先頭・下書きが末尾なので、truncation_side=left で
  下書きと生成プロンプトを優先する。
  """
  messages = [{"role": "user", "content": user_content}]
  template_kwargs = {
    "tokenize": False,
    "add_generation_prompt": True,
  }
  # Qwen3: enable_thinking=False で空の think ブロックを挿入し、思考モードを止める
  try:
    prompt = tokenizer.apply_chat_template(
      messages,
      enable_thinking=enable_thinking,
      **template_kwargs,
    )
  except TypeError:
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
  prev = getattr(tokenizer, "truncation_side", "right")
  tokenizer.truncation_side = "left"
  try:
    encoded = tokenizer(
      prompt,
      return_tensors="pt",
      truncation=True,
      max_length=max_input_tokens,
    )
  finally:
    tokenizer.truncation_side = prev
  return encoded, int(encoded["input_ids"].shape[1])


def revision_max_new_tokens(tokenizer, draft: str, *, mode: str, cap: int) -> int:
  """base_norms では下書き長に応じて生成上限を絞り、不当な膨張を抑える。"""
  if mode != "base_norms":
    return cap
  n = len(tokenizer.encode(draft, add_special_tokens=False))
  return max(32, min(cap, int(n * 1.5) + 24))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--heldout", default="data/edit_sft/heldout.jsonl")
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--adapter", default="outputs/edit-sft/Qwen__Qwen3-8B/adapter")
  parser.add_argument(
    "--mode",
    default="adapter",
    choices=["adapter", "base", "base_norms"],
  )
  parser.add_argument("--norms-file", default="data/tech-writing-norms.md")
  parser.add_argument("--out", default="")
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  parser.add_argument("--limit", type=int, default=0, help="0 = all matching samples")
  parser.add_argument("--min-chars", type=int, default=1)
  parser.add_argument("--max-chars", type=int, default=0, help="0 = no upper bound")
  parser.add_argument("--max-new-tokens", type=int, default=512)
  parser.add_argument(
    "--max-input-tokens",
    type=int,
    default=3072,
    help="プロンプト上限（生成トークン分の余裕を残す）",
  )
  parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
  parser.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes 4bit で読む")
  parser.add_argument(
    "--enable-thinking",
    action="store_true",
    help="Qwen3 の思考モードを有効化（既定は無効）",
  )
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  norms_text = ""
  if args.mode == "base_norms":
    norms_path = Path(args.norms_file)
    if not norms_path.is_file():
      raise SystemExit(f"missing norms file: {norms_path}")
    norms_text = norms_path.read_text(encoding="utf-8").strip()

  samples = load_heldout(
    Path(args.heldout),
    limit=args.limit,
    min_chars=args.min_chars,
    max_chars=args.max_chars,
  )
  if not samples:
    raise SystemExit("no heldout samples matched length filters")

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
      print(f"NOTE: cpu + {args.dtype} → float32", flush=True)
      dtype = torch.float32

  load_in_4bit = bool(args.load_in_4bit) and device == "cuda"
  tok_src = args.adapter if (args.mode == "adapter" and Path(args.adapter).is_dir()) else args.base_model
  print(
    f"mode={args.mode} model={args.base_model} device={device} "
    f"dtype={dtype} 4bit={load_in_4bit} n={len(samples)} "
    f"max_input={args.max_input_tokens} max_new={args.max_new_tokens} "
    f"enable_thinking={bool(args.enable_thinking)}",
    flush=True,
  )
  tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
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
    print(f"loading adapter {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
  model.eval()

  out_path = Path(args.out) if args.out else Path(f"outputs/edit-sft-eval/{args.mode}.jsonl")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as fout:
    for i, sample in enumerate(samples, start=1):
      user_content = build_user_content(sample, mode=args.mode, norms_text=norms_text)
      inputs, n_in = fit_chat_inputs(
        tokenizer,
        user_content,
        max_input_tokens=args.max_input_tokens,
        enable_thinking=bool(args.enable_thinking),
      )
      first_param = next(model.parameters())
      inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
      max_new = revision_max_new_tokens(
        tokenizer,
        sample["draft"],
        mode=args.mode,
        cap=args.max_new_tokens,
      )
      print(
        f"[{i}/{len(samples)}] {sample['project_id']} "
        f"draft_chars={len(sample['draft'])} input_tokens={n_in} "
        f"max_new={max_new} thinking={bool(args.enable_thinking)}",
        flush=True,
      )
      with torch.no_grad():
        out_ids = model.generate(
          **inputs,
          max_new_tokens=max_new,
          do_sample=False,
          pad_token_id=tokenizer.pad_token_id,
          use_cache=True,
        )
      gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
      text = decode_generated(tokenizer, gen_ids)
      row = {
        "id": sample["id"],
        "project_id": sample["project_id"],
        "mode": args.mode,
        "draft": sample["draft"],
        "gold": sample["gold"],
        "generated": text,
        "input_tokens": n_in,
      }
      fout.write(json.dumps(row, ensure_ascii=False) + "\n")
      fout.flush()
      if device == "cuda":
        del out_ids, inputs
        torch.cuda.empty_cache()

  print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
  # 断片化緩和（親プロセスの環境でも可）
  os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
  main()
