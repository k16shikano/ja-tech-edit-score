#!/usr/bin/env python3
"""A で SFT した Qwen アダプタの尤度で、600 対プローブと B 検証 50 件を検定する。

f の代理: Δ(x,y) = log π_adapter(y|x) - log π_adapter(x|x)
補助: log π_adapter(y|x) - log π_base(y|x) から x 基準を引いた差
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

POSITION_RANK = {"a": 0, "eq": 1, "b": 1, "c": 2, "d": 3}


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def load_user_prompts(train_path: Path) -> dict[str, str]:
  out: dict[str, str] = {}
  with train_path.open(encoding="utf-8") as f:
    for line in f:
      obj = json.loads(line)
      meta = obj.get("meta") or {}
      rid = str(meta.get("id") or "")
      if rid:
        out[rid] = obj["messages"][0]["content"]
  return out


def _template_text(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
  try:
    return tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=add_generation_prompt,
      enable_thinking=False,
    )
  except TypeError:
    return tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=add_generation_prompt,
    )


def build_prompt_ids(tokenizer, user_content: str, *, max_input_tokens: int) -> list[int]:
  prompt_text = _template_text(
    tokenizer,
    [{"role": "user", "content": user_content}],
    add_generation_prompt=True,
  )
  ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  if len(ids) > max_input_tokens:
    ids = ids[-max_input_tokens:]
  return ids


def assistant_token_ids(tokenizer, user_content: str, assistant_content: str) -> list[int]:
  full_text = _template_text(
    tokenizer,
    [
      {"role": "user", "content": user_content},
      {"role": "assistant", "content": assistant_content},
    ],
    add_generation_prompt=False,
  )
  prompt_text = _template_text(
    tokenizer,
    [{"role": "user", "content": user_content}],
    add_generation_prompt=True,
  )
  full_ids = tokenizer.encode(full_text, add_special_tokens=False)
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  if len(full_ids) <= len(prompt_ids):
    return []
  if full_ids[: len(prompt_ids)] != prompt_ids:
    # テンプレ差分があるときは末尾長で切る
    return full_ids[-(len(full_ids) - len(prompt_ids)) :]
  return full_ids[len(prompt_ids) :]


def score_texts(
  model,
  tokenizer,
  *,
  user_content: str,
  texts: list[str],
  device,
  max_input_tokens: int,
  use_adapter: bool,
) -> list[float | None]:
  import torch
  from torch.nn import functional as F

  if hasattr(model, "enable_adapter_layers"):
    if use_adapter:
      model.enable_adapter_layers()
    else:
      model.disable_adapter_layers()

  prompt_ids = build_prompt_ids(tokenizer, user_content, max_input_tokens=max_input_tokens)
  seqs: list[list[int]] = []
  ans_lens: list[int] = []
  for text in texts:
    ans_ids = assistant_token_ids(tokenizer, user_content, text)
    if not ans_ids:
      seqs.append([])
      ans_lens.append(0)
      continue
    seqs.append(prompt_ids + ans_ids)
    ans_lens.append(len(ans_ids))

  max_len = max((len(s) for s in seqs if s), default=0)
  if max_len == 0:
    return [None] * len(texts)

  logps: list[float | None] = []
  for seq, n_ans in zip(seqs, ans_lens, strict=True):
    if not seq or n_ans == 0:
      logps.append(None)
      continue
    input_ids = torch.tensor([seq], device=device)
    with torch.no_grad():
      logits = model(input_ids).logits
    # next-token prediction on assistant span
    start = len(seq) - n_ans
    target = input_ids[:, start:]
    pred = logits[:, start - 1 : -1, :]
    lp = F.log_softmax(pred, dim=-1)
    tok_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    logps.append(float(tok_lp.sum().item()))
  return logps


def spearman(xs: list[float], ys: list[float]) -> float | None:
  n = len(xs)
  if n < 2:
    return None
  def rank(v):
    order = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    for ri, i in enumerate(order):
      r[i] = ri + 1
    return r
  rx, ry = rank(xs), rank(ys)
  mx, my = stats.mean(rx), stats.mean(ry)
  num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
  den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
  if den == 0:
    return None
  return num / den


def pearson(xs: list[float], ys: list[float]) -> float | None:
  if len(xs) < 2:
    return None
  mx, my = stats.mean(xs), stats.mean(ys)
  num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
  den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
  if den == 0:
    return None
  return num / den


def summarize_probe(rows: list[dict]) -> dict:
  by_pos: dict[str, list[float]] = defaultdict(list)
  for r in rows:
    if r.get("delta_adapter_y") is not None:
      by_pos[r["position"]].append(r["delta_adapter_y"])

  def gt0(pos: str) -> float:
    xs = by_pos.get(pos, [])
    return sum(x > 0 for x in xs) / len(xs) if xs else float("nan")

  c_rows = [r for r in rows if r["position"] == "c"]
  c_partial = [
    r
    for r in c_rows
    if (r.get("delta_adapter_y") or -999) > 0
    and (r.get("delta_adapter_g") or -999) > (r.get("delta_adapter_y") or 999)
  ]
  a_rows = [r for r in rows if r["position"] == "a"]
  a_neg = sum((r.get("delta_adapter_y") or 0) < 0 for r in a_rows) / max(1, len(a_rows))

  ranks = []
  deltas = []
  lens = []
  delta_lens = []
  for r in rows:
    if r.get("delta_adapter_y") is None:
      continue
    pr = POSITION_RANK.get(r["position"])
    if pr is None:
      continue
    ranks.append(float(pr))
    deltas.append(r["delta_adapter_y"])
    lens.append(r.get("len_y", 0))
    delta_lens.append(r["delta_adapter_y"])

  return {
    "n": len(rows),
    "delta_adapter_y_by_position_median": {
      p: stats.median(by_pos[p]) if by_pos.get(p) else None for p in ["a", "eq", "b", "c", "d"]
    },
    "frac_delta_y_gt_0": {p: gt0(p) for p in ["a", "eq", "b", "c", "d"]},
    "c_partial_order_draft_lt_y_lt_g": len(c_partial) / max(1, len(c_rows)),
    "a_frac_delta_y_lt_0": a_neg,
    "spearman_rank_vs_delta_y": spearman(ranks, deltas),
    "pearson_len_y_vs_delta_y": pearson(lens, delta_lens),
  }


def load_probe_samples(samples_dir: Path) -> dict[tuple[str, str], dict]:
  out: dict[tuple[str, str], dict] = {}
  for mode in ("base", "base_norms", "adapter"):
    path = samples_dir / f"{mode}_samples.jsonl"
    for row in load_jsonl(path):
      key = (str(row["id"]), mode)
      if key not in out or row.get("sample_index", 0) == 0:
        out[key] = row
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--judgments", default="outputs/a1-probe/position_judgments.jsonl")
  parser.add_argument("--samples-dir", default="outputs/a1-probe")
  parser.add_argument("--train-prompts", default="data/edit_sft_hunk_nopara/train.jsonl")
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--adapter", default="outputs/Qwen__Qwen3-8B-pairsplit-v2/adapter")
  parser.add_argument("--out", default="outputs/a1-probe/adapter_likelihood_eval.json")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-input-tokens", type=int, default=3072)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch
  from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer

  judgments = load_jsonl(Path(args.judgments))
  samples = load_probe_samples(Path(args.samples_dir))
  user_by_id = load_user_prompts(Path(args.train_prompts))

  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

  dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
  if device == "cpu":
    dtype = torch.float32

  tok_src = args.adapter if Path(args.adapter, "tokenizer.json").is_file() else args.base_model
  tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else "cpu",
    trust_remote_code=args.trust_remote_code,
    low_cpu_mem_usage=True,
  )
  model = PeftModel.from_pretrained(model, args.adapter)
  model.eval()

  rows_out: list[dict] = []
  missing_prompt = 0
  for j in judgments:
    item_id = str(j["item_id"])
    mode = str(j["mode"])
    sample = samples.get((item_id, mode))
    if not sample:
      continue
    user = user_by_id.get(item_id)
    if not user:
      missing_prompt += 1
      continue
    draft = sample["draft"]
    gold = sample["gold"]
    generated = sample["generated"]
    texts = [generated, draft, gold]

    ad_logps = score_texts(
      model,
      tokenizer,
      user_content=user,
      texts=texts,
      device=next(model.parameters()).device,
      max_input_tokens=args.max_input_tokens,
      use_adapter=True,
    )
    base_logps = score_texts(
      model,
      tokenizer,
      user_content=user,
      texts=texts,
      device=next(model.parameters()).device,
      max_input_tokens=args.max_input_tokens,
      use_adapter=False,
    )
    lp_y_ad, lp_x_ad, lp_g_ad = ad_logps
    lp_y_base, lp_x_base, lp_g_base = base_logps
    if lp_y_ad is None or lp_x_ad is None or lp_g_ad is None:
      continue

    delta_y = lp_y_ad - lp_x_ad
    delta_g = lp_g_ad - lp_x_ad
    ratio_y = (lp_y_ad - (lp_y_base or 0)) - (lp_x_ad - (lp_x_base or 0))

    rows_out.append(
      {
        "pair_id": j["pair_id"],
        "item_id": item_id,
        "mode": mode,
        "position": j["position"],
        "len_y": len(generated),
        "len_x": len(draft),
        "len_g": len(gold),
        "logp_adapter_y": lp_y_ad,
        "logp_adapter_x": lp_x_ad,
        "logp_adapter_g": lp_g_ad,
        "delta_adapter_y": delta_y,
        "delta_adapter_g": delta_g,
        "delta_ratio_y": ratio_y,
      }
    )

  summary = summarize_probe(rows_out)
  summary["missing_user_prompt"] = missing_prompt

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(
    json.dumps({"summary": summary, "rows": rows_out}, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
