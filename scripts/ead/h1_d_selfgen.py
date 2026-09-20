#!/usr/bin/env python3
"""R5-B: D 200 下書きの自己生成テスト（R6 典型性警告用）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from ead.common import ead_out, ead_reports, ead_work, fmt_pct, load_jsonl, md_table, repo_root, write_json
from ead.den_score import load_density_model, score_candidate
from generate_edit_sft import decode_generated, fit_chat_inputs, revision_max_new_tokens
from section_middle_utils import build_revision_prompt

try:
  import torch
except ImportError as exc:
  raise SystemExit(f"torch required: {exc}") from exc


def load_d_items(root: Path) -> list[dict]:
  rows = load_jsonl(root / "data/d/train.jsonl") + load_jsonl(root / "data/d/valid.jsonl")
  by_id: dict[str, dict] = {}
  for row in rows:
    item_id = str(row["item_id"])
    if row.get("y_kind") == "human":
      by_id[item_id] = {
        "item_id": item_id,
        "draft": str(row["draft"]),
        "human": str(row["y"]),
      }
  if len(by_id) != 200:
    raise SystemExit(f"expected 200 D drafts, got {len(by_id)}")
  return [by_id[k] for k in sorted(by_id)]


def generate_one(
  model,
  tokenizer,
  *,
  draft: str,
  device,
  max_input_tokens: int,
  max_new_cap: int,
  seed: int,
) -> str:
  user_content = build_revision_prompt(draft)
  inputs, _ = fit_chat_inputs(
    tokenizer,
    user_content,
    max_input_tokens=max_input_tokens,
    enable_thinking=False,
  )
  first_param = next(model.parameters())
  inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
  max_new = revision_max_new_tokens(tokenizer, draft, mode="adapter", cap=max_new_cap)
  torch.manual_seed(seed)
  if device == "cuda":
    torch.cuda.manual_seed_all(seed)
  with torch.no_grad():
    out_ids = model.generate(
      **inputs,
      max_new_tokens=max_new,
      pad_token_id=tokenizer.pad_token_id,
      do_sample=False,
      use_cache=True,
    )
  gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
  return decode_generated(tokenizer, gen_ids)


def load_or_generate(
  items: list[dict],
  *,
  model,
  tokenizer,
  device: str,
  cache_path: Path,
  refresh: bool,
  max_input_tokens: int,
  max_new_cap: int,
  seed: int,
) -> list[dict]:
  if cache_path.is_file() and not refresh:
    cached = load_jsonl(cache_path)
    by_id = {str(r["item_id"]): r for r in cached}
    if len(by_id) == len(items):
      return [by_id[str(it["item_id"])] for it in items]

  out: list[dict] = []
  for i, item in enumerate(items, start=1):
    generated = generate_one(
      model,
      tokenizer,
      draft=item["draft"],
      device=device,
      max_input_tokens=max_input_tokens,
      max_new_cap=max_new_cap,
      seed=seed + i,
    )
    out.append({**item, "generated": generated})
    print(f"[{i}/{len(items)}] generated_chars={len(generated)}", flush=True)
    if device == "cuda":
      torch.cuda.empty_cache()

  cache_path.parent.mkdir(parents=True, exist_ok=True)
  with cache_path.open("w", encoding="utf-8") as f:
    for row in out:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  return out


def score_pairs(
  rows: list[dict],
  model,
  tokenizer,
  device,
  *,
  max_seq_len: int,
) -> list[dict]:
  scored: list[dict] = []
  for row in rows:
    draft = row["draft"]
    user = build_revision_prompt(draft)
    for source, cand in (("human", row["human"]), ("selfgen", row["generated"])):
      metrics = score_candidate(
        model,
        tokenizer,
        user_content=user,
        draft=draft,
        candidate=cand,
        device=device,
        max_seq_len=max_seq_len,
      )
      if metrics is None:
        raise SystemExit(f"score failed item_id={row['item_id']} source={source}")
      scored.append({"item_id": row["item_id"], "source": source, **metrics})
  return scored


def summarize(rows: list[dict], scored: list[dict]) -> dict[str, Any]:
  lookup: dict[tuple[str, str], float] = {}
  for row in scored:
    lookup[(str(row["item_id"]), str(row["source"]))] = float(row["s_sum"])

  human_wins = 0
  selfgen_wins = 0
  ties = 0
  per_item: list[dict] = []
  for row in rows:
    item_id = str(row["item_id"])
    sh = lookup[(item_id, "human")]
    sg = lookup[(item_id, "selfgen")]
    if sh > sg:
      human_wins += 1
      winner = "human"
    elif sg > sh:
      selfgen_wins += 1
      winner = "selfgen"
    else:
      ties += 1
      winner = "tie"
    per_item.append(
      {
        "item_id": item_id,
        "s_sum_human": sh,
        "s_sum_selfgen": sg,
        "winner": winner,
      }
    )

  n = len(rows)
  return {
    "n": n,
    "human_win_rate": human_wins / n,
    "selfgen_win_rate": selfgen_wins / n,
    "tie_rate": ties / n,
    "human_wins": human_wins,
    "selfgen_wins": selfgen_wins,
    "ties": ties,
    "per_item": per_item,
  }


def build_report(summary: dict[str, Any]) -> str:
  hr = summary["human_win_rate"]
  lines = [
    "# ead-h1-d-selfgen",
    "",
    "R5-B: D 200 下書きについて `ead-den-a-excl` で 1 本生成し、人間の推敲と $s_\\mathrm{sum}$ を比較。",
    "目的は R6（典型性崩壊）の**警告**のみ。人間側の勝率が極端に低い場合、C でも同族分布の交絡が起きうる。",
    "",
    md_table(
      ["指標", "値"],
      [
        ["件数", str(summary["n"])],
        ["人間側が $s_\\mathrm{sum}$ で勝つ割合", fmt_pct(hr)],
        ["自己生成側が勝つ割合", fmt_pct(summary["selfgen_win_rate"])],
        ["同点", fmt_pct(summary["tie_rate"])],
      ],
    ),
    "",
    "## 解釈",
    "",
  ]
  if hr < 0.35:
    lines.append(
      f"- 人間側の勝率 {hr:.3f} は低い。**R6 の典型性崩壊の警告**として記録する。C でも同族分布の交絡が起きうる。"
    )
  elif hr > 0.65:
    lines.append(
      f"- 人間側の勝率 {hr:.3f} は高い。ただしどちらが実際に良いかの人手判定はないため、**H1 の証明にはならない**。"
    )
  else:
    lines.append(
      f"- 人間側の勝率 {hr:.3f}。中間域のため R6 警告は確定しない。H1 の証明にもならない（人手の良し悪し判定がない）。"
    )
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--adapter-dir", type=Path, default=None)
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-input-tokens", type=int, default=4096)
  parser.add_argument("--max-new-tokens", type=int, default=2048)
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--refresh-generations", action="store_true")
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  root = args.root.resolve()
  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

  items = load_d_items(root)
  adapter_dir = args.adapter_dir or (ead_out() / "adapters" / "ead-den-a-excl")
  if (adapter_dir / "adapter").is_dir():
    adapter_dir = adapter_dir / "adapter"

  model, tokenizer, dev = load_density_model(
    adapter_dir, args.base_model, device, trust_remote_code=args.trust_remote_code
  )
  cache_path = ead_work() / "h1_d_selfgen_samples.jsonl"
  rows = load_or_generate(
    items,
    model=model,
    tokenizer=tokenizer,
    device=device,
    cache_path=cache_path,
    refresh=args.refresh_generations,
    max_input_tokens=args.max_input_tokens,
    max_new_cap=args.max_new_tokens,
    seed=args.seed,
  )
  scored = score_pairs(rows, model, tokenizer, dev, max_seq_len=args.max_seq_length)
  summary = summarize(rows, scored)

  reports = ead_reports()
  write_json(reports / "ead-h1-d-selfgen.json", summary)
  (reports / "ead-h1-d-selfgen.md").write_text(build_report(summary), encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-h1-d-selfgen.md"), "human_win_rate": summary["human_win_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
  main()
