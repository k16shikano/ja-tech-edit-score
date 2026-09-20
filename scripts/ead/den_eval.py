#!/usr/bin/env python3
"""Phase 2: B valid 上の密度モデル評価。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import human_pref, merge_rows
from ead.common import (
  ead_out,
  ead_reports,
  ead_work,
  fmt_rate,
  load_jsonl,
  md_table,
  rate_summary,
  repo_root,
  write_json,
)
from eval_pref_valid50_gold_vs_composer import agreement_for_pairs, human_picked_source
from eval_pref_multigranular_blind60 import scorer_pref_from_scores

from ead.den_score import (
  apply_resid,
  fit_resid_model,
  load_density_model,
  score_fit_rows_from_train,
  score_rows,
)

NORMS = ("s_sum", "s_mean", "s_resid")


def constant_gold_floor(merged: list[dict]) -> dict:
  comparable = [r for r in merged if r.get("choice") != "tie"]
  agree_n = sum(1 for r in comparable if human_picked_source(r) == "gold")
  return rate_summary(agree_n, len(comparable))


def decile_buckets(deltas: list[float]) -> list[int]:
  if not deltas:
    return []
  arr = np.asarray(deltas, dtype=np.float64)
  edges = np.quantile(arr, np.linspace(0, 1, 11))
  edges = np.unique(edges)
  if len(edges) <= 2:
    return [0] * len(deltas)
  return list(np.digitize(arr, edges[1:-1], right=True))


def score_lookup(scored: list[dict]) -> dict[tuple[str, str], dict]:
  out: dict[tuple[str, str], dict] = {}
  for row in scored:
    key = (str(row.get("item_id") or ""), str(row.get("source") or ""))
    out[key] = row
  return out


def pair_scores(pair: dict, lookup: dict[tuple[str, str], dict], norm: str) -> tuple[float, float] | None:
  item_id = str(pair.get("item_id") or "")
  a_src = str(pair.get("a_source") or "")
  b_src = str(pair.get("b_source") or "")
  sa = lookup.get((item_id, a_src))
  sb = lookup.get((item_id, b_src))
  if not sa or not sb:
    return None
  va = sa.get(norm)
  vb = sb.get(norm)
  if va is None or vb is None:
    return None
  return float(va), float(vb)


def agreement_with_norm(pairs: list[dict], lookup: dict[tuple[str, str], dict], norm: str) -> dict:
  rows: list[dict] = []
  for pair in pairs:
    scores = pair_scores(pair, lookup, norm)
    if scores is None:
      continue
    score_a, score_b = scores
    rows.append({**pair, "score_a": score_a, "score_b": score_b})
  stats = agreement_for_pairs(rows)
  return stats


def strata_by_delta(
  pairs: list[dict],
  lookup: dict[tuple[str, str], dict],
  norm: str,
) -> dict[str, dict]:
  items: list[tuple[int, bool | None]] = []
  for pair in pairs:
    if pair.get("choice") == "tie":
      continue
    item_id = str(pair.get("item_id") or "")
    gold_src = "gold"
    gold_row = lookup.get((item_id, gold_src))
    if not gold_row:
      if str(pair.get("a_source") or "") == "gold":
        gold_row = lookup.get((item_id, str(pair.get("a_source"))))
      elif str(pair.get("b_source") or "") == "gold":
        gold_row = lookup.get((item_id, str(pair.get("b_source"))))
    if not gold_row:
      continue
    scores = pair_scores(pair, lookup, norm)
    if scores is None:
      continue
    score_a, score_b = scores
    hp = human_pref(pair)
    sp = scorer_pref_from_scores(score_a, score_b)
    agree = (hp == sp) if hp is not None and sp is not None else None
    items.append((int(gold_row.get("delta_chars") or 0), agree))

  if not items:
    return {}
  deltas = [d for d, _ in items]
  deciles = decile_buckets([float(d) for d in deltas])
  strata: dict[str, dict] = {}
  for dec, (_, agree) in zip(deciles, items, strict=True):
    key = str(dec)
    bucket = strata.setdefault(key, {"n": 0, "agree_n": 0})
    bucket["n"] += 1
    if agree:
      bucket["agree_n"] += 1
  for bucket in strata.values():
    bucket.update(rate_summary(bucket["agree_n"], bucket["n"]))
  return strata


def pick_norm(norm_results: dict[str, dict], *, floor: dict) -> str:
  best = NORMS[0]
  best_agree = -1
  for norm in NORMS:
    agree_n = int(norm_results[norm].get("agree_n") or 0)
    if agree_n > best_agree or (agree_n == best_agree and norm == "s_sum"):
      best = norm
      best_agree = agree_n
  return best


def build_report(
  *,
  variant: str,
  norm_results: dict[str, dict],
  selected_norm: str,
  floor: dict,
  truncate: dict | None,
) -> tuple[str, dict]:
  summary = {
    "variant": variant,
    "constant_gold_floor": floor,
    "selected_norm": selected_norm,
    "truncate_train": truncate,
    "norms": {},
  }
  lines = [
    f"# ead-den-a-{variant}",
    "",
    "B valid（人間の推敲対 Composer）との一致率。同等 3 件は除外し 47 件で符号一致。",
    "",
    f"床（常に gold 側が上の定数予測器）: {fmt_rate(floor)}",
    "",
    "Phase 2 acceptance criteria は、この床を有意に上回ること。層別 n=4〜5 では Wilson 判定不能。",
    "",
    "## 正規化比較",
    "",
    md_table(
      ["正規化", "一致", "comparable", "gold 側が高い"],
      [
        [
          norm,
          fmt_rate(rate_summary(norm_results[norm]["agree_n"], norm_results[norm]["comparable_n"])),
          str(norm_results[norm]["comparable_n"]),
          str(norm_results[norm]["gold_higher_n"]),
        ]
        for norm in NORMS
      ],
    ),
    "",
    f"採用正規化（暫定）: `{selected_norm}`。ablation と D valid 結果で見直す。",
    "",
  ]

  for norm in NORMS:
    summary["norms"][norm] = {
      k: norm_results[norm][k]
      for k in (
        "n_total",
        "human_tie",
        "comparable_n",
        "agree_n",
        "human_gold",
        "human_composer",
        "gold_higher_n",
      )
    }
    summary["norms"][norm]["overall"] = rate_summary(
      norm_results[norm]["agree_n"], norm_results[norm]["comparable_n"]
    )
    strata = norm_results[norm]["strata_delta_decile"]
    summary["norms"][norm]["strata_delta_decile"] = strata
    lines.extend(
      [
        f"## Δ文字数十分位（{norm}）",
        "",
        md_table(
          ["十分位", "n", "一致率"],
          [
            [
              dec,
              str(bucket.get("n") or 0),
              fmt_rate(bucket) if bucket.get("n") else "—",
            ]
            for dec, bucket in sorted(strata.items(), key=lambda x: int(x[0]))
          ],
        ),
        "",
      ]
    )

  return "\n".join(lines), summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--variant", choices=("excl", "full"), default="excl")
  parser.add_argument("--adapter-dir", type=Path, default=None)
  parser.add_argument("--fit-train-jsonl", type=Path, default=None)
  parser.add_argument("--pairs", type=Path, default=Path("data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl"))
  parser.add_argument("--judgments", type=Path, default=Path("data/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl"))
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch

  root = args.root.resolve()
  variant = args.variant
  adapter_root = args.adapter_dir or (ead_out() / "adapters" / f"ead-den-a-{variant}")
  fit_path = args.fit_train_jsonl or (ead_work() / f"ead-den-a-{variant}-train.jsonl")
  meta_path = adapter_root / "train_meta.json"
  truncate = None
  if meta_path.is_file():
    truncate = json.loads(meta_path.read_text(encoding="utf-8")).get("truncate")

  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

  pairs = load_jsonl(root / args.pairs)
  judgments = load_jsonl(root / args.judgments)
  merged = merge_rows(pairs, judgments)
  if len(merged) != 50:
    raise SystemExit(f"expected 50 merged rows, got {len(merged)}")

  adapter_dir = adapter_root.resolve()
  if (adapter_dir / "adapter").is_dir():
    adapter_dir = adapter_dir / "adapter"

  model, tokenizer, dev = load_density_model(
    adapter_dir, args.base_model, device, trust_remote_code=args.trust_remote_code
  )

  if not fit_path.is_absolute():
    fit_path = root / fit_path
  train_rows = load_jsonl(fit_path)

  fit_scored = score_fit_rows_from_train(
    train_rows, model, tokenizer, dev, max_seq_len=args.max_seq_length
  )
  resid = fit_resid_model(fit_scored)

  scored = score_rows(model, tokenizer, dev, merged, max_seq_len=args.max_seq_length)
  apply_resid(scored, resid)

  lookup = score_lookup(scored)
  norm_results: dict[str, dict] = {}
  for norm in NORMS:
    stats = agreement_with_norm(merged, lookup, norm)
    stats["strata_delta_decile"] = strata_by_delta(merged, lookup, norm)
    norm_results[norm] = stats

  floor = constant_gold_floor(merged)
  selected = pick_norm(norm_results, floor=floor)

  md, summary = build_report(
    variant=variant,
    norm_results=norm_results,
    selected_norm=selected,
    floor=floor,
    truncate=truncate,
  )

  reports = ead_reports()
  reports.mkdir(parents=True, exist_ok=True)
  stem = f"ead-den-a-{variant}"
  write_json(reports / f"{stem}.json", summary)
  (reports / f"{stem}.md").write_text(md, encoding="utf-8")
  print(
    json.dumps(
      {
        "wrote_md": str(reports / f"{stem}.md"),
        "wrote_json": str(reports / f"{stem}.json"),
        "selected_norm": selected,
        "agree_excl_tie": {
          norm: f"{norm_results[norm]['agree_n']}/{norm_results[norm]['comparable_n']}"
          for norm in NORMS
        },
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
