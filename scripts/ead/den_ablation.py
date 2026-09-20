#!/usr/bin/env python3
"""Phase 2 ablation: log p_θ, log p_0, s_den の対判別寄与。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import human_pref, merge_rows
from ead.common import (
  POSITION_RANK,
  ead_reports,
  fmt_rate,
  load_jsonl,
  md_table,
  rate_summary,
  repo_root,
  write_json,
)
from eval_pref_valid50_gold_vs_composer import agreement_for_pairs, human_picked_source
from eval_pref_multigranular_blind60 import scorer_pref_from_scores

from ead.den_score import load_density_model, score_rows

SIGNALS = (
  ("logp_adapter", "log p_θ(y|x)"),
  ("logp_base", "log p_0(y|x)"),
  ("s_sum", "s_den = log p_θ − log p_0"),
)

ORIGIN_LABELS = {
  "gen_c_human_d": "(生成-c, 人間-d)",
  "gen_c_gen_d": "(生成-c, 生成-d)",
}


def y_origin(y_kind: str) -> str:
  return "human" if y_kind == "human" else "gen"


def c_vs_d_origin_key(low_y_kind: str, high_y_kind: str) -> str | None:
  if y_origin(low_y_kind) != "gen":
    return None
  if y_origin(high_y_kind) == "human":
    return "gen_c_human_d"
  if y_origin(high_y_kind) == "gen":
    return "gen_c_gen_d"
  return None


def position_rank(position: str) -> int | None:
  if position == "eq":
    return POSITION_RANK["b"]
  return POSITION_RANK.get(position)


def constant_gold_floor(merged: list[dict]) -> dict:
  comparable = [r for r in merged if r.get("choice") != "tie"]
  agree_n = sum(1 for r in comparable if human_picked_source(r) == "gold")
  return rate_summary(agree_n, len(comparable))


def score_lookup(scored: list[dict]) -> dict[tuple[str, str], dict]:
  out: dict[tuple[str, str], dict] = {}
  for row in scored:
    key = (str(row.get("item_id") or ""), str(row.get("source") or ""))
    out[key] = row
  return out


def pair_signal_scores(
  pair: dict, lookup: dict[tuple[str, str], dict], signal: str
) -> tuple[float, float] | None:
  item_id = str(pair.get("item_id") or "")
  a_src = str(pair.get("a_source") or "")
  b_src = str(pair.get("b_source") or "")
  sa = lookup.get((item_id, a_src))
  sb = lookup.get((item_id, b_src))
  if not sa or not sb:
    return None
  va = sa.get(signal)
  vb = sb.get(signal)
  if va is None or vb is None:
    return None
  return float(va), float(vb)


def confusion_matrix(merged: list[dict], lookup: dict[tuple[str, str], dict], signal: str) -> dict:
  hh = hc = ch = cc = 0
  for pair in merged:
    if pair.get("choice") == "tie":
      continue
    scores = pair_signal_scores(pair, lookup, signal)
    if scores is None:
      continue
    score_a, score_b = scores
    picked = human_picked_source(pair)
    sp = scorer_pref_from_scores(score_a, score_b)
    if sp is None:
      continue
    a_src = str(pair.get("a_source") or "")
    b_src = str(pair.get("b_source") or "")
    model_human = (sp == "a" and a_src == "gold") or (sp == "b" and b_src == "gold")
    human_human = picked == "gold"
    if human_human and model_human:
      hh += 1
    elif human_human and not model_human:
      hc += 1
    elif not human_human and model_human:
      ch += 1
    else:
      cc += 1
  return {
    "human_human_model_human": hh,
    "human_human_model_composer": hc,
    "human_composer_model_human": ch,
    "human_composer_model_composer": cc,
  }


def b_valid_ablation(merged: list[dict], scored: list[dict]) -> dict:
  lookup = score_lookup(scored)
  floor = constant_gold_floor(merged)
  out: dict = {"constant_gold_floor": floor, "signals": {}}
  for key, label in SIGNALS:
    rows: list[dict] = []
    for pair in merged:
      scores = pair_signal_scores(pair, lookup, key)
      if scores is None:
        continue
      score_a, score_b = scores
      rows.append({**pair, "score_a": score_a, "score_b": score_b})
    stats = agreement_for_pairs(rows)
    stats["label"] = label
    stats["confusion"] = confusion_matrix(merged, lookup, key)
    stats["overall"] = rate_summary(stats["agree_n"], stats["comparable_n"])
    out["signals"][key] = stats
  return out


def d_row_to_score_input(row: dict) -> dict:
  return {
    "item_id": row["item_id"],
    "draft": row["draft"],
    "y": row["y"],
    "source": row["row_id"],
    "position": row["position"],
    "y_kind": row.get("y_kind"),
  }


def c_vs_d_pair_metrics(rows: list[dict], scores: dict[str, float | None]) -> dict:
  by_item: dict[str, list[dict]] = defaultdict(list)
  for row in rows:
    by_item[str(row["item_id"])].append(row)

  grouped: dict[str, dict] = {"all": {"correct": 0.0, "total": 0}}
  origin_grouped: dict[str, dict] = {}

  for item_rows in by_item.values():
    indexed = []
    for row in item_rows:
      score = scores.get(row["row_id"])
      if score is None:
        continue
      rank = position_rank(str(row["position"]))
      if rank is None:
        continue
      indexed.append(
        {
          "position": row["position"],
          "rank": rank,
          "score": score,
          "y_kind": str(row.get("y_kind") or ""),
          "row_id": row["row_id"],
        }
      )
    for a, b in combinations(indexed, 2):
      if a["rank"] == b["rank"]:
        continue
      low, high = (a, b) if a["rank"] < b["rank"] else (b, a)
      if not (low["position"] == "c" and high["position"] == "d"):
        continue
      grouped["all"]["total"] += 1
      correct = 0.0
      if high["score"] > low["score"]:
        correct = 1.0
      elif high["score"] == low["score"]:
        correct = 0.5
      grouped["all"]["correct"] += correct
      origin = c_vs_d_origin_key(low["y_kind"], high["y_kind"])
      if origin is not None:
        bucket = origin_grouped.setdefault(origin, {"correct": 0.0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += correct

  out_all = rate_summary(int(round(grouped["all"]["correct"])), grouped["all"]["total"])
  by_origin = {
    origin: rate_summary(int(round(bucket["correct"])), bucket["total"])
    for origin, bucket in origin_grouped.items()
  }
  return {"all": out_all, "by_origin": by_origin}


def d_valid_ablation(d_rows: list[dict], scored: list[dict]) -> dict:
  by_row_id: dict[str, dict] = {str(row.get("source") or ""): row for row in scored}
  out: dict = {"signals": {}}
  for key, label in SIGNALS:
    scores_map = {
      rid: float(metrics[key])
      for rid, metrics in by_row_id.items()
      if metrics.get(key) is not None
    }
    metrics = c_vs_d_pair_metrics(d_rows, scores_map)
    metrics["label"] = label
    out["signals"][key] = metrics
  return out


def build_markdown(b_valid: dict, d_valid: dict) -> str:
  floor = b_valid["constant_gold_floor"]
  lines = [
    "# ead-den-a-excl-ablation",
    "",
    "B valid: 人手判定との符号一致（同等 3 件除外）。床は「常に人間（gold）側が上」の定数予測器。",
    "",
    f"定数予測器（常に gold）: {fmt_rate(floor)}",
    "",
    "## B valid",
    "",
    md_table(
      ["信号", "一致", "comparable", "定数床との差（件数）"],
      [
        [
          b_valid["signals"][key]["label"],
          fmt_rate(b_valid["signals"][key]["overall"]),
          str(b_valid["signals"][key]["comparable_n"]),
          str(b_valid["signals"][key]["agree_n"] - floor["k"]),
        ]
        for key, _ in SIGNALS
      ],
    ),
    "",
  ]
  for key, _ in SIGNALS:
    conf = b_valid["signals"][key]["confusion"]
    lines.extend(
      [
        f"### B valid / {b_valid['signals'][key]['label']}",
        "",
        md_table(
          ["", "モデル: 人間↑", "モデル: Composer↑"],
          [
            ["人手: 人間↑", str(conf["human_human_model_human"]), str(conf["human_human_model_composer"])],
            [
              "人手: Composer↑",
              str(conf["human_composer_model_human"]),
              str(conf["human_composer_model_composer"]),
            ],
          ],
        ),
        "",
      ]
    )

  lines.extend(
    [
      "## D valid / c vs d",
      "",
      "同一下書き内で d のスコアが c より高い割合。",
      "",
      md_table(
        ["信号", "全体", "n", "(生成-c, 人間-d)", "n", "(生成-c, 生成-d)", "n"],
        [
          [
            d_valid["signals"][key]["label"],
            fmt_rate(d_valid["signals"][key]["all"]),
            str(d_valid["signals"][key]["all"].get("n", "—")),
            fmt_rate(d_valid["signals"][key]["by_origin"].get("gen_c_human_d", {})),
            str(d_valid["signals"][key]["by_origin"].get("gen_c_human_d", {}).get("n", "—")),
            fmt_rate(d_valid["signals"][key]["by_origin"].get("gen_c_gen_d", {})),
            str(d_valid["signals"][key]["by_origin"].get("gen_c_gen_d", {}).get("n", "—")),
          ]
          for key, _ in SIGNALS
        ],
      ),
      "",
    ]
  )
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--variant", choices=("excl", "full"), default="excl")
  parser.add_argument("--adapter-dir", type=Path, default=None)
  parser.add_argument("--pairs", type=Path, default=Path("data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl"))
  parser.add_argument("--judgments", type=Path, default=Path("data/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl"))
  parser.add_argument("--d-valid", type=Path, default=Path("data/d/valid.jsonl"))
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  import torch

  root = args.root.resolve()
  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

  adapter_root = args.adapter_dir or (root / "outputs/ead/adapters" / f"ead-den-a-{args.variant}")
  adapter_dir = adapter_root.resolve()
  if (adapter_dir / "adapter").is_dir():
    adapter_dir = adapter_dir / "adapter"

  pairs = load_jsonl(root / args.pairs)
  judgments = load_jsonl(root / args.judgments)
  merged = merge_rows(pairs, judgments)
  d_rows = load_jsonl(root / args.d_valid)

  model, tokenizer, dev = load_density_model(
    adapter_dir, args.base_model, device, trust_remote_code=args.trust_remote_code
  )

  b_scored = score_rows(model, tokenizer, dev, merged, max_seq_len=args.max_seq_length)
  d_input = [d_row_to_score_input(r) for r in d_rows]
  d_scored = score_rows(model, tokenizer, dev, d_input, max_seq_len=args.max_seq_length)

  b_result = b_valid_ablation(merged, b_scored)
  d_result = d_valid_ablation(d_rows, d_scored)

  summary = {
    "variant": args.variant,
    "b_valid": b_result,
    "d_valid_c_vs_d": d_result,
  }

  reports = ead_reports()
  stem = f"ead-den-a-{args.variant}-ablation"
  write_json(reports / f"{stem}.json", summary)
  (reports / f"{stem}.md").write_text(build_markdown(b_result, d_result), encoding="utf-8")
  print(
    json.dumps(
      {
        "wrote_md": str(reports / f"{stem}.md"),
        "constant_floor": fmt_rate(b_result["constant_gold_floor"]),
        "b_valid": {
          key: f"{b_result['signals'][key]['agree_n']}/{b_result['signals'][key]['comparable_n']}"
          for key, _ in SIGNALS
        },
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
