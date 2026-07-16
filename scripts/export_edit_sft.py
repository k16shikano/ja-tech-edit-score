#!/usr/bin/env python3
"""系統1フェーズ0: 推敲ペアを編集モデル SFT 形式へ書き出す。

- 入力: data/revision_pairs.jsonl（draft / revised / project_id）
- 出力: chat messages 形式の train / heldout JSONL と、基準分布 stats.json
- held-out は書籍（project_id）単位で固定する（LOPO と同じ粒度）
- 指示は短い固定文。規範スキル全文は入れない（スキル依存を測る対照のため）
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from steering_utils import load_revision_pairs

INSTRUCTION = "次の下書きを、意味を保ったまま日本語の技術文書として推敲せよ。"

# held-out は編集の重さ（draft/revised 類似度の中央値）が偏らないように選ぶ:
#   what-is-monad=軽め 0.78 / computer-arch-revisit=重め 0.31 / ir-system=重い 0.26
DEFAULT_HOLDOUT = ["what-is-monad", "computer-arch-revisit", "ir-system"]


def to_chat_row(pair: dict) -> dict:
  return {
    "messages": [
      {"role": "user", "content": f"{INSTRUCTION}\n\n{pair['draft']}"},
      {"role": "assistant", "content": pair["revised"]},
    ],
    "meta": {"id": pair["id"], "project_id": pair["project_id"]},
  }


def pair_stats(pairs: list[dict]) -> dict:
  len_ratios = []
  similarities = []
  for p in pairs:
    d, r = p["draft"], p["revised"]
    len_ratios.append(len(r) / max(len(d), 1))
    similarities.append(difflib.SequenceMatcher(None, d, r).ratio())

  def pct(a: list[float]) -> dict:
    arr = np.asarray(a)
    return {
      "mean": float(arr.mean()),
      "p10": float(np.percentile(arr, 10)),
      "p50": float(np.percentile(arr, 50)),
      "p90": float(np.percentile(arr, 90)),
    }

  return {
    "n": len(pairs),
    "length_ratio_revised_over_draft": pct(len_ratios),
    "char_similarity_ratio": pct(similarities),
    "projects": dict(Counter(p["project_id"] for p in pairs)),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/revision_pairs.jsonl")
  parser.add_argument("--out-dir", default="data/edit_sft")
  parser.add_argument(
    "--holdout-projects",
    default=",".join(DEFAULT_HOLDOUT),
    help="held-out にする project_id（カンマ区切り）",
  )
  parser.add_argument("--seed", type=int, default=0, help="train のシャッフル用")
  args = parser.parse_args()

  pairs = load_revision_pairs(Path(args.pairs))
  if not pairs:
    raise SystemExit("no pairs")

  holdout_projects = {p.strip() for p in args.holdout_projects.split(",") if p.strip()}
  known = {p["project_id"] for p in pairs}
  missing = holdout_projects - known
  if missing:
    raise SystemExit(f"unknown holdout projects: {sorted(missing)}")

  train = [p for p in pairs if p["project_id"] not in holdout_projects]
  heldout = [p for p in pairs if p["project_id"] in holdout_projects]
  random.Random(args.seed).shuffle(train)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  for name, subset in (("train", train), ("heldout", heldout)):
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
      for p in subset:
        f.write(json.dumps(to_chat_row(p), ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(subset)} rows)")

  stats = {
    "instruction": INSTRUCTION,
    "holdout_projects": sorted(holdout_projects),
    "seed": args.seed,
    "train": pair_stats(train),
    "heldout": pair_stats(heldout),
  }
  stats_path = out_dir / "stats.json"
  stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {stats_path}")
  print(
    f"train={len(train)} heldout={len(heldout)} "
    f"({len(heldout) / (len(train) + len(heldout)):.1%} held out)"
  )


if __name__ == "__main__":
  main()
