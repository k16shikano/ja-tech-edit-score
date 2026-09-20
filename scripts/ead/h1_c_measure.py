#!/usr/bin/env python3
"""R5-A: C 56 件の H1 事前登録測定（選択ではなく測定）。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import merge_rows
from ead.common import ead_out, ead_reports, ead_work, fmt_rate, load_jsonl, md_table, rate_summary, repo_root, write_json
from eval_pref_multigranular_blind60 import agreement_for_pairs

from ead.den_score import apply_resid, fit_resid_model, load_density_model, score_fit_rows_from_train, score_rows

try:
  import torch
except ImportError as exc:
  raise SystemExit(f"torch required: {exc}") from exc


COMPARE_TYPE = "5_gold_vs_adapter_selected"
PREREG = {
  "purpose": "H1 measurement on C (not selection)",
  "normalization": "s_sum",
  "metric": "sign_agreement_rate",
  "denominator": "56 comparable pairs (60 total, 4 human ties excluded)",
  "forbidden": [
    "threshold tuning on C",
    "model selection on C",
    "normalization selection on C",
    "changing any pipeline based on this result before Phase 4",
  ],
  "adapter": "ead-den-a-excl",
}


def write_preregister(path: Path) -> None:
  obj = {**PREREG, "registered_at": datetime.now(timezone.utc).isoformat()}
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_c_merged(root: Path, pairs_path: Path, judgments_path: Path) -> list[dict]:
  pairs = [p for p in load_jsonl(pairs_path) if p.get("compare_type") == COMPARE_TYPE]
  judgments = load_jsonl(judgments_path)
  merged = merge_rows(pairs, judgments)
  if len(merged) != 60:
    raise SystemExit(f"expected 60 C pairs, got {len(merged)}")
  return merged


def score_lookup(scored: list[dict]) -> dict[tuple[str, str], dict]:
  out: dict[tuple[str, str], dict] = {}
  for row in scored:
    key = (str(row.get("item_id") or ""), str(row.get("source") or ""))
    out[key] = row
  return out


def pair_s_sum(pair: dict, lookup: dict[tuple[str, str], dict]) -> tuple[float, float] | None:
  item_id = str(pair.get("item_id") or "")
  a_src = str(pair.get("a_source") or "")
  b_src = str(pair.get("b_source") or "")
  sa = lookup.get((item_id, a_src))
  sb = lookup.get((item_id, b_src))
  if not sa or not sb:
    return None
  return float(sa["s_sum"]), float(sb["s_sum"])


def build_report(stats: dict, *, prereg_path: Path) -> str:
  lines = [
    "# ead-h1-c-measure",
    "",
    "R5-A: C 上の H1 **事前登録測定**。選択・閾値調整・正規化選択ではない。",
    "",
    "## 事前登録",
    "",
    f"- 正規化: `{PREREG['normalization']}`",
    f"- 指標: {PREREG['metric']}（分母 {PREREG['denominator']}）",
    "- 閾値調整・モデル選択: 行わない",
    "- 結果に基づくパイプライン変更: Phase 4 着手前は禁止",
    f"- 登録ファイル: `{prereg_path.name}`",
    "",
    "## 結果",
    "",
    md_table(
      ["指標", "値"],
      [
        ["総件数", str(stats["n_total"])],
        ["人手同等（除外）", str(stats["human_tie"])],
        ["符号一致可能", str(stats["comparable_n"])],
        ["符号一致", str(stats["agree_n"])],
        ["符号一致率", fmt_rate(rate_summary(stats["agree_n"], stats["comparable_n"]))],
        ["gold 側が $s_\\mathrm{sum}$ で高い", str(stats["gold_higher_n"])],
      ],
    ),
    "",
    "Phase 4 の best-of-8 一致率とは別測定である。H1 の結果を見て Phase 4 の設定を変えてはならない。",
  ]
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--adapter-dir", type=Path, default=None)
  parser.add_argument("--fit-train-jsonl", type=Path, default=None)
  parser.add_argument("--pairs", type=Path, default=Path("data/blind_eval/pairs_gold_vs_adapter_selected.jsonl"))
  parser.add_argument("--judgments", type=Path, default=Path("data/blind_eval/judgments_gold_vs_adapter_selected.jsonl"))
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--trust-remote-code", action="store_true")
  args = parser.parse_args()

  root = args.root.resolve()
  prereg_path = ead_work() / "h1_c_preregister.json"
  write_preregister(prereg_path)

  device = args.device
  if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

  merged = load_c_merged(root, root / args.pairs, root / args.judgments)
  adapter_root = args.adapter_dir or (ead_out() / "adapters" / "ead-den-a-excl")
  adapter_dir = adapter_root.resolve()
  if (adapter_dir / "adapter").is_dir():
    adapter_dir = adapter_dir / "adapter"

  fit_path = args.fit_train_jsonl or (ead_work() / "ead-den-a-excl-train.jsonl")
  if not fit_path.is_absolute():
    fit_path = root / fit_path

  model, tokenizer, dev = load_density_model(
    adapter_dir, args.base_model, device, trust_remote_code=args.trust_remote_code
  )
  fit_scored = score_fit_rows_from_train(
    load_jsonl(fit_path), model, tokenizer, dev, max_seq_len=args.max_seq_length
  )
  resid = fit_resid_model(fit_scored)

  scored = score_rows(model, tokenizer, dev, merged, max_seq_len=args.max_seq_length)
  apply_resid(scored, resid)
  lookup = score_lookup(scored)

  scored_pairs: list[dict] = []
  for pair in merged:
    vals = pair_s_sum(pair, lookup)
    if vals is None:
      continue
    score_a, score_b = vals
    scored_pairs.append({**pair, "score_a": score_a, "score_b": score_b})

  stats = agreement_for_pairs(scored_pairs)
  stats["normalization"] = "s_sum"
  stats["preregister"] = PREREG

  reports = ead_reports()
  write_json(reports / "ead-h1-c-measure.json", stats)
  (reports / "ead-h1-c-measure.md").write_text(build_report(stats, prereg_path=prereg_path), encoding="utf-8")
  print(
    json.dumps(
      {
        "wrote": str(reports / "ead-h1-c-measure.md"),
        "agree_n": stats["agree_n"],
        "comparable_n": stats["comparable_n"],
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
