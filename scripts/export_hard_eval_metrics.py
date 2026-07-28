#!/usr/bin/env python3
"""Hard Eval の既存スコアレポートから、公開用の項目正誤・候補メタデータを書き出す。

候補本文は出力しない。文字数は採点時と同じく Python の len(text)
（Unicode コードポイント数、改行・Markdown 記号を含む、前処理なし）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# evaluation_set -> labeled jsonl（候補長の取得用。本文は出力しない）
EVAL_INPUTS: dict[str, Path] = {
  "v1": ROOT / "data/hard_eval/bases_v1_labeled.jsonl",
  "v2b": ROOT / "data/hard_eval/bases_v2b_human_fable_copy.jsonl",
  "v2c": ROOT / "data/hard_eval/bases_v2c_human_machine_copy.jsonl",
}

# (evaluation_set, model_name, model_config) -> report path
# model_config は学習率・エポック・容量など識別用。不明なキーは空にしない。
REPORT_REGISTRY: list[dict[str, str]] = [
  # v1
  {
    "evaluation_set": "v1",
    "model_name": "pref-bt",
    "model_config": "default",
    "report": "outputs/hard_eval_report_bt.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-ce",
    "model_config": "default",
    "report": "outputs/hard_eval_report_ce.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-ce",
    "model_config": "beyond-para",
    "report": "outputs/hard_eval_report_ce_beyond_para.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-bt",
    "model_config": "machine-neg",
    "report": "outputs/hard_eval_v1_report_bt_machine_neg.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-bt",
    "model_config": "machine-neg-w5",
    "report": "outputs/hard_eval_v1_report_bt_machine_neg_w5.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-sentseq",
    "model_config": "lr1e-4-ep20",
    "report": "outputs/hard_eval_v1_report_sentseq.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-sentseq",
    "model_config": "lr1e-4-ep40",
    "report": "outputs/hard_eval_v1_report_sentseq40.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-sentseq",
    "model_config": "lr3e-4-ep40",
    "report": "outputs/hard_eval_v1_report_sentseq3e4.json",
  },
  {
    "evaluation_set": "v1",
    "model_name": "pref-sentseq",
    "model_config": "d512-l4-lr1e-4-ep40",
    "report": "outputs/hard_eval_v1_report_sentseq512.json",
  },
  # v2b
  {
    "evaluation_set": "v2b",
    "model_name": "pref-bt",
    "model_config": "default",
    "report": "outputs/hard_eval_v2b_report_bt.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-bt",
    "model_config": "machine-neg",
    "report": "outputs/hard_eval_v2b_report_bt_machine_neg.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-bt",
    "model_config": "machine-neg-w5",
    "report": "outputs/hard_eval_v2b_report_bt_machine_neg_w5.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-ce",
    "model_config": "beyond-para",
    "report": "outputs/hard_eval_v2b_report_ce_beyond_para.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-ce",
    "model_config": "hunk-only",
    "report": "outputs/hard_eval_v2b_report_ce_hunk_only.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-ce",
    "model_config": "ml2048",
    "report": "outputs/hard_eval_v2b_report_ce_ml2048.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-ce",
    "model_config": "with-negative-construct",
    "report": "outputs/hard_eval_v2b_report_ce_with_neg.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-sentseq",
    "model_config": "lr1e-4-ep20",
    "report": "outputs/hard_eval_v2b_report_sentseq.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-sentseq",
    "model_config": "lr1e-4-ep40",
    "report": "outputs/hard_eval_v2b_report_sentseq40.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-sentseq",
    "model_config": "lr3e-4-ep40",
    "report": "outputs/hard_eval_v2b_report_sentseq3e4.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-sentseq",
    "model_config": "d512-l4-lr1e-4-ep40",
    "report": "outputs/hard_eval_v2b_report_sentseq512.json",
  },
  {
    "evaluation_set": "v2b",
    "model_name": "pref-sentseq",
    "model_config": "anchor-full-lr3e-4-ep40",
    "report": "outputs/hard_eval_v2b_report_sentseq_anchor.json",
  },
  # v2c
  {
    "evaluation_set": "v2c",
    "model_name": "pref-bt",
    "model_config": "default",
    "report": "outputs/hard_eval_v2c_report_bt.json",
  },
  {
    "evaluation_set": "v2c",
    "model_name": "pref-bt",
    "model_config": "machine-neg",
    "report": "outputs/hard_eval_v2c_report_bt_machine_neg.json",
  },
  {
    "evaluation_set": "v2c",
    "model_name": "pref-bt",
    "model_config": "machine-neg-w5",
    "report": "outputs/hard_eval_v2c_report_bt_machine_neg_w5.json",
  },
  {
    "evaluation_set": "v2c",
    "model_name": "pref-sentseq",
    "model_config": "lr1e-4-ep20",
    "report": "outputs/hard_eval_v2c_report_sentseq.json",
  },
  {
    "evaluation_set": "v2c",
    "model_name": "pref-sentseq",
    "model_config": "lr1e-4-ep40",
    "report": "outputs/hard_eval_v2c_report_sentseq40.json",
  },
  {
    "evaluation_set": "v2c",
    "model_name": "pref-sentseq",
    "model_config": "lr3e-4-ep40",
    "report": "outputs/hard_eval_v2c_report_sentseq3e4.json",
  },
  {
    "evaluation_set": "v2c",
    "model_name": "pref-sentseq",
    "model_config": "d512-l4-lr1e-4-ep40",
    "report": "outputs/hard_eval_v2c_report_sentseq512.json",
  },
  {
    "evaluation_set": "v2c",
    "model_name": "pref-sentseq",
    "model_config": "anchor-full-lr3e-4-ep40",
    "report": "outputs/hard_eval_v2c_report_sentseq_anchor.json",
  },
]


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def load_labeled(path: Path) -> dict[str, dict]:
  by_id: dict[str, dict] = {}
  with path.open(encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if row.get("status") != "labeled":
        continue
      by_id[str(row["id"])] = row
  return by_id


def candidate_type(cand_id: str, generator: str | None) -> str:
  if cand_id in {"human", "copy", "base"}:
    return cand_id
  if cand_id.startswith("model-"):
    return "machine"
  if cand_id in {"fable", "machine"}:
    return "machine"
  if generator:
    return str(generator)
  return cand_id


def public_candidate_id(evaluation_set: str, item_id: str, cand_id: str) -> str:
  digest = hashlib.sha256(f"{evaluation_set}:{item_id}:{cand_id}".encode()).hexdigest()[:12]
  return f"{evaluation_set}-{digest}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=ROOT)
  parser.add_argument(
    "--item-out",
    type=Path,
    default=ROOT / "results/hard_eval_item_correctness.csv",
  )
  parser.add_argument(
    "--cand-out",
    type=Path,
    default=ROOT / "results/hard_eval_candidate_metadata.csv",
  )
  parser.add_argument(
    "--pairwise-detail-out",
    type=Path,
    default=ROOT / "results/raw/hard_eval_pairwise_detail.json",
    help="項目内ペア判定の再計算用中間（非公開向けでも本文なし）",
  )
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()

  for out in (args.item_out, args.cand_out, args.pairwise_detail_out):
    if out.exists() and not args.overwrite:
      raise SystemExit(f"exists (pass --overwrite): {out}")

  item_rows: list[dict] = []
  cand_rows: list[dict] = []
  pairwise_detail: dict = {
    "char_length_definition": {
      "method": "Python len(text)",
      "unicode_code_points": True,
      "includes_newlines": True,
      "includes_markdown_markup": True,
      "preprocessing": "none (same as score_hard_eval.py)",
    },
    "reference_rank_notes": {
      "v1": (
        "human.rank は構築時に設定。全項目で human が参照最上位。"
        "独立した人手の盲検順位ではない。"
      ),
      "v2b": (
        "構築時参照順位 human > fable > copy。"
        "独立した人手品質評価ではない。"
      ),
      "v2c": (
        "構築時参照順位 human > machine(composer-2.5) > copy。"
        "独立した人手品質評価ではない。"
      ),
    },
    "models": {},
  }

  missing: list[str] = []
  labeled_cache: dict[str, dict[str, dict]] = {}

  for entry in REPORT_REGISTRY:
    report_path = args.root / entry["report"]
    if not report_path.is_file():
      missing.append(str(report_path))
      continue
    report = load_json(report_path)
    summary = report["summary"]
    eval_set = entry["evaluation_set"]
    model_name = entry["model_name"]
    model_config = entry["model_config"]
    model_key = f"{eval_set}/{model_name}/{model_config}"

    expected_input = EVAL_INPUTS[eval_set]
    report_input = Path(summary["input"])
    if report_input.name != expected_input.name:
      raise SystemExit(
        f"input mismatch for {report_path}: "
        f"summary.input={summary['input']} expected={expected_input}"
      )

    if eval_set not in labeled_cache:
      if not expected_input.is_file():
        raise SystemExit(f"missing labeled input: {expected_input}")
      labeled_cache[eval_set] = load_labeled(expected_input)
    labeled = labeled_cache[eval_set]

    model_pair_events: list[dict] = []
    for item in report["items"]:
      item_id = str(item["id"])
      if item_id not in labeled:
        raise SystemExit(f"{report_path}: item {item_id} not in labeled jsonl")
      src = labeled[item_id]
      human_rank = item.get("human_rank") or src["human"].get("rank")
      if not human_rank:
        raise SystemExit(f"{item_id}: missing human_rank")
      scores = item["scores"]
      n_cands = len(scores)
      top1 = 1 if item["top1_hit"] else 0
      pairwise_correct = item.get("pairwise_agree")
      pairwise_total = item.get("pairwise_total")
      if pairwise_correct is None or pairwise_total is None:
        # スコアから再計算（集計済み数値への依存を避ける）
        agree = 0
        total = 0
        for i, better in enumerate(human_rank):
          for worse in human_rank[i + 1 :]:
            total += 1
            if scores[better] > scores[worse]:
              agree += 1
        pairwise_correct, pairwise_total = agree, total

      # 参照 top と予測 top のスコア差（同点時は 0）
      ref_top = human_rank[0]
      pred_top = item["best_model"]
      score_margin = float(scores[pred_top] - scores[ref_top]) if pred_top in scores else ""

      item_rows.append(
        {
          "evaluation_set": eval_set,
          "item_id": item_id,
          "model_name": model_name,
          "model_config": model_config,
          "checkpoint": summary.get("model", ""),
          "top1_correct": top1,
          "predicted_top_candidate": pred_top,
          "reference_top_candidate": ref_top,
          "num_candidates": n_cands,
          "pairwise_correct": pairwise_correct,
          "pairwise_total": pairwise_total,
          "score_margin": score_margin,
          "seed": "",
        }
      )

      # 候補メタ
      id_to_cand = {c["id"]: c for c in src["candidates"]}
      model_rank = item["model_rank"]
      ref_rank_pos = {cid: i + 1 for i, cid in enumerate(human_rank)}
      pred_rank_pos = {cid: i + 1 for i, cid in enumerate(model_rank)}
      for cid, score in scores.items():
        text = id_to_cand[cid]["text"]
        gen = id_to_cand[cid].get("generator")
        cand_rows.append(
          {
            "evaluation_set": eval_set,
            "item_id": item_id,
            "candidate_id": public_candidate_id(eval_set, item_id, cid),
            "candidate_type": candidate_type(cid, gen),
            "candidate_role": cid,
            "char_length": len(text),
            "model_score": score,
            "model_name": model_name,
            "model_config": model_config,
            "reference_rank": ref_rank_pos[cid],
            "predicted_rank": pred_rank_pos[cid],
          }
        )

      # pairwise 詳細（本文なし）
      for i, better in enumerate(human_rank):
        for worse in human_rank[i + 1 :]:
          model_pair_events.append(
            {
              "item_id": item_id,
              "better": better,
              "worse": worse,
              "agree": bool(scores[better] > scores[worse]),
              "score_better": scores[better],
              "score_worse": scores[worse],
            }
          )

    pairwise_detail["models"][model_key] = {
      "report": str(report_path.relative_to(args.root)),
      "checkpoint": summary.get("model"),
      "summary_pairwise_agree": summary.get("pairwise_agree"),
      "summary_pairwise_total": summary.get("pairwise_total"),
      "recomputed_pairwise_agree": sum(1 for e in model_pair_events if e["agree"]),
      "recomputed_pairwise_total": len(model_pair_events),
      "pairs": model_pair_events,
    }
    print(
      f"{model_key}: items={len(report['items'])} "
      f"top1={sum(1 for it in report['items'] if it['top1_hit'])} "
      f"pairwise={sum(1 for e in model_pair_events if e['agree'])}/{len(model_pair_events)}"
    )

  if missing:
    print("MISSING reports (skipped):", file=sys.stderr)
    for path in missing:
      print(f"  {path}", file=sys.stderr)
    if not item_rows:
      raise SystemExit("no reports found")

  write_csv(
    args.item_out,
    item_rows,
    [
      "evaluation_set",
      "item_id",
      "model_name",
      "model_config",
      "checkpoint",
      "top1_correct",
      "predicted_top_candidate",
      "reference_top_candidate",
      "num_candidates",
      "pairwise_correct",
      "pairwise_total",
      "score_margin",
      "seed",
    ],
  )
  write_csv(
    args.cand_out,
    cand_rows,
    [
      "evaluation_set",
      "item_id",
      "candidate_id",
      "candidate_type",
      "candidate_role",
      "char_length",
      "model_score",
      "model_name",
      "model_config",
      "reference_rank",
      "predicted_rank",
    ],
  )
  args.pairwise_detail_out.parent.mkdir(parents=True, exist_ok=True)
  args.pairwise_detail_out.write_text(
    json.dumps(pairwise_detail, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"wrote {args.item_out} ({len(item_rows)} rows)")
  print(f"wrote {args.cand_out} ({len(cand_rows)} rows)")
  print(f"wrote {args.pairwise_detail_out}")


if __name__ == "__main__":
  main()
