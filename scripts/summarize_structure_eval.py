#!/usr/bin/env python3
"""層別 Hard Eval の採点結果を集計する。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

STRATA = ("structure_dominant", "expression_dominant", "mixed", "unclassified")
MODELS = (
  ("bt", "outputs/pref-bt"),
  ("ce", "outputs/pref-ce-beyond-para"),
  ("ce", "outputs/pref-ce-ml2048"),
  ("ce", "outputs/pref-ce-with-negative-construct-example"),
)
MODEL_LABELS = {
  "outputs/pref-bt": "pref-bt",
  "outputs/pref-ce-beyond-para": "pref-ce-beyond-para",
  "outputs/pref-ce-ml2048": "pref-ce-ml2048",
  "outputs/pref-ce-with-negative-construct-example": "pref-ce-with-negative-construct-example",
}


def load_report(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def load_items(path: Path) -> list[dict]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def filter_rows(report: dict, items: list[dict], *, fits_512_only: bool) -> list[dict]:
  if fits_512_only:
    allowed = {it["id"] for it in items if it.get("meta", {}).get("fits_512")}
    return [row for row in report["items"] if row["id"] in allowed]
  return list(report["items"])


def margins_from_rows(rows: list[dict]) -> list[float]:
  out: list[float] = []
  for row in rows:
    scores = row.get("scores") or {}
    if "human" not in scores or "base" not in scores:
      continue
    out.append(float(scores["human"]) - float(scores["base"]))
  return out


def summarize_margins(margins: list[float]) -> dict:
  if not margins:
    return {
      "count": 0,
      "mean": None,
      "median": None,
      "min": None,
    }
  return {
    "count": len(margins),
    "mean": float(statistics.mean(margins)),
    "median": float(statistics.median(margins)),
    "min": float(min(margins)),
  }


def subset_accuracy(rows: list[dict]) -> tuple[float | None, int, int]:
  if not rows:
    return None, 0, 0
  hits = sum(1 for row in rows if row["top1_hit"])
  return hits / len(rows), hits, len(rows)


def pct(value: float | None) -> str:
  if value is None:
    return "—"
  return f"{value:.1%}"


def num(value: float | None, *, digits: int = 3) -> str:
  if value is None:
    return "—"
  return f"{value:.{digits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--eval-dir", default="data/hard_eval")
  parser.add_argument("--report-dir", default="outputs/structure_eval")
  parser.add_argument("--out", default="outputs/structure_eval/summary.md")
  args = parser.parse_args()

  root = Path(__file__).resolve().parent.parent
  eval_dir = Path(args.eval_dir)
  if not eval_dir.is_absolute():
    eval_dir = root / eval_dir
  report_dir = Path(args.report_dir)
  if not report_dir.is_absolute():
    report_dir = root / report_dir
  out_path = Path(args.out)
  if not out_path.is_absolute():
    out_path = root / out_path

  build_report_path = eval_dir / "structure_eval_build_report.json"
  build_report = load_report(build_report_path) if build_report_path.exists() else {}

  table_rows: list[dict] = []
  model_all_margins: dict[str, list[float]] = {}

  for stratum in STRATA:
    eval_path = eval_dir / f"{stratum}.jsonl"
    items = load_items(eval_path) if eval_path.exists() else []
    for _scorer, model_rel in MODELS:
      model_label = MODEL_LABELS[model_rel]
      report_path = report_dir / f"{model_label}__{stratum}.json"
      if not report_path.exists():
        continue
      report = load_report(report_path)
      all_rows = filter_rows(report, items, fits_512_only=False)
      fit_rows = filter_rows(report, items, fits_512_only=True)
      all_acc, all_hits, all_n = subset_accuracy(all_rows)
      fit_acc, fit_hits, fit_n = subset_accuracy(fit_rows)
      all_margin_stats = summarize_margins(margins_from_rows(all_rows))
      fit_margin_stats = summarize_margins(margins_from_rows(fit_rows))
      model_all_margins.setdefault(model_label, []).extend(margins_from_rows(all_rows))

      table_rows.append(
        {
          "stratum": stratum,
          "model": model_label,
          "all_accuracy": all_acc,
          "all_hits": all_hits,
          "all_n": all_n,
          "fits_512_accuracy": fit_acc,
          "fits_512_hits": fit_hits,
          "fits_512_n": fit_n,
          "margin_mean": all_margin_stats["mean"],
          "margin_median": all_margin_stats["median"],
          "margin_min": all_margin_stats["min"],
          "margin_fits512_mean": fit_margin_stats["mean"],
          "margin_fits512_median": fit_margin_stats["median"],
        }
      )

  model_margin_medians = {
    model: statistics.median(margins) if margins else None
    for model, margins in model_all_margins.items()
  }
  for row in table_rows:
    denom = model_margin_medians.get(row["model"])
    if denom and denom != 0 and row["margin_median"] is not None:
      row["margin_median_normalized"] = row["margin_median"] / denom
      row["margin_mean_normalized"] = (
        row["margin_mean"] / denom if row["margin_mean"] is not None else None
      )
    else:
      row["margin_median_normalized"] = None
      row["margin_mean_normalized"] = None

  lines = [
    "# Structure Eval 集計",
    "",
    "各層で `human > base`（top1）の勝率と、スコア差 `score(human) - score(base)` の分布。",
    "",
  ]
  if build_report:
    th = build_report.get("thresholds", {})
    lines.extend(
      [
        "## 層の閾値と件数",
        "",
        f"- min_matched_sentences: {th.get('min_matched_sentences')}",
        f"- min_alignment_coverage: {th.get('min_alignment_coverage')}（structure-dominant 必須）",
        f"- high_expression_change: {th.get('high_expression_change')}（`expression_change.total`）",
        f"- high_structure_change: {th.get('high_structure_change')}",
        f"- low_expression_change: {th.get('low_expression_change')}（`expression_change.matched`、structure-dominant）",
        "",
        f"- 修正前 structure-dominant（held-out）: {build_report.get('legacy_structure_dominant_count')}",
        f"- coverage 条件で除外: {build_report.get('excluded_from_structure_dominant_count')}",
        "",
        "### held-out 由来（2048 token 以内）",
        "",
      ]
    )
    for key in ("structure-dominant", "expression-dominant", "mixed"):
      count = build_report.get("strata_counts", {}).get(key, 0)
      bucket = build_report.get("strata_bucket_counts", {}).get(key, 0)
      fits = build_report.get("fits_512_counts", {}).get(key, 0)
      dropped = build_report.get("token_dropped", {}).get(key, 0)
      lines.append(f"- {key}: {count} 件（層内 {bucket}、fits_512: {fits}, token drop: {dropped}）")
    lines.extend(
      [
        f"- unclassified: {build_report.get('unclassified_eval_count', 0)} 件 "
        f"（held-out 層外 {build_report.get('heldout_unclassified', 0)}、"
        f"fits_512: {build_report.get('unclassified_fits_512_count', 0)}）",
        f"- raw structure-dominant train candidates: {build_report.get('train_candidates_raw_structure_dominant', 0)}",
        "",
      ]
    )
    examples = build_report.get("excluded_from_structure_dominant_examples") or []
    if examples:
      lines.extend(["### structure-dominant から外れた代表例（coverage 不足）", ""])
      for ex in examples[:5]:
        lines.append(
          f"- `{ex['id']}` coverage={ex.get('coverage_source'):.3f}/"
          f"{ex.get('coverage_edited'):.3f}, matched={ex.get('expression_change_matched'):.3f}, "
          f"total={ex.get('expression_change_total'):.3f}, "
          f"del/add={ex.get('deleted_count')}/{ex.get('added_count')}"
        )
      lines.append("")

  lines.extend(
    [
      "## 勝率",
      "",
      "| 層 | モデル | 勝率 | n | fits_512 勝率 | n |",
      "|---|---|---:|---:|---:|---:|",
    ]
  )
  for row in table_rows:
    lines.append(
      f"| {row['stratum']} | {row['model']} | {pct(row['all_accuracy'])} | {row['all_n']} | "
      f"{pct(row['fits_512_accuracy'])} | {row['fits_512_n']} |"
    )

  lines.extend(
    [
      "",
      "## 点差（human − base）",
      "",
      "| 層 | モデル | 平均 | 中央値 | 最小 | 中央値/モデル中央値 |",
      "|---|---|---:|---:|---:|---:|",
    ]
  )
  for row in table_rows:
    lines.append(
      f"| {row['stratum']} | {row['model']} | {num(row['margin_mean'])} | "
      f"{num(row['margin_median'])} | {num(row['margin_min'])} | "
      f"{num(row['margin_median_normalized'])} |"
    )
  lines.append("")

  # structure vs unclassified comparison per model
  lines.extend(["## 構成支配 vs unclassified（点差中央値・正規化）", ""])
  for model_label in sorted(model_margin_medians.keys()):
    sd = next((r for r in table_rows if r["model"] == model_label and r["stratum"] == "structure_dominant"), None)
    un = next((r for r in table_rows if r["model"] == model_label and r["stratum"] == "unclassified"), None)
    if not sd or not un:
      continue
    lines.append(
      f"- {model_label}: structure_median={num(sd['margin_median'])}, "
      f"unclassified_median={num(un['margin_median'])}, "
      f"structure_norm={num(sd['margin_median_normalized'])}, "
      f"unclassified_norm={num(un['margin_median_normalized'])}"
    )
  lines.append("")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  summary_json = {
    "build_report": build_report,
    "model_margin_medians": model_margin_medians,
    "results": table_rows,
  }
  out_path.with_suffix(".json").write_text(
    json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"wrote {out_path}")


if __name__ == "__main__":
  main()
