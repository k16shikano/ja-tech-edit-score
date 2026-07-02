#!/usr/bin/env python3
"""Score edit-branch revisions against a learned preference model (base vs edit hunks)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))


def format_pref_detail_line(detail: dict[str, float]) -> str:
  if not detail:
    return ""
  pa = detail.get("prob_a_order_AB", 0.0)
  pb = detail.get("prob_b_order_AB", 0.0)
  pbf = detail.get("prob_first_is_B_order_BAfirst", 0.0)
  pba = detail.get("prob_second_is_A_order_BAfirst", 0.0)
  return (
    f"    [対称内訳] (入力,edit,base): P(edit)= {pa:.4f}, P(base)= {pb:.4f} | "
    f"(入力,base,edit): P(先=base)= {pbf:.4f}, P(後=edit)= {pba:.4f}"
  )


def format_stats_text(stats: dict, thresholds: dict, model_metrics: dict) -> list[str]:
  lines = [
    "",
    "数値サマリー（hunk 単位）:",
    f"  mean margin (edit-base): {stats['mean_margin']:+.4f}  "
    f"[min {stats['min_margin']:+.4f}, max {stats['max_margin']:+.4f}]",
    f"  mean score_edit: {stats['mean_score_edit']:.4f}  mean score_base: {stats['mean_score_base']:.4f}",
    f"  edit 優位 hunk 比率 (margin>0): {stats['edit_win_rate']:.1%}",
    "",
    "判定閾値:",
    f"  preferred_edit: margin >= {thresholds['uncertain']:+.2f}",
    f"  uncertain: |margin| < {thresholds['uncertain']:.2f}",
    f"  preferred_base: margin <= {-thresholds['uncertain']:+.2f}",
    f"  reject_edit: margin <= {-thresholds['reject']:+.2f}",
  ]
  if model_metrics:
    acc = model_metrics.get("eval_accuracy")
    n_eval = model_metrics.get("eval_samples")
    embed = model_metrics.get("embedding_model", "")
    lines.extend(
      [
        "",
        "モデル参考値 (metrics.json):",
        f"  eval_accuracy: {acc:.4f}" if acc is not None else "  eval_accuracy: n/a",
        f"  eval_samples: {n_eval}" if n_eval is not None else "",
        f"  embedding: {embed}" if embed else "",
      ]
    )
  return [ln for ln in lines if ln]


def format_hunk_scores_text(hunks: list[dict], *, all_hunks: bool) -> list[str]:
  lines: list[str] = []
  if all_hunks:
    lines.extend(["", "全 hunk スコア:"])
    for hunk in hunks:
      loc = f"L{hunk['line_new']} " if hunk.get("line_new") else ""
      lines.append(
        f"  #{hunk['index']} {loc}"
        f"edit={hunk['score_edit']:.4f} base={hunk['score_base']:.4f} "
        f"margin={hunk['margin']:+.4f} [{hunk['verdict']}]"
      )
      detail_line = format_pref_detail_line(hunk.get("pref_detail") or {})
      if detail_line:
        lines.append(detail_line)
  return lines


def format_text(payload: dict) -> str:
  meta = payload["meta"]
  summary = payload["summary"]
  stats = payload.get("statistics", {})
  thresholds = payload.get("thresholds", {})
  model_metrics = payload.get("model_metrics", {})
  lines = [
    f"推敲選好チェック: {meta['file']}",
    f"  repo: {meta['repo']}",
    f"  compare: {meta['compare_label']}",
    f"  diff: {meta['diff_mode']}",
    f"  model: {meta['model_dir']}",
    "",
    "サマリー:",
    f"  hunks: {summary['total']}",
    f"  採用方向 (edit優位): {summary['preferred_edit']}",
    f"  要確認 (uncertain): {summary['uncertain']}",
    f"  下書き優位: {summary['preferred_base']}",
  ]
  if summary.get("reject_edit"):
    lines.append(f"  強い逆方向: {summary['reject_edit']}")

  if stats:
    lines.extend(format_stats_text(stats, thresholds, model_metrics))

  lines.extend(format_hunk_scores_text(payload["hunks"], all_hunks=True))

  flagged = [h for h in payload["hunks"] if h["verdict"] != "preferred_edit"]
  if flagged:
    lines.extend(["", "要確認 hunk（抜粋）:"])
    for hunk in flagged:
      loc = ""
      if hunk.get("line_new"):
        loc = f"L{hunk['line_new']} "
      lines.append(
        f"  #{hunk['index']} {loc}"
        f"edit={hunk['score_edit']:.4f} base={hunk['score_base']:.4f} "
        f"margin={hunk['margin']:+.4f} [{hunk['verdict']}]"
      )
      detail_line = format_pref_detail_line(hunk.get("pref_detail") or {})
      if detail_line:
        lines.append(detail_line)
      lines.append(f"    - {hunk['source_preview']}")
      lines.append(f"    + {hunk['edited_preview']}")
  elif not payload["hunks"]:
    lines.extend(["", "該当 hunk なし。"])
  return "\n".join(lines)


def format_markdown(payload: dict) -> str:
  meta = payload["meta"]
  summary = payload["summary"]
  stats = payload.get("statistics", {})
  thresholds = payload.get("thresholds", {})
  model_metrics = payload.get("model_metrics", {})
  lines = [
    f"# 推敲選好チェック: `{meta['file']}`",
    "",
    "| 項目 | 値 |",
    "|------|-----|",
    f"| リポジトリ | `{meta['repo']}` |",
    f"| 比較 | `{meta['compare_label']}` |",
    f"| diff | `{meta['diff_mode']}` |",
    f"| モデル | `{meta['model_dir']}` |",
    "",
    "## サマリー",
    "",
    "| 判定 | 件数 |",
    "|------|------|",
    f"| edit 優位 | {summary['preferred_edit']} |",
    f"| 要確認 | {summary['uncertain']} |",
    f"| 下書き優位 | {summary['preferred_base']} |",
    f"| **合計 hunk** | **{summary['total']}** |",
  ]

  if stats:
    lines.extend(
      [
        "",
        "## 数値サマリー",
        "",
        "| 指標 | 値 |",
        "|------|-----|",
        f"| mean margin (edit−base) | `{stats['mean_margin']:+.4f}` |",
        f"| margin 範囲 | `[{stats['min_margin']:+.4f}, {stats['max_margin']:+.4f}]` |",
        f"| mean score_edit | `{stats['mean_score_edit']:.4f}` |",
        f"| mean score_base | `{stats['mean_score_base']:.4f}` |",
        f"| edit 優位 hunk 比率 (margin>0) | `{stats['edit_win_rate']:.1%}` |",
        "",
        "### 判定閾値",
        "",
        f"- `preferred_edit`: margin ≥ `{thresholds.get('uncertain', 0):+.2f}`",
        f"- `uncertain`: |margin| < `{thresholds.get('uncertain', 0):.2f}`",
        f"- `preferred_base`: margin ≤ `{-thresholds.get('uncertain', 0):+.2f}`",
        f"- `reject_edit`: margin ≤ `{-thresholds.get('reject', 0):+.2f}`",
      ]
    )
    if model_metrics:
      acc = model_metrics.get("eval_accuracy")
      n_eval = model_metrics.get("eval_samples")
      embed = model_metrics.get("embedding_model", "")
      lines.extend(
        [
          "",
          "### モデル参考値 (`metrics.json`)",
          "",
          "| 項目 | 値 |",
          "|------|-----|",
          f"| eval_accuracy | `{acc:.4f}` |" if acc is not None else "| eval_accuracy | n/a |",
          f"| eval_samples | `{n_eval}` |" if n_eval is not None else "",
          f"| embedding | `{embed}` |" if embed else "",
        ]
      )

  if payload["hunks"]:
    lines.extend(
      [
        "",
        "## 全 hunk スコア",
        "",
        "| # | 行 | score_edit | score_base | margin | 判定 |",
        "|---|-----|------------|------------|--------|------|",
      ]
    )
    for hunk in payload["hunks"]:
      loc = str(hunk["line_new"]) if hunk.get("line_new") else "—"
      lines.append(
        f"| {hunk['index']} | L{loc} | `{hunk['score_edit']:.4f}` | "
        f"`{hunk['score_base']:.4f}` | `{hunk['margin']:+.4f}` | `{hunk['verdict']}` |"
      )
    lines.extend(
      [
        "",
        "スコアは対称ペア評価の合成値（0–1、高いほどその側が選好）。",
        "margin = score_edit − score_base。",
        "",
        "### 対称評価の内訳（各 hunk）",
        "",
      ]
    )
    for hunk in payload["hunks"]:
      detail = hunk.get("pref_detail") or {}
      if not detail:
        continue
      loc = f"L{hunk['line_new']}" if hunk.get("line_new") else f"#{hunk['index']}"
      lines.append(
        f"- **{loc}**: `(入力,edit,base)` P(edit)={detail.get('prob_a_order_AB', 0):.4f}, "
        f"P(base)={detail.get('prob_b_order_AB', 0):.4f}; "
        f"`(入力,base,edit)` P(先=base)={detail.get('prob_first_is_B_order_BAfirst', 0):.4f}, "
        f"P(後=edit)={detail.get('prob_second_is_A_order_BAfirst', 0):.4f}"
      )
    lines.append("")

  flagged = [h for h in payload["hunks"] if h["verdict"] != "preferred_edit"]
  if flagged:
    lines.extend(["", "## 要確認 hunk（本文）", ""])
    for hunk in flagged:
      loc = f"（新 L{hunk['line_new']}）" if hunk.get("line_new") else ""
      lines.extend(
        [
          f"### #{hunk['index']} {loc} — `{hunk['verdict']}`",
          "",
          f"- edit: `{hunk['score_edit']:.4f}` / base: `{hunk['score_base']:.4f}` / margin: `{hunk['margin']:+.4f}`",
          "",
          "修正前:",
          "",
          f"> {hunk['source_preview']}",
          "",
          "修正後:",
          "",
          f"> {hunk['edited_preview']}",
          "",
        ]
      )
  return "\n".join(lines)


def run_check(params: dict) -> dict:
  """Prefer warm daemon; fall back to one-shot local batch inference."""
  if not os.environ.get("JA_TECH_EDIT_SCORE_NO_DAEMON", "").strip():
    try:
      from pref_client import check_via_daemon

      return check_via_daemon(params)
    except (FileNotFoundError, TimeoutError, OSError, RuntimeError):
      pass

  from check_revision_core import build_check_payload, default_model_dir
  from pref_runtime import load_pref_model

  model_dir = Path(params.get("model") or default_model_dir())
  if not (model_dir / "model.joblib").is_file():
    raise SystemExit(
      f"model not found: {model_dir / 'model.joblib'}\n"
      "expected bundled model at outputs/pref-static/; run `make train` to rebuild"
    )
  loaded = load_pref_model(model_dir)
  return build_check_payload(params, loaded)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Check whether edit-branch hunks align with ja-tech-edit-score static model"
  )
  parser.add_argument("file", help="target file (path relative to cwd or absolute)")
  parser.add_argument("--repo", default="", help="git repo root (default: infer from file)")
  parser.add_argument("--base", default="", help="base/draft branch (default: infer from edit/*)")
  parser.add_argument("--edit", default="", help="current branch label (default: current branch)")
  parser.add_argument(
    "--committed",
    action="store_true",
    help="compare committed base..edit branches only (default: base branch vs working tree)",
  )
  parser.add_argument("--project-id", default="", help="project id label (default: repo name)")
  parser.add_argument("--model", default="", help="trained model directory")
  parser.add_argument(
    "--format",
    choices=("text", "json", "markdown"),
    default="text",
    help="output format",
  )
  parser.add_argument(
    "--only-flagged",
    action="store_true",
    help="include only hunks that are not preferred_edit (json/markdown hunk list)",
  )
  parser.add_argument(
    "--no-daemon",
    action="store_true",
    help="load model in-process (skip warm daemon; slower cold start)",
  )
  parser.add_argument(
    "--uncertain-threshold",
    type=float,
    default=0.12,
    help="margin below this (but above -threshold) counts as uncertain",
  )
  parser.add_argument(
    "--reject-threshold",
    type=float,
    default=0.25,
    help="margin below -this counts as reject_edit",
  )
  args = parser.parse_args()

  if args.no_daemon:
    os.environ["JA_TECH_EDIT_SCORE_NO_DAEMON"] = "1"

  file_path = Path(args.file).expanduser()
  if not file_path.is_absolute():
    file_path = (Path.cwd() / file_path).resolve()

  params = {
    "file": str(file_path),
    "repo": args.repo,
    "base": args.base,
    "edit": args.edit,
    "committed": args.committed,
    "project_id": args.project_id,
    "model": args.model,
    "only_flagged": args.only_flagged,
    "uncertain_threshold": args.uncertain_threshold,
    "reject_threshold": args.reject_threshold,
  }

  try:
    payload = run_check(params)
  except ValueError as exc:
    raise SystemExit(str(exc)) from exc

  if args.format == "json":
    print(json.dumps(payload, ensure_ascii=False, indent=2))
  elif args.format == "markdown":
    print(format_markdown(payload))
  else:
    print(format_text(payload))


if __name__ == "__main__":
  main()
