#!/usr/bin/env python3
"""学習済み段落遷移モデルで、下書き vs 人間編集の節ペアを採点する。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer

from extract_paragraph_transitions import filtered_paragraphs
from pref_static_utils import encode_texts, load_jsonl, normalize_truncate_dim
from train_paragraph_transition import assemble_transition_features, collect_unique_paragraph_texts

# 修正前（評価時フィルタ未適用）の held-out 結果。比較用に固定。
BASELINE_BEFORE_FILTER_FIX = {
  "note": "評価時に \\n\\n 分割のみで採点していた版（2026-07-25 初回実行）",
  "composition_changed": {
    "mean_win_rate": 0.46601941747572817,
    "min_win_rate": 0.2912621359223301,
  },
  "all": {
    "mean_win_rate": 0.4528301886792453,
    "min_win_rate": 0.2830188679245283,
  },
}


def boundary_scores(classifier, text_to_embedding: dict[str, np.ndarray], paras: list[str]) -> list[float]:
  if len(paras) < 2:
    return []
  classes = list(classifier.named_steps["classifier"].classes_)
  pos_idx = classes.index(1) if 1 in classes else 0
  scores: list[float] = []
  for i in range(len(paras) - 1):
    feat = assemble_transition_features(
      text_to_embedding[paras[i]],
      text_to_embedding[paras[i + 1]],
    ).reshape(1, -1)
    prob = float(classifier.predict_proba(feat)[0, pos_idx])
    scores.append(prob)
  return scores


def document_scores(boundary: list[float]) -> dict[str, float | None]:
  if not boundary:
    return {"mean": None, "min": None, "boundaries": 0}
  return {
    "mean": float(sum(boundary) / len(boundary)),
    "min": float(min(boundary)),
    "boundaries": len(boundary),
  }


def paragraph_sequence_changed(source_paras: list[str], edited_paras: list[str]) -> bool:
  return source_paras != edited_paras


def composition_changed(row: dict, source_paras: list[str], edited_paras: list[str]) -> bool:
  meta = row.get("meta", {})
  pc_src = meta.get("paragraph_count_source")
  pc_ed = meta.get("paragraph_count_edited")
  if pc_src is not None and pc_ed is not None and int(pc_src) != int(pc_ed):
    return True
  return paragraph_sequence_changed(source_paras, edited_paras)


def summarize_group(items: list[dict], *, metric: str) -> dict:
  scored = [
    x
    for x in items
    if x[f"source_{metric}"] is not None and x[f"edited_{metric}"] is not None
  ]
  wins = [x for x in scored if x[f"edited_{metric}"] > x[f"source_{metric}"]]
  diffs = [x[f"delta_{metric}"] for x in scored if x[f"delta_{metric}"] is not None]
  return {
    "count": len(items),
    "scored_count": len(scored),
    "win_rate": float(len(wins) / len(scored)) if scored else float("nan"),
    "mean_delta": float(statistics.mean(diffs)) if diffs else float("nan"),
    "median_delta": float(statistics.median(diffs)) if diffs else float("nan"),
  }


def summarize_boundary_mismatch(items: list[dict]) -> dict:
  scored = [x for x in items if x["source_boundaries"] > 0 and x["edited_boundaries"] > 0]
  same = [x for x in scored if x["same_boundary_count"]]
  different = [x for x in scored if not x["same_boundary_count"]]
  return {
    "count": len(items),
    "scored_count": len(scored),
    "same_boundary_count": len(same),
    "different_boundary_count": len(different),
  }


def pct(value: float) -> str:
  if value != value:
    return "—"
  return f"{value:.1%}"


def render_markdown(report: dict) -> str:
  baseline = report["baseline_before_filter_fix"]
  mismatch = report["boundary_mismatch"]
  lines = [
    "# 段落遷移モデル: 編集前後の検証",
    "",
    report["summary"],
    "",
    "## 境界数の偏り",
    "",
    (
      "文書スコアの最小値は、境界数が多いほど機械的に下がりやすい。"
      "下書きと編集後で境界数が異なると、最小値指標の比較は不公平になりうる。"
    ),
    "",
    f"- (a) 段落構成変化: 境界数同じ {mismatch['composition_changed']['same_boundary_count']} / "
    f"異なる {mismatch['composition_changed']['different_boundary_count']} "
    f"（採点可能 {mismatch['composition_changed']['scored_count']}）",
    f"- (b) 全ペア: 境界数同じ {mismatch['all']['same_boundary_count']} / "
    f"異なる {mismatch['all']['different_boundary_count']} "
    f"（採点可能 {mismatch['all']['scored_count']}）",
    "",
    "## 修正前後の勝率比較",
    "",
    "評価時の段落フィルタを学習時と揃える修正前後。",
    "",
    "| グループ | 指標 | 修正前 | 修正後（全ペア） | 修正後（境界数同じ） |",
    "|---|---|---:|---:|---:|",
  ]

  for group_key, group_label in (
    ("composition_changed", "(a) 構成変化"),
    ("all", "(b) 全ペア"),
  ):
    after_all = report["groups"][group_key]
    after_same = report["groups_same_boundary_count"][group_key]
    baseline_group = baseline[group_key]
    lines.append(
      f"| {group_label} | 平均 | {pct(baseline_group['mean_win_rate'])} | "
      f"{pct(after_all['mean']['win_rate'])} | {pct(after_same['mean']['win_rate'])} |"
    )
    lines.append(
      f"| {group_label} | 最小 | {pct(baseline_group['min_win_rate'])} | "
      f"{pct(after_all['min']['win_rate'])} | {pct(after_same['min']['win_rate'])} |"
    )

  lines.extend(
    [
      "",
      "## 集計（修正後・全ペア）",
      "",
      "### (a) 段落構成が変わったペア",
      "",
    ]
  )
  a = report["groups"]["composition_changed"]
  lines.extend(
    [
      f"- 件数: {a['mean']['count']}（採点可能: {a['mean']['scored_count']}）",
      f"- 編集後スコア > 下書きスコア（平均）の勝率: {a['mean']['win_rate']:.1%}",
      f"- 平均スコア差（平均指標）: {a['mean']['mean_delta']:.4f}",
      f"- 中央値スコア差（平均指標）: {a['mean']['median_delta']:.4f}",
      f"- 編集後スコア > 下書きスコア（最小）の勝率: {a['min']['win_rate']:.1%}",
      f"- 平均スコア差（最小指標）: {a['min']['mean_delta']:.4f}",
      "",
      "### (b) 全ペア",
      "",
    ]
  )
  b = report["groups"]["all"]
  lines.extend(
    [
      f"- 件数: {b['mean']['count']}（採点可能: {b['mean']['scored_count']}）",
      f"- 編集後スコア > 下書きスコア（平均）の勝率: {b['mean']['win_rate']:.1%}",
      f"- 平均スコア差（平均指標）: {b['mean']['mean_delta']:.4f}",
      f"- 中央値スコア差（平均指標）: {b['mean']['median_delta']:.4f}",
      f"- 編集後スコア > 下書きスコア（最小）の勝率: {b['min']['win_rate']:.1%}",
      f"- 平均スコア差（最小指標）: {b['min']['mean_delta']:.4f}",
      "",
      "## 集計（修正後・境界数同じペアのみ）",
      "",
      "### (a) 段落構成が変わったペア",
      "",
    ]
  )
  a_same = report["groups_same_boundary_count"]["composition_changed"]
  lines.extend(
    [
      f"- 件数: {a_same['mean']['count']}（採点可能: {a_same['mean']['scored_count']}）",
      f"- 編集後スコア > 下書きスコア（平均）の勝率: {a_same['mean']['win_rate']:.1%}",
      f"- 編集後スコア > 下書きスコア（最小）の勝率: {a_same['min']['win_rate']:.1%}",
      "",
      "### (b) 全ペア",
      "",
    ]
  )
  b_same = report["groups_same_boundary_count"]["all"]
  lines.extend(
    [
      f"- 件数: {b_same['mean']['count']}（採点可能: {b_same['mean']['scored_count']}）",
      f"- 編集後スコア > 下書きスコア（平均）の勝率: {b_same['mean']['win_rate']:.1%}",
      f"- 編集後スコア > 下書きスコア（最小）の勝率: {b_same['min']['win_rate']:.1%}",
      "",
      "## 解釈",
      "",
      report["interpretation"],
    ]
  )
  return "\n".join(lines) + "\n"


def build_group_report(items: list[dict]) -> dict:
  return {
    "mean": summarize_group(items, metric="mean"),
    "min": summarize_group(items, metric="min"),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", default="data/examples.section.heldout.jsonl")
  parser.add_argument("--model", default="outputs/paragraph-transition/model.joblib")
  parser.add_argument("--report", default="outputs/paragraph-transition/eval_on_edits.json")
  parser.add_argument("--markdown", default="outputs/paragraph-transition/eval_on_edits.md")
  parser.add_argument("--batch-size", type=int, default=32)
  args = parser.parse_args()

  root = Path(__file__).resolve().parent.parent
  input_path = Path(args.input)
  if not input_path.is_absolute():
    input_path = root / input_path
  model_path = Path(args.model)
  if not model_path.is_absolute():
    model_path = root / model_path
  report_path = Path(args.report)
  if not report_path.is_absolute():
    report_path = root / report_path
  markdown_path = Path(args.markdown)
  if not markdown_path.is_absolute():
    markdown_path = root / markdown_path

  rows = load_jsonl(str(input_path))
  if not rows:
    raise SystemExit("input is empty")

  artifact = load(model_path)
  classifier = artifact["classifier"]
  text_prefix = artifact.get("text_prefix", "文章: ")
  truncate_dim = normalize_truncate_dim(artifact.get("truncate_dim"))
  encoder = SentenceTransformer(
    artifact["sentence_model_name"],
    device="cpu",
    truncate_dim=truncate_dim,
  )
  max_seq_length = artifact.get("max_seq_length")
  if max_seq_length:
    encoder.max_seq_length = int(max_seq_length)

  pseudo_rows: list[dict] = []
  for row in rows:
    for para in filtered_paragraphs(row["source_text"]) + filtered_paragraphs(row["edited_text"]):
      pseudo_rows.append({"text_a": para, "text_b": para})
  unique_texts = collect_unique_paragraph_texts(pseudo_rows)

  embeddings = encode_texts(
    encoder,
    unique_texts,
    batch_size=args.batch_size,
    normalize_embeddings=True,
    text_prefix=text_prefix,
    show_progress_bar=True,
  )
  text_to_embedding = {text: emb for text, emb in zip(unique_texts, embeddings, strict=True)}

  per_pair: list[dict] = []
  for row in rows:
    source_paras = filtered_paragraphs(row["source_text"])
    edited_paras = filtered_paragraphs(row["edited_text"])
    source_boundary = boundary_scores(classifier, text_to_embedding, source_paras)
    edited_boundary = boundary_scores(classifier, text_to_embedding, edited_paras)
    source_doc = document_scores(source_boundary)
    edited_doc = document_scores(edited_boundary)
    same_boundary_count = (
      source_doc["boundaries"] > 0
      and edited_doc["boundaries"] > 0
      and source_doc["boundaries"] == edited_doc["boundaries"]
    )

    item = {
      "id": row["id"],
      "project_id": row.get("project_id"),
      "composition_changed": composition_changed(row, source_paras, edited_paras),
      "same_boundary_count": same_boundary_count,
      "source_mean": source_doc["mean"],
      "edited_mean": edited_doc["mean"],
      "source_min": source_doc["min"],
      "edited_min": edited_doc["min"],
      "delta_mean": (
        None
        if source_doc["mean"] is None or edited_doc["mean"] is None
        else edited_doc["mean"] - source_doc["mean"]
      ),
      "delta_min": (
        None
        if source_doc["min"] is None or edited_doc["min"] is None
        else edited_doc["min"] - source_doc["min"]
      ),
      "source_boundaries": source_doc["boundaries"],
      "edited_boundaries": edited_doc["boundaries"],
    }
    per_pair.append(item)

  comp_changed = [x for x in per_pair if x["composition_changed"]]
  same_boundary = [x for x in per_pair if x["same_boundary_count"]]
  comp_changed_same = [x for x in comp_changed if x["same_boundary_count"]]

  report = {
    "input": str(input_path),
    "model": str(model_path),
    "pairs": len(per_pair),
    "paragraph_filter": "extract_paragraph_transitions.filtered_paragraphs (same as training)",
    "summary": (
      "held-out の節ペアについて、学習時と同じ段落フィルタを通した段落列の"
      "隣接境界ごとに遷移スコア（隣接らしさの確率）を出し、"
      "境界平均・最小で文書スコアを作った。"
    ),
    "baseline_before_filter_fix": BASELINE_BEFORE_FILTER_FIX,
    "boundary_mismatch": {
      "composition_changed": summarize_boundary_mismatch(comp_changed),
      "all": summarize_boundary_mismatch(per_pair),
    },
    "groups": {
      "composition_changed": build_group_report(comp_changed),
      "all": build_group_report(per_pair),
    },
    "groups_same_boundary_count": {
      "composition_changed": build_group_report(comp_changed_same),
      "all": build_group_report(same_boundary),
    },
    "interpretation": (
      "勝率が低くても失敗とは限らない。"
      "隣接性の学習が話題連続性だけを拾っている場合、編集前後でスコアは動きにくい。"
      "最小値指標は境界数の差に引っ張られやすいので、境界数同じペアの勝率も見る。"
    ),
    "items": per_pair,
  }

  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
  markdown_path.write_text(render_markdown(report), encoding="utf-8")

  mismatch = report["boundary_mismatch"]["all"]
  print(
    f"boundary mismatch (all): same={mismatch['same_boundary_count']} "
    f"different={mismatch['different_boundary_count']}"
  )
  for name, group in report["groups"].items():
    same_group = report["groups_same_boundary_count"][name]
    print(f"[{name}] n={group['mean']['count']}")
    print(
      f"  all pairs   mean win={group['mean']['win_rate']:.1%} "
      f"min win={group['min']['win_rate']:.1%}"
    )
    print(
      f"  same bounds mean win={same_group['mean']['win_rate']:.1%} "
      f"min win={same_group['min']['win_rate']:.1%}"
    )
  print(f"report: {report_path}")
  print(f"markdown: {markdown_path}")


if __name__ == "__main__":
  main()
