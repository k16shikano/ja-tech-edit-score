#!/usr/bin/env python3
"""A 1716 の推敲動き分布と、C 56 での典型性 vs s_den。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  draft_from_row,
  ead_reports,
  fmt_rate,
  load_jsonl,
  md_table,
  rate_summary,
  repo_root,
  surface_features,
  write_json,
)

COMPARE_TYPE = "5_gold_vs_adapter_selected"
NOTE_BLOCK_RE = re.compile(r"^>\s*(note|コラム|NOTE)", re.M)
FOOTNOTE_RE = re.compile(r"\[\^[^\]]+\]")


def note_added(draft: str, cand: str) -> bool:
  note_block = bool(NOTE_BLOCK_RE.search(cand)) and not bool(NOTE_BLOCK_RE.search(draft))
  footnote = bool(set(FOOTNOTE_RE.findall(cand)) - set(FOOTNOTE_RE.findall(draft)))
  return note_block or footnote


def motion_features(draft: str, cand: str) -> dict[str, Any]:
  feats = surface_features(draft, cand)
  feats["note_added"] = note_added(draft, cand)
  return feats


def delta_cat(delta: float) -> str:
  if delta < 0:
    return "shrink"
  if delta > 0:
    return "expand"
  return "neutral"


def ed_cat(value: float, edges: tuple[float, float]) -> str:
  if value <= edges[0]:
    return "low"
  if value <= edges[1]:
    return "mid"
  return "high"


def motion_cell(feats: dict[str, Any], ed_edges: tuple[float, float]) -> tuple[str, str, bool]:
  return (
    delta_cat(float(feats["delta_chars"])),
    ed_cat(float(feats["edit_distance_norm"]), ed_edges),
    bool(feats["note_added"]),
  )


def corpus_profile(train_rows: list[dict]) -> dict[str, Any]:
  items: list[dict[str, Any]] = []
  for row in train_rows:
    draft = draft_from_row(row)
    cand = str(row["messages"][1]["content"])
    items.append(motion_features(draft, cand))

  deltas = np.asarray([x["delta_chars"] for x in items], dtype=np.float64)
  eds = np.asarray([x["edit_distance_norm"] for x in items], dtype=np.float64)
  notes = np.asarray([1.0 if x["note_added"] else 0.0 for x in items], dtype=np.float64)
  ed_edges = tuple(float(x) for x in np.quantile(eds, [1 / 3, 2 / 3]))

  cells = Counter(motion_cell(x, ed_edges) for x in items)
  mode_cell, mode_count = cells.most_common(1)[0]

  cell_prob = {k: v / len(items) for k, v in cells.items()}

  def log_typicality(feats: dict[str, Any]) -> float:
    return float(np.log(cell_prob[motion_cell(feats, ed_edges)]))

  mu = np.array([deltas.mean(), eds.mean(), notes.mean()])
  sd = np.array([
    max(deltas.std(ddof=1), 1e-9),
    max(eds.std(ddof=1), 1e-9),
    max(notes.std(ddof=1), 1e-9),
  ])

  def zdist(feats: dict[str, Any]) -> float:
    x = np.array([
      float(feats["delta_chars"]),
      float(feats["edit_distance_norm"]),
      1.0 if feats["note_added"] else 0.0,
    ])
    return float(np.linalg.norm((x - mu) / sd))

  return {
    "n": len(items),
    "delta_chars": {
      "median": float(np.median(deltas)),
      "mean": float(deltas.mean()),
      "p25": float(np.quantile(deltas, 0.25)),
      "p75": float(np.quantile(deltas, 0.75)),
      "shrink_n": int(sum(1 for x in items if delta_cat(x["delta_chars"]) == "shrink")),
      "expand_n": int(sum(1 for x in items if delta_cat(x["delta_chars"]) == "expand")),
      "neutral_n": int(sum(1 for x in items if delta_cat(x["delta_chars"]) == "neutral")),
    },
    "edit_distance_norm": {
      "median": float(np.median(eds)),
      "mean": float(eds.mean()),
      "p25": float(np.quantile(eds, 0.25)),
      "p75": float(np.quantile(eds, 0.75)),
      "tertile_edges": list(ed_edges),
    },
    "note_added_n": int(notes.sum()),
    "note_added_rate": float(notes.mean()),
    "mode_cell": {
      "delta_cat": mode_cell[0],
      "ed_cat": mode_cell[1],
      "note_added": mode_cell[2],
      "count": mode_count,
      "rate": mode_count / len(items),
    },
    "top_cells": [
      {
        "delta_cat": cell[0],
        "ed_cat": cell[1],
        "note_added": cell[2],
        "count": count,
        "rate": count / len(items),
      }
      for cell, count in cells.most_common(10)
    ],
    "ed_edges": ed_edges,
    "log_typicality": log_typicality,
    "zdist": zdist,
    "modal_axis": {
      "delta_cat": Counter(delta_cat(x["delta_chars"]) for x in items).most_common(1)[0][0],
      "ed_cat": Counter(ed_cat(x["edit_distance_norm"], ed_edges) for x in items).most_common(1)[0][0],
      "note_added": False,
    },
  }


def gold_adapter_texts(pair: dict) -> tuple[str, str]:
  if pair.get("a_source") == "gold":
    return str(pair["a_text"]), str(pair["b_text"])
  return str(pair["b_text"]), str(pair["a_text"])


def source_name(pair: dict, side: str) -> str:
  if side == "a":
    return str(pair.get("a_source") or "a")
  return str(pair.get("b_source") or "b")


def pick_by_score(score_gold: float, score_adapter: float) -> str:
  if score_gold > score_adapter:
    return "gold"
  if score_adapter > score_gold:
    return "adapter_selected"
  return "tie"


def build_c_rows(
  profile: dict[str, Any],
  pairs: list[dict],
  measure_items: list[dict],
) -> list[dict[str, Any]]:
  by_pair = {str(x["pair_id"]): x for x in measure_items if x.get("agree") is not None}
  log_typ = profile["log_typicality"]
  zdist = profile["zdist"]
  ed_edges = profile["ed_edges"]

  rows: list[dict[str, Any]] = []
  for pair in pairs:
    pair_id = str(pair["pair_id"])
    measure = by_pair.get(pair_id)
    if not measure:
      continue

    draft = str(pair["context_draft"])
    gold, adapter = gold_adapter_texts(pair)
    fg = motion_features(draft, gold)
    fa = motion_features(draft, adapter)

    tg, ta = log_typ(fg), log_typ(fa)
    zg, za = zdist(fg), zdist(fa)

    if tg == ta:
      typ_joint = "tie"
    else:
      typ_joint = "gold" if tg > ta else "adapter_selected"

    if zg == za:
      typ_z = "tie"
    else:
      typ_z = "gold" if zg < za else "adapter_selected"

    sden_pick = source_name(pair, "a" if float(measure["score_a"]) > float(measure["score_b"]) else "b")
    human_pick = source_name(pair, str(measure["human_pref"]))

    sden_side_feats = fg if sden_pick == "gold" else fa
    rows.append(
      {
        "pair_id": pair_id,
        "item_id": str(pair.get("item_id") or ""),
        "gold_cell": motion_cell(fg, ed_edges),
        "adapter_cell": motion_cell(fa, ed_edges),
        "gold": {
          "delta_chars": fg["delta_chars"],
          "edit_distance_norm": fg["edit_distance_norm"],
          "note_added": fg["note_added"],
          "log_typicality": tg,
          "zdist": zg,
        },
        "adapter": {
          "delta_chars": fa["delta_chars"],
          "edit_distance_norm": fa["edit_distance_norm"],
          "note_added": fa["note_added"],
          "log_typicality": ta,
          "zdist": za,
        },
        "typical_pick_joint": typ_joint,
        "typical_pick_zdist": typ_z,
        "s_den_pick": sden_pick,
        "human_pick": human_pick,
        "s_den_margin": abs(float(measure["score_a"]) - float(measure["score_b"])),
        "human_agree": bool(measure["agree"]),
        "s_den_side": {
          "delta_cat": delta_cat(float(sden_side_feats["delta_chars"])),
          "ed_cat": ed_cat(float(sden_side_feats["edit_distance_norm"]), ed_edges),
          "note_added": bool(sden_side_feats["note_added"]),
        },
      }
    )
  return rows


def match_rate(rows: list[dict], left: str, right: str) -> dict[str, Any]:
  comparable = [r for r in rows if r[left] not in ("tie", "") and r[right] not in ("tie", "")]
  agree = sum(1 for r in comparable if r[left] == r[right])
  return rate_summary(agree, len(comparable))


def side_preference(rows: list[dict], profile: dict[str, Any]) -> dict[str, Any]:
  ed_edges = profile["ed_edges"]
  modal = profile["modal_axis"]
  expand_n = 0
  higher_ed_n = 0
  modal_delta_n = 0
  modal_ed_n = 0
  modal_note_n = 0

  for row in rows:
    side = row["s_den_side"]
    if side["delta_cat"] == "expand":
      expand_n += 1
    if side["delta_cat"] == modal["delta_cat"]:
      modal_delta_n += 1
    if side["ed_cat"] == modal["ed_cat"]:
      modal_ed_n += 1
    if side["note_added"] == modal["note_added"]:
      modal_note_n += 1

    gold_ed = float(row["gold"]["edit_distance_norm"])
    ad_ed = float(row["adapter"]["edit_distance_norm"])
    if row["s_den_pick"] == "gold" and gold_ed > ad_ed:
      higher_ed_n += 1
    if row["s_den_pick"] == "adapter_selected" and ad_ed > gold_ed:
      higher_ed_n += 1

  n = len(rows)
  return {
    "expand_rate": rate_summary(expand_n, n),
    "higher_edit_distance_rate": rate_summary(higher_ed_n, n),
    "modal_delta_rate": rate_summary(modal_delta_n, n),
    "modal_ed_rate": rate_summary(modal_ed_n, n),
    "modal_note_rate": rate_summary(modal_note_n, n),
  }


def build_report(payload: dict[str, Any]) -> str:
  corp = payload["corpus"]
  comp = payload["comparisons"]
  pref = payload["s_den_side_preferences"]
  lines = [
    "# ead-typicality-c",
    "",
    "A 1716（`data/edit_sft_all/train.jsonl`）の人間推敲の表面特徴分布と、C 56 で gold / adapter のどちらがコーパス典型に近いかを見る。",
    "s_den は `ead-h1-c-measure.json` の $s_\\mathrm{sum}$ 符号。",
    "",
    "## A 1716 分布",
    "",
    md_table(
      ["量", "値"],
      [
        ["件数", str(corp["n"])],
        ["Δ文字数 中央値", f"{corp['delta_chars']['median']:.1f}"],
        ["Δ文字数 平均", f"{corp['delta_chars']['mean']:.2f}"],
        ["Δ文字数 p25 / p75", f"{corp['delta_chars']['p25']:.1f} / {corp['delta_chars']['p75']:.1f}"],
        ["shrink / expand / neutral", f"{corp['delta_chars']['shrink_n']} / {corp['delta_chars']['expand_n']} / {corp['delta_chars']['neutral_n']}"],
        ["edit_distance_norm 中央値", f"{corp['edit_distance_norm']['median']:.4f}"],
        ["edit_distance_norm p25 / p75", f"{corp['edit_distance_norm']['p25']:.4f} / {corp['edit_distance_norm']['p75']:.4f}"],
        ["注記追加", f"{corp['note_added_n']}/{corp['n']} = {corp['note_added_rate']:.3f}"],
      ],
    ),
    "",
    "3 軸（Δ符号・edit_distance  tertile・注記追加）の最頻セル:",
    "",
    f"- ({corp['mode_cell']['delta_cat']}, {corp['mode_cell']['ed_cat']}, note={corp['mode_cell']['note_added']}) "
    f"= {corp['mode_cell']['count']}/{corp['n']} = {corp['mode_cell']['rate']:.3f}",
    "",
    "## C 56: 典型性 vs s_den",
    "",
    "典型性（joint）= 3 軸セルのコーパス相対頻度の log が高い側。",
    "典型性（zdist）= Δ文字数・edit_distance_norm・注記有無を A で標準化した距離が小さい側。",
    "",
    md_table(
      ["比較", "一致"],
      [
        ["s_den vs 人手", fmt_rate(comp["s_den_vs_human"])],
        ["典型性 joint vs 人手", fmt_rate(comp["typical_joint_vs_human"])],
        ["典型性 zdist vs 人手", fmt_rate(comp["typical_zdist_vs_human"])],
        ["s_den vs 典型性 joint", fmt_rate(comp["s_den_vs_typical_joint"])],
        ["s_den vs 典型性 zdist", fmt_rate(comp["s_den_vs_typical_zdist"])],
      ],
    ),
    "",
    "s_den が選んだ側の表面特徴（C 56）:",
    "",
    md_table(
      ["軸", "コーパス最頻 / 傾向", "s_den 側が一致"],
      [
        ["Δ符号 = expand", f"コーパス最頻 = {corp['modal_axis']['delta_cat']}", fmt_rate(pref["expand_rate"])],
        ["edit_distance が gold / adapter 内で大きい方", "—", fmt_rate(pref["higher_edit_distance_rate"])],
        [f"Δ符号 = {corp['modal_axis']['delta_cat']}", "—", fmt_rate(pref["modal_delta_rate"])],
        [f"ed = {corp['modal_axis']['ed_cat']}", "—", fmt_rate(pref["modal_ed_rate"])],
        [f"note_added = {corp['modal_axis']['note_added']}", "—", fmt_rate(pref["modal_note_rate"])],
      ],
    ),
    "",
    "## 読み",
    "",
    payload["interpretation"],
  ]
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--train-jsonl", type=Path, default=Path("data/edit_sft_all/train.jsonl"))
  parser.add_argument("--pairs", type=Path, default=Path("data/blind_eval/pairs_gold_vs_adapter_selected.jsonl"))
  parser.add_argument("--measure-json", type=Path, default=Path("outputs/ead/reports/ead-h1-c-measure.json"))
  args = parser.parse_args()

  root = args.root.resolve()
  train_rows = load_jsonl(root / args.train_jsonl)
  if len(train_rows) != 1716:
    raise SystemExit(f"expected 1716 train rows, got {len(train_rows)}")

  profile = corpus_profile(train_rows)
  pairs = [
    p for p in load_jsonl(root / args.pairs) if p.get("compare_type") == COMPARE_TYPE
  ]
  measure = json.loads((root / args.measure_json).read_text(encoding="utf-8"))
  rows = build_c_rows(profile, pairs, measure["items"])
  if len(rows) != 56:
    raise SystemExit(f"expected 56 comparable C rows, got {len(rows)}")

  comparisons = {
    "s_den_vs_human": match_rate(rows, "s_den_pick", "human_pick"),
    "typical_joint_vs_human": match_rate(rows, "typical_pick_joint", "human_pick"),
    "typical_zdist_vs_human": match_rate(rows, "typical_pick_zdist", "human_pick"),
    "s_den_vs_typical_joint": match_rate(rows, "s_den_pick", "typical_pick_joint"),
    "s_den_vs_typical_zdist": match_rate(rows, "s_den_pick", "typical_pick_zdist"),
  }
  preferences = side_preference(rows, profile)

  s_den_typical_joint = comparisons["s_den_vs_typical_joint"]["rate"] or 0.0
  if s_den_typical_joint >= 0.7:
    interpretation = (
      "s_den がコーパス典型側を選ぶ割合が高い。"
      "表面特徴の典型性と s_den の符号が同方向である。"
    )
  elif s_den_typical_joint <= 0.4:
    interpretation = (
      f"s_den が joint 典型側を選ぶのは {comparisons['s_den_vs_typical_joint']['k']}/{comparisons['s_den_vs_typical_joint']['n']} と低い。"
      "s_den = 表面典型性の復元、とは読めない。"
      f"s_den は expand 側（{preferences['expand_rate']['k']}/{preferences['expand_rate']['n']}）や "
      f"edit_distance が大きい側（{preferences['higher_edit_distance_rate']['k']}/{preferences['higher_edit_distance_rate']['n']}）を取りやすく、"
      "コーパス最頻の shrink とはずれる。"
      f"人手一致 {comparisons['s_den_vs_human']['k']}/{comparisons['s_den_vs_human']['n']} は "
      f"joint 典型 {comparisons['typical_joint_vs_human']['k']}/{comparisons['typical_joint_vs_human']['n']} を上回る。"
      "s_den は「いつ典型動作を使うか」だけでなく、典型性以外の信号も混ざっている。"
    )
  else:
    interpretation = "s_den と表面典型性の一致は中程度。追加の層別が要る。"

  payload = {
    "corpus_source": str(args.train_jsonl),
    "corpus": {
      k: profile[k]
      for k in (
        "n",
        "delta_chars",
        "edit_distance_norm",
        "note_added_n",
        "note_added_rate",
        "mode_cell",
        "top_cells",
        "modal_axis",
      )
    },
    "comparisons": comparisons,
    "s_den_side_preferences": preferences,
    "interpretation": interpretation,
    "items": rows,
  }

  reports = ead_reports()
  write_json(reports / "ead-typicality-c.json", payload)
  (reports / "ead-typicality-c.md").write_text(build_report(payload), encoding="utf-8")
  print(
    json.dumps(
      {
        "wrote": str(reports / "ead-typicality-c.md"),
        "s_den_vs_typical_joint": fmt_rate(comparisons["s_den_vs_typical_joint"]),
        "s_den_vs_human": fmt_rate(comparisons["s_den_vs_human"]),
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
