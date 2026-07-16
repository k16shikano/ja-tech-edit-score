#!/usr/bin/env python3
"""編集モデル評価の採点（手元 CPU）。

DOK で生成した adapter.jsonl / base_norms.jsonl などを突き合わせ、
BT 報酬 s(draft, candidate) と長さ比で比較する。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def strip_thinking(text: str) -> str:
  """生成時に漏れ残った Qwen3 思考ブロックを採点前に落とす。"""
  if not text:
    return text
  text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE)
  text = re.sub(r"<think>[\s\S]*\Z", "", text, flags=re.IGNORECASE)
  text = text.replace("<think>", "").replace("</think>", "")
  return text.strip()


def strip_meta_revision(text: str) -> str:
  """プロンプト漏れのメタ前置き・解説を落とす（保険）。"""
  if not text:
    return text
  t = text.strip()
  m = re.search(r"\*\*推敲後[^*]*\*\*[：:]*\s*\n+(.*)", t, flags=re.DOTALL)
  if m:
    t = m.group(1).strip()
  if re.match(r"以下は[^\n]{0,200}推敲", t):
    parts = re.split(r"\n---\n+", t, maxsplit=1)
    if len(parts) == 2:
      t = parts[1].strip()
    else:
      parts = re.split(r"\n\s*\n", t, maxsplit=1)
      if len(parts) == 2:
        t = parts[1].strip()
  for pat in (
    r"\n---\s*\n+(?:\#\#?\#?\s*)?(?:説明|解説|根拠|変更点|推敲の根拠)",
    r"\n\*\*(?:説明|解説|根拠|変更点|推敲の根拠)[^*]*\*\*",
    r"\n(?:\#\#?\#?\s*)?(?:説明|解説|根拠|変更点)[：:]",
  ):
    m = re.search(pat, t)
    if m:
      t = t[: m.start()].strip()
  t = re.sub(r"^\*\*推敲後[^*]*\*\*[：:]*\s*\n*", "", t)
  t = re.sub(r"^推敲後の(?:文|本文)[：:]\s*\n*", "", t)
  t = re.sub(r"(?:\n\s*)?---\s*$", "", t)
  return t.strip()


def load_gens(path: Path) -> dict[str, dict]:
  out: dict[str, dict] = {}
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if "generated" in row and isinstance(row["generated"], str):
        cleaned = strip_meta_revision(strip_thinking(row["generated"]))
        row = {**row, "generated": cleaned}
      out[str(row["id"])] = row
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--eval-dir",
    default="outputs/edit-sft-eval",
    help="adapter.jsonl 等があるディレクトリ",
  )
  parser.add_argument("--bt-model", default="outputs/pref-bt")
  parser.add_argument(
    "--modes",
    default="adapter,base_norms,gold",
    help="比較する候補。gold は人間推敲。カンマ区切り",
  )
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--report", default="")
  args = parser.parse_args()

  from pref_bt_runtime import load_bt_model, score_candidates_bt

  eval_dir = Path(args.eval_dir)
  mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
  tables: dict[str, dict[str, dict]] = {}
  for mode in mode_names:
    if mode == "gold":
      continue
    path = eval_dir / f"{mode}.jsonl"
    if not path.is_file():
      raise SystemExit(f"missing {path}")
    tables[mode] = load_gens(path)

  # id の共通集合
  id_sets = [set(t.keys()) for t in tables.values()]
  common_ids = sorted(set.intersection(*id_sets)) if id_sets else []
  if not common_ids:
    raise SystemExit("no common ids across generation files")

  # gold / draft はいずれかのファイルから
  first = next(iter(tables.values()))
  loaded = load_bt_model(Path(args.bt_model))

  per_id: list[dict] = []
  score_lists: dict[str, list[float]] = defaultdict(list)
  len_ratio_lists: dict[str, list[float]] = defaultdict(list)

  for eid in common_ids:
    draft = first[eid]["draft"]
    gold = first[eid]["gold"]
    project_id = first[eid].get("project_id", "")
    candidates: list[str] = []
    labels: list[str] = []
    for mode in mode_names:
      if mode == "gold":
        candidates.append(gold)
        labels.append("gold")
      else:
        candidates.append(tables[mode][eid]["generated"])
        labels.append(mode)
    scores = score_candidates_bt(
      loaded, draft, candidates, batch_size=args.batch_size
    )
    row = {
      "id": eid,
      "project_id": project_id,
      "draft_chars": len(draft),
      "scores": {},
      "length_ratio": {},
    }
    for label, text, score in zip(labels, candidates, scores, strict=True):
      row["scores"][label] = float(score)
      ratio = len(text) / max(len(draft), 1)
      row["length_ratio"][label] = float(ratio)
      score_lists[label].append(float(score))
      len_ratio_lists[label].append(float(ratio))
    # 勝敗（adapter vs base_norms があれば）
    if "adapter" in row["scores"] and "base_norms" in row["scores"]:
      a, b = row["scores"]["adapter"], row["scores"]["base_norms"]
      if a > b:
        row["adapter_vs_base_norms"] = "adapter"
      elif b > a:
        row["adapter_vs_base_norms"] = "base_norms"
      else:
        row["adapter_vs_base_norms"] = "tie"
    per_id.append(row)

  summary: dict = {
    "n": len(common_ids),
    "modes": mode_names,
    "mean_score": {k: float(np.mean(v)) for k, v in score_lists.items()},
    "mean_length_ratio": {k: float(np.mean(v)) for k, v in len_ratio_lists.items()},
  }
  if "adapter" in score_lists and "base_norms" in score_lists:
    wins = sum(1 for r in per_id if r.get("adapter_vs_base_norms") == "adapter")
    losses = sum(1 for r in per_id if r.get("adapter_vs_base_norms") == "base_norms")
    ties = sum(1 for r in per_id if r.get("adapter_vs_base_norms") == "tie")
    summary["adapter_vs_base_norms"] = {
      "wins": wins,
      "losses": losses,
      "ties": ties,
      "win_rate": wins / max(wins + losses, 1),
    }
  # プロジェクト別
  by_proj: dict[str, list[dict]] = defaultdict(list)
  for r in per_id:
    by_proj[str(r["project_id"])].append(r)
  summary["by_project"] = {}
  for pid, rows in sorted(by_proj.items()):
    entry: dict = {"n": len(rows)}
    if "adapter" in score_lists and "base_norms" in score_lists:
      w = sum(1 for r in rows if r.get("adapter_vs_base_norms") == "adapter")
      l = sum(1 for r in rows if r.get("adapter_vs_base_norms") == "base_norms")
      entry["adapter_win_rate"] = w / max(w + l, 1)
    summary["by_project"][pid] = entry

  report = {"summary": summary, "rows": per_id}
  report_path = Path(args.report) if args.report else eval_dir / "score_report.json"
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  md_path = report_path.with_suffix(".md")
  lines = [
    "# Edit-SFT evaluation (BT reward)",
    "",
    f"- n: {summary['n']}",
    f"- mean_score: {json.dumps(summary['mean_score'], ensure_ascii=False)}",
    f"- mean_length_ratio: {json.dumps(summary['mean_length_ratio'], ensure_ascii=False)}",
  ]
  if "adapter_vs_base_norms" in summary:
    avb = summary["adapter_vs_base_norms"]
    lines.append(
      f"- adapter vs base_norms: wins={avb['wins']} losses={avb['losses']} "
      f"ties={avb['ties']} win_rate={avb['win_rate']:.3f}"
    )
  lines.append("")
  lines.append("| project | n | adapter win rate |")
  lines.append("|---|---:|---:|")
  for pid, entry in summary["by_project"].items():
    wr = entry.get("adapter_win_rate")
    wr_s = f"{wr:.3f}" if wr is not None else "-"
    lines.append(f"| {pid} | {entry['n']} | {wr_s} |")
  lines.append("")
  md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  print(json.dumps(summary, ensure_ascii=False, indent=2))
  print(f"wrote {report_path}")
  print(f"wrote {md_path}")


if __name__ == "__main__":
  main()
