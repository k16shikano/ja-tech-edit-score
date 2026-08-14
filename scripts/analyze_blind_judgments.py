#!/usr/bin/env python3
"""ブラインド人手判定（工程 6）の集計（工程 7）。

pairs.jsonl と judgments.jsonl を結合し、PLAN.md の判定基準に沿って
推敲モデル・評価器の指標を出す。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from generation_integrity import similarity
from pref_scorer import LoadedScorer, load_scorer


COMPARE_LABELS = {
  "1_draft_vs_base_greedy": "比較1",
  "2_draft_vs_adapter_greedy": "比較2",
  "3_base_vs_adapter_greedy": "比較3",
  "4_draft_vs_adapter_selected": "比較4",
  "5_gold_vs_adapter_selected": "比較5",
  "6_gold_vs_draft": "比較6",
}

# pairs.jsonl / judgments.jsonl の a_source, b_source に入る識別子
SOURCE_LABELS: dict[str, str] = {
  "draft": "下書き",
  "gold": "人間の推敲",
  "base_greedy": "素のモデル・貪欲生成",
  "adapter_greedy": "推敲モデル・貪欲生成",
  "adapter_selected": "推敲モデル・選抜1本",
}

COMPARE_DEFINITIONS: list[tuple[str, str, str, str]] = [
  (
    "比較1",
    "1_draft_vs_base_greedy",
    "下書き",
    "素のモデル・貪欲生成",
  ),
  (
    "比較2",
    "2_draft_vs_adapter_greedy",
    "下書き",
    "推敲モデル・貪欲生成",
  ),
  (
    "比較3",
    "3_base_vs_adapter_greedy",
    "素のモデル・貪欲生成",
    "推敲モデル・貪欲生成",
  ),
  (
    "比較4",
    "4_draft_vs_adapter_selected",
    "下書き",
    "推敲モデル・選抜1本",
  ),
  (
    "比較5",
    "5_gold_vs_adapter_selected",
    "人間の推敲",
    "推敲モデル・選抜1本",
  ),
  (
    "比較6",
    "6_gold_vs_draft",
    "人間の推敲",
    "下書き",
  ),
]


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n <= 0:
    return (float("nan"), float("nan"))
  p = k / n
  denom = 1.0 + z * z / n
  centre = p + z * z / (2.0 * n)
  margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
  return ((centre - margin) / denom, (centre + margin) / denom)


def spearman(xs: list[float], ys: list[float]) -> float | None:
  n = len(xs)
  if n < 2:
    return None

  def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    i = 0
    while i < n:
      j = i
      while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
        j += 1
      avg = (i + j) / 2.0 + 1.0
      for k in range(i, j + 1):
        out[order[k]] = avg
      i = j + 1
    return out

  rx, ry = ranks(xs), ranks(ys)
  mx = sum(rx) / n
  my = sum(ry) / n
  num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
  denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
  deny = math.sqrt(sum((b - my) ** 2 for b in ry))
  if denx == 0 or deny == 0:
    return None
  return num / (denx * deny)


def winner_source(row: dict) -> str | None:
  choice = row.get("choice")
  if choice not in ("a", "b"):
    return None
  return row["a_source"] if choice == "a" else row["b_source"]


def side_flags(row: dict, source: str) -> tuple[bool, bool]:
  """(broken, noedit) for the given source label."""
  if row.get("a_source") == source:
    return bool(row.get("a_broken")), bool(row.get("a_noedit"))
  if row.get("b_source") == source:
    return bool(row.get("b_broken")), bool(row.get("b_noedit"))
  raise KeyError(f"source {source!r} not in pair {row.get('pair_id')}")


def classify_win_type(row: dict) -> str | None:
  choice = row.get("choice")
  if choice not in ("a", "b"):
    return None
  loser = "b" if choice == "a" else "a"
  if row.get(f"{loser}_broken"):
    return "disqualified"
  if row.get(f"{choice}_noedit"):
    return "noedit"
  return "quality"


def merge_rows(pairs: list[dict], judgments: list[dict]) -> list[dict]:
  by_pair = {str(p["pair_id"]): p for p in pairs}
  merged: list[dict] = []
  missing_pairs: list[str] = []
  for j in judgments:
    pid = str(j.get("pair_id") or "")
    p = by_pair.get(pid)
    if not p:
      missing_pairs.append(pid)
      continue
    row = {**p, **j}
    row["winner_source"] = winner_source(row)
    row["win_type"] = classify_win_type(row)
    merged.append(row)
  if missing_pairs:
    raise SystemExit(
      f"judgments without pairs ({len(missing_pairs)}): e.g. {missing_pairs[:3]}"
    )
  if len(merged) != len(judgments):
    raise SystemExit(
      f"judgment count mismatch: pairs={len(pairs)} judgments={len(judgments)} "
      f"merged={len(merged)}"
    )
  return merged


def rate_summary(k: int, n: int) -> dict:
  lo, hi = wilson_ci(k, n)
  return {
    "k": k,
    "n": n,
    "rate": (k / n) if n else None,
    "wilson_95_lower": lo,
    "wilson_95_upper": hi,
  }


def compare_win_stats(rows: list[dict], *, target_source: str) -> dict:
  wins = sum(1 for r in rows if r.get("winner_source") == target_source)
  ties = sum(1 for r in rows if r.get("choice") == "tie")
  n = len(rows)
  out = {
    "target_source": target_source,
    "n": n,
    "wins": wins,
    "ties": ties,
    "losses": n - wins - ties,
    **rate_summary(wins, n),
  }
  win_types = Counter(r["win_type"] for r in rows if r.get("winner_source") == target_source)
  out["win_types"] = dict(win_types)
  return out


def adapter_absolute_rates(rows: list[dict]) -> dict:
  """adapter 貪欲の破壊率・無編集率（比較2の絶対チェック）。"""
  c2 = [r for r in rows if r.get("compare_type") == "2_draft_vs_adapter_greedy"]
  broken = noedit = 0
  for r in c2:
    b, n = side_flags(r, "adapter_greedy")
    broken += int(b)
    noedit += int(n)
  n = len(c2)
  return {
    "source": "adapter_greedy",
    "from_compare": "2_draft_vs_adapter_greedy",
    "broken": rate_summary(broken, n),
    "noedit": rate_summary(noedit, n),
  }


def improvement_rate(rows: list[dict]) -> dict:
  c2 = [r for r in rows if r.get("compare_type") == "2_draft_vs_adapter_greedy"]
  quality_wins = sum(
    1
    for r in c2
    if r.get("winner_source") == "adapter_greedy" and r.get("win_type") == "quality"
  )
  n = len(c2)
  return {
    "description": "比較2で adapter 貪欲が品質勝ち",
    **rate_summary(quality_wins, n),
  }


def source_flag_rates(rows: list[dict]) -> dict[str, dict]:
  """出所ごとに、当該側として現れた判定での broken/noedit 率。"""
  stats: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {"broken": [], "noedit": []})
  for r in rows:
    for side, src_key in (("a", "a_source"), ("b", "b_source")):
      src = str(r.get(src_key) or "")
      if not src:
        continue
      stats[src]["broken"].append(bool(r.get(f"{side}_broken")))
      stats[src]["noedit"].append(bool(r.get(f"{side}_noedit")))
  out: dict[str, dict] = {}
  for src, flags in sorted(stats.items()):
    n = len(flags["broken"])
    out[src] = {
      "appearances": n,
      "broken": rate_summary(sum(flags["broken"]), n),
      "noedit": rate_summary(sum(flags["noedit"]), n),
    }
  return out


def mechanical_noedit(items: list[dict], generation_path: Path) -> dict:
  gen_by_id = {str(r["id"]): r for r in load_jsonl(generation_path)}
  sims: list[float] = []
  exact = ge095 = 0
  missing = 0
  for item in items:
    iid = str(item["id"])
    g = gen_by_id.get(iid)
    if not g:
      missing += 1
      continue
    draft = str(item.get("draft") or g.get("draft") or "")
    generated = str(g.get("generated") or "")
    s = similarity(draft, generated)
    sims.append(s)
    if generated.strip() == draft.strip():
      exact += 1
    if s >= 0.95:
      ge095 += 1
  n = len(items) - missing
  return {
    "generation_file": str(generation_path),
    "n_items": len(items),
    "n_matched": n,
    "missing_ids": missing,
    "exact_copy": rate_summary(exact, n),
    "sim_ge_095": rate_summary(ge095, n),
    "sim_median": statistics.median(sims) if sims else None,
    "sim_mean": statistics.mean(sims) if sims else None,
  }


@dataclass
class ScorerEval:
  name: str
  loaded: LoadedScorer
  scores_by_pair: dict[str, dict]


def score_all_pairs(rows: list[dict], model_dir: Path) -> ScorerEval:
  loaded = load_scorer(model_dir)
  scores_by_pair: dict[str, dict] = {}
  for r in rows:
    draft = str(r.get("context_draft") or "")
    a_text = str(r.get("a_text") or "")
    b_text = str(r.get("b_text") or "")
    sa, sb = loaded.score(draft, [a_text, b_text])
    pref = None
    if sa > sb:
      pref = "a"
    elif sb > sa:
      pref = "b"
    scores_by_pair[str(r["pair_id"])] = {
      "score_a": float(sa),
      "score_b": float(sb),
      "pref": pref,
      "len_a": len(a_text),
      "len_b": len(b_text),
    }
  return ScorerEval(name=model_dir.name, loaded=loaded, scores_by_pair=scores_by_pair)


def human_pref(row: dict) -> str | None:
  c = row.get("choice")
  return c if c in ("a", "b") else None


def agreement_stats(rows: list[dict], scorer: ScorerEval) -> dict:
  all_scores: list[float] = []
  all_lens: list[float] = []
  by_compare: dict[str, dict] = {}
  disagreements: list[dict] = []

  grouped: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    grouped[str(r.get("compare_type") or "")].append(r)

  for ctype, group in sorted(grouped.items()):
    n_total = len(group)
    human_tie = 0
    human_pref_n = 0
    scorer_tie = 0
    comparable_agree = comparable_total = 0
    broken_reported = 0
    for r in group:
      pid = str(r["pair_id"])
      sc = scorer.scores_by_pair[pid]
      hp = human_pref(r)
      sp = sc["pref"]
      all_scores.extend([sc["score_a"], sc["score_b"]])
      all_lens.extend([float(sc["len_a"]), float(sc["len_b"])])
      if bool(r.get("a_broken")) or bool(r.get("b_broken")):
        broken_reported += 1
      if r.get("choice") == "tie":
        human_tie += 1
      if hp is not None:
        human_pref_n += 1
      if hp is not None and sp is None:
        scorer_tie += 1
      if hp is None or sp is None:
        continue
      comparable_total += 1
      agree = hp == sp
      if agree:
        comparable_agree += 1
      if not agree:
        disagreements.append(
          {
            "pair_id": pid,
            "compare_type": ctype,
            "human_pref": hp,
            "scorer_pref": sp,
            "human_winner_source": r.get("winner_source"),
            "win_type": r.get("win_type"),
            "a_broken": r.get("a_broken"),
            "b_broken": r.get("b_broken"),
            "a_noedit": r.get("a_noedit"),
            "b_noedit": r.get("b_noedit"),
            "a_source": r.get("a_source"),
            "b_source": r.get("b_source"),
            "score_a": sc["score_a"],
            "score_b": sc["score_b"],
          }
        )
    by_compare[ctype] = {
      "label": COMPARE_LABELS.get(ctype, ctype),
      "n_total": n_total,
      "human_tie": rate_summary(human_tie, n_total),
      "human_pref": rate_summary(human_pref_n, n_total),
      "scorer_tie_given_human_pref": rate_summary(scorer_tie, human_pref_n),
      "comparable": rate_summary(comparable_agree, comparable_total),
      "broken_reported": rate_summary(broken_reported, n_total),
    }

  return {
    "model_dir": scorer.loaded.model_dir,
    "kind": scorer.loaded.kind,
    "score_length_spearman": spearman(all_scores, all_lens),
    "by_compare_type": by_compare,
    "disagreements": disagreements,
    "disagreement_breakdown": {
      "|".join(str(x) for x in key): count
      for key, count in Counter(
        (
          d["compare_type"],
          d.get("win_type") or "tie",
          d["a_source"],
          d["b_source"],
        )
        for d in disagreements
      ).items()
    },
  }


def compare3_binomial(rows: list[dict]) -> dict:
  c3 = [r for r in rows if r.get("compare_type") == "3_base_vs_adapter_greedy"]
  wins = sum(1 for r in c3 if r.get("winner_source") == "adapter_greedy")
  n = len(c3)
  test = binomtest(wins, n=n, p=0.5, alternative="greater")
  lo, hi = wilson_ci(wins, n)
  return {
    "n": n,
    "adapter_wins": wins,
    "win_rate": wins / n if n else None,
    "criterion_rate": 0.6,
    "criterion_wins": 36,
    "passes": wins >= 36,
    "binom_p_one_sided": float(test.pvalue),
    "wilson_95_lower": lo,
    "wilson_95_upper": hi,
    "note": "比較3は「壊れにくい方」の測定。成立しても推敲の質向上の根拠にはしない。",
  }


def win_type_by_compare(rows: list[dict]) -> dict:
  out: dict[str, dict] = {}
  for ctype in sorted({str(r.get("compare_type") or "") for r in rows}):
    subset = [r for r in rows if r.get("compare_type") == ctype]
    wins = [r for r in subset if r.get("winner_source")]
    n = len(wins)
    counts = Counter(r["win_type"] for r in wins)
    out[ctype] = {
      "label": COMPARE_LABELS.get(ctype, ctype),
      "wins": n,
      "win_types": {k: rate_summary(v, n) for k, v in sorted(counts.items())},
    }
  return out


def length_compare3(rows: list[dict]) -> dict:
  c3 = [r for r in rows if r.get("compare_type") == "3_base_vs_adapter_greedy"]
  ratios: list[float] = []
  for r in c3:
    base_len = ad_len = None
    if r.get("a_source") == "base_greedy":
      base_len, ad_len = len(str(r["a_text"])), len(str(r["b_text"]))
    elif r.get("b_source") == "base_greedy":
      base_len, ad_len = len(str(r["b_text"])), len(str(r["a_text"]))
    if base_len and ad_len:
      ratios.append(ad_len / base_len)
  return {
    "adapter_over_base_median": statistics.median(ratios) if ratios else None,
    "adapter_over_base_mean": statistics.mean(ratios) if ratios else None,
    "n": len(ratios),
  }


def render_markdown(analysis: dict) -> str:
  m = analysis["model_metrics"]
  c3 = analysis["compare3"]
  imp = analysis["improvement_rate"]
  ad = analysis["adapter_absolute"]
  mech = analysis["mechanical_noedit"]
  lines: list[str] = []

  def pct(stat: dict) -> str:
    return f"{stat['k']}/{stat['n']} = {stat['rate']:.3f}"

  def ci(stat: dict) -> str:
    return f"[{stat['wilson_95_lower']:.3f}, {stat['wilson_95_upper']:.3f}]"

  def src_label(key: str) -> str:
    return SOURCE_LABELS.get(key, key)

  lines.extend(
    [
      "# ブラインド判定 集計結果",
      "",
      "## 1. 概要",
      "",
      "本書は、工程 6 の人手ブラインド判定 360 件を `scripts/analyze_blind_judgments.py` で集計した結果である。",
      "判定の生データは `data/blind_eval/judgments.jsonl`、ペア定義は `data/blind_eval/pairs.jsonl` にある。",
      "数値の詳細は `data/blind_eval/analysis.json` に同一内容を JSON で保存している。",
      "",
      f"- 判定対象節数：{analysis['inputs']['n_items']}（各節 6 比較、計 {analysis['inputs']['n_pairs']} 判定）",
      f"- 生成物の出所：`outputs/edit-sft-eval-v3/`（工程 C の本生成）",
      f"- 評価器：`outputs/pref-sentseq-keep-pairsplit`（文列型）、`outputs/pref-bt-keep-pairsplit`（Bradley-Terry 型）",
      *(
        [
          "- 工程 8-mid の文列型：`outputs/pref-sentseq-section-triples`（同じ 360 件を採点し直したもの。§5.5）",
        ]
        if analysis["scorers"].get("sentseq_section_triples") is not None
        else []
      ),
      "",
      "再生成：`make analyze-blind-judgments`",
      "",
      "## 2. 用語",
      "",
      "PLAN.md の用語表に加え、本集計で使う識別子を次に固定する。",
      "",
      "**下書き**：判定対象節の推敲前原文。`items.jsonl` の `draft`、画面上の「文脈」、評価器採点時の `source` に使う。",
      "",
      "**人間の推敲**：同一節のレビュー済み正解文。`items.jsonl` の `gold`。",
      "",
      "**素のモデル**：LoRA を載せない Qwen3-8B。PLAN の「素のモデル」と同じ。",
      "",
      "**推敲モデル**：SFT で得た LoRA を載せた Qwen3-8B。PLAN の「推敲モデル」と同じ。",
      "",
      "**貪欲生成**：各ステップで最確語のみを選ぶ生成。同一入力なら出力は 1 本に定まる（PLAN 用語表）。",
      "",
      "**サンプリング生成**：温度 0.7・top_p 0.9・シード固定で語をサンプルする生成。同一入力から複数案を出す。",
      "",
      "**素のモデル・貪欲生成**（データ上の識別子 `base_greedy`）：素のモデルが指示文付きで貪欲生成した 1 文。",
      "ファイル `outputs/edit-sft-eval-v3/base_greedy.jsonl` の `generated`。",
      "",
      "**推敲モデル・貪欲生成**（`adapter_greedy`）：推敲モデルが同一指示文で貪欲生成した 1 文。",
      "ファイル `outputs/edit-sft-eval-v3/adapter_greedy.jsonl` の `generated`。",
      "",
      "**推敲モデル・選抜1本**（`adapter_selected`）：推敲モデルがサンプリング 8 本を出し、",
      "見出し・【図】【コード】の本数を保った案だけを残したうえで評価器（文列型）が最高点を採った 1 文。",
      "健全なサンプルが 0 本のときは `adapter_greedy` にフォールバックする（`scripts/build_blind_pairs.py`）。",
      "",
      "**相対選択**：判定者が「候補 A のほうがまし」「候補 B のほうがまし」「同等」のいずれかを選んだ結果。",
      "`judgments.jsonl` の `choice`（`a` / `b` / `tie`）。",
      "",
      "**致命的欠落・破壊**（`a_broken` / `b_broken`）：見出し・【図】【コード】の欠落、内容の破壊、文体（ですます／である）の変更など、",
      "採用できない欠陥があると判定者が付けたチェック。",
      "",
      "**事実上の差分なし**（`a_noedit` / `b_noedit`）：下書きと完全一致でなくても、言い換え程度で編集と呼べないと判定者が付けたチェック。",
      "",
      "**失格勝ち**：相対選択で勝った側の相手に致命的欠落・破壊がある勝ち。",
      "**無編集勝ち**：相対選択で勝った側に事実上の差分なしがある勝ち。",
      "**品質勝ち**：上記のどちらでもない勝ち。勝った側が編集として成立しているとみなす。",
      "",
      "**破壊率**（本集計）：比較2の 60 節について、`adapter_greedy` に致命的欠落・破壊チェックが付いた割合。",
      "",
      "**無編集率**（人手）：比較2の 60 節について、`adapter_greedy` に事実上の差分なしチェックが付いた割合。",
      "",
      "**改善率**（本集計）：比較2の 60 節について、`adapter_greedy` が相対選択で勝ち、かつ品質勝ちに分類された割合。",
      "",
      "**Wilson 95% CI**：二項比率の 95% 信頼区間。",
      "",
      "## 3. 比較の種類",
      "",
      "判定対象 60 節それぞれについて、次の 6 ペアを出題した。",
      "左右（候補 A / B）の割付はシード固定乱数で決め、`pairs.jsonl` の `swapped` に記録する。",
      "",
    ]
  )
  for label, _ctype, left, right in COMPARE_DEFINITIONS:
    lines.append(f"- **{label}**：{left} vs {right}")
  lines.extend(
    [
      "",
      "## 4. 推敲モデルに関する集計",
      "",
      "対象は主に **推敲モデル・貪欲生成**（`adapter_greedy`）と **推敲モデル・選抜1本**（`adapter_selected`）。",
      "",
      "### 4.1 三つの率（`adapter_greedy`、比較2・60 節）",
      "",
      "| 指標 | 定義 | 値 | Wilson 95% CI |",
      "|---|---|---:|---:|",
      f"| 破壊率 | 致命的欠落・破壊チェック | {pct(ad['broken'])} | {ci(ad['broken'])} |",
      f"| 無編集率 | 事実上の差分なしチェック | {pct(ad['noedit'])} | {ci(ad['noedit'])} |",
      f"| 改善率 | 相対選択で勝ちかつ品質勝ち | {pct(imp)} | {ci(imp)} |",
      "",
      "### 4.2 機械計測（`adapter_greedy`、判定対象 60 節）",
      "",
      "下書きと生成文の文字列類似度。成否基準には使わない（PLAN 判定基準）。",
      "",
      f"- 完全一致：{pct(mech['exact_copy'])}",
      f"- 類似度 ≥ 0.95：{pct(mech['sim_ge_095'])}",
      f"- 類似度中央値：{mech['sim_median']:.3f}" if mech["sim_median"] is not None else "- 類似度中央値：—",
      "",
      "### 4.3 比較3（成否基準）",
      "",
      "比較3は **素のモデル・貪欲生成** と **推敲モデル・貪欲生成** の相対選択 60 件。",
      "PLAN の成立基準は推敲モデル側の勝ちが 36/60 以上。",
      "",
      f"- 推敲モデル・貪欲生成の勝ち：{c3['adapter_wins']}/{c3['n']} = {c3['win_rate']:.3f}（Wilson {ci(c3)}）",
      f"- 成立基準 36/60：{'成立' if c3['passes'] else '不成立'}",
      f"- 片側二項検定（H1: 勝率 > 0.5）：p = {c3['binom_p_one_sided']:.4f}",
      "",
      "### 4.4 各比較の相対選択勝率",
      "",
      "「勝ち」は `choice` が当該候補側を指した件数。分母は各比較 60 件（同点含む）。",
      "",
      "| 比較 | 集計対象候補 | 勝ち | 同点 | 勝率 |",
      "|---|---|---:|---:|---:|",
    ]
  )
  metric_rows = [
    ("compare1_base", "比較1", "base_greedy"),
    ("compare2_adapter", "比較2", "adapter_greedy"),
    ("compare4_adapter_selected", "比較4", "adapter_selected"),
    ("compare5_adapter_selected", "比較5", "adapter_selected"),
    ("compare6_gold", "比較6", "gold"),
  ]
  for key, label, src in metric_rows:
    s = m[key]
    lines.append(
      f"| {label} | {src_label(src)} | {s['wins']}/{s['n']} | {s['ties']} | {s['rate']:.3f} |"
    )

  lc = analysis["length_compare3"]
  if lc["adapter_over_base_median"] is not None:
    lines.append("")
    lines.append(
      f"比較3 における文字数比（推敲モデル・貪欲生成 / 素のモデル・貪欲生成）中央値："
      f"{lc['adapter_over_base_median']:.3f}"
    )

  lines.extend(["", "### 4.5 勝ちの内訳（相対選択で勝った側の分類）", ""])
  lines.append("分母は当該比較で同点を除いた勝敗件数。")
  lines.append("")
  win_type_ja = {
    "disqualified": "失格勝ち",
    "noedit": "無編集勝ち",
    "quality": "品質勝ち",
  }
  for ctype, wt in analysis["win_types_by_compare"].items():
    label = COMPARE_LABELS.get(ctype, ctype)
    parts = []
    for name, stat in wt["win_types"].items():
      parts.append(f"{win_type_ja.get(name, name)} {stat['k']}/{wt['wins']}")
    lines.append(f"- **{label}**：{', '.join(parts) if parts else '—'}")

  lines.extend(
    [
      "",
      "### 4.6 候補ラベルごとの絶対評価（出現回数ベース）",
      "",
      "同一の生成物が複数比較に登場するため、分母はそのラベルとして画面に示された回数。",
      "例：`adapter_greedy` は比較2・比較3 に各 1 回ずつ出るので分母 120。",
      "",
      "| 識別子 | 日本語名 | 出現回数 | 致命的欠落・破壊 | 事実上の差分なし |",
      "|---|---|---:|---:|---:|",
    ]
  )
  for src, st in analysis["source_flags"].items():
    lines.append(
      f"| `{src}` | {src_label(src)} | {st['appearances']} | "
      f"{st['broken']['k']} ({st['broken']['rate']:.3f}) | "
      f"{st['noedit']['k']} ({st['noedit']['rate']:.3f}) |"
    )

  lines.extend(
    [
      "",
      "## 5. 評価器に関する集計",
      "",
      "### 5.1 測定方法",
      "",
      "各判定ペアについて、下書き（`context_draft`）を source、候補 A/B を candidates として評価器が点数を付けた。",
      "点数が高い側を **評価器の選択**（`a` または `b`）とする。",
      "**人手の選択**は `judgments.jsonl` の `choice`（`a` / `b` / `tie`）。",
      "",
      "**一致**：比較可能ペアのうち、人手の選択と評価器の選択が同じ（どちらも `a`、またはどちらも `b`）。",
      "",
      "**比較可能ペア**：人手が `a` または `b` を選び、かつ評価器が同点でないペア。",
      "候補に致命的欠落・破壊チェック（`a_broken` / `b_broken`）が付いていても、人手が選好を付けていれば比較可能に含める。",
      "",
      "**人手同等**（`tie`）：2 候補のどちらも採用したくないと判定者が付けた件数。評価器一致の分母には入れない。",
      "",
      "**欠陥報告あり**（参考）：その比較種別 60 件のうち、候補 A または B の少なくとも一方に",
      "致命的欠落・破壊チェックが付いている件数。一致率の計算とは独立。",
      "",
      "**スコアと文字数の Spearman 相関**：全候補（720 点）について、評価器点数と UTF-8 文字数の順位相関。",
      "",
      "### 5.2 人手の相対選択（比較種別ごと）",
      "",
      "評価器とは独立。各比較種別 60 件について、判定者がどう付けたか。",
      "",
      "| 比較 | 全件 | 人手同等 | 人手 A/B 選択 | 欠陥報告あり |",
      "|---|---:|---:|---:|---:|",
    ]
  )
  # 人手同等・欠陥報告は評価器に依存しないので sentseq の by_compare から取る
  ref = analysis["scorers"]["sentseq"]["by_compare_type"]
  for ctype, st in ref.items():
    short = COMPARE_LABELS.get(ctype, ctype)
    tie = st["human_tie"]
    pref = st["human_pref"]
    br = st["broken_reported"]
    lines.append(
      f"| {short} | {st['n_total']} | {tie['k']} ({tie['rate']:.3f}) | "
      f"{pref['k']} ({pref['rate']:.3f}) | {br['k']} ({br['rate']:.3f}) |"
    )
  lines.append("")

  section_no = 3
  for scorer_name, title in (
    ("sentseq", "文列型（`pref-sentseq-keep-pairsplit`）"),
    ("bt", "Bradley-Terry 型（`pref-bt-keep-pairsplit`）"),
  ):
    se = analysis["scorers"][scorer_name]
    lines.append(f"### 5.{section_no} {title}")
    section_no += 1
    lines.append("")
    sp = se["score_length_spearman"]
    lines.append(
      f"- スコアと文字数の Spearman：{sp:.3f}" if sp is not None else "- スコアと文字数の Spearman：—"
    )
    lines.append("")
    lines.append(
      "| 比較 | 全件 | 人手同等 | 比較可能 | 評価器一致 | 欠陥報告あり |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ctype, st in se["by_compare_type"].items():
      short = COMPARE_LABELS.get(ctype, ctype)
      tie = st["human_tie"]
      comp = st["comparable"]
      br = st["broken_reported"]
      lines.append(
        f"| {short} | {st['n_total']} | {tie['k']} | {comp['n']} | "
        f"{comp['k']}/{comp['n']} ({comp['rate']:.3f}) | {br['k']} |"
      )
    nd = len(se["disagreements"])
    comp_all = sum(st["comparable"]["n"] for st in se["by_compare_type"].values())
    lines.append("")
    lines.append(
      f"**不一致**：比較可能 {comp_all} 件のうち **{nd} 件**"
      f"（`analysis.json` → `scorers.{scorer_name}.disagreements`）。"
    )
    lines.append("")

  extra = analysis["scorers"].get("sentseq_section_triples")
  if extra is not None:
    lines.append("### 5.5 文列型（`pref-sentseq-section-triples`）")
    lines.append("")
    lines.append(
      "工程 8-mid の三つ組みで学び直した文列型で、同じ 360 件を採点し直した結果である。"
    )
    lines.append(
      "判定 60 件のうち節 50 件は、この文列型の valid（最良 epoch の選定）に入っている。"
      "比較 6（人間の推敲対下書き）は、その valid の対と同じ文である。"
      "比較 2・3・5 の候補は教師に無い生成文である。"
    )
    lines.append(
      "比較 4・5 の「選抜1本」は、当時の文列型が選んだ文のままである。"
      "選抜をやり直した結果ではない。"
    )
    lines.append("")
    sp = extra["score_length_spearman"]
    lines.append(
      f"- スコアと文字数の Spearman：{sp:.3f}" if sp is not None else "- スコアと文字数の Spearman：—"
    )
    lines.append("")
    lines.append(
      "| 比較 | 全件 | 人手同等 | 比較可能 | 評価器一致 | 欠陥報告あり |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ctype, st in extra["by_compare_type"].items():
      short = COMPARE_LABELS.get(ctype, ctype)
      tie = st["human_tie"]
      comp = st["comparable"]
      br = st["broken_reported"]
      lines.append(
        f"| {short} | {st['n_total']} | {tie['k']} | {comp['n']} | "
        f"{comp['k']}/{comp['n']} ({comp['rate']:.3f}) | {br['k']} |"
      )
    nd = len(extra["disagreements"])
    comp_all = sum(st["comparable"]["n"] for st in extra["by_compare_type"].values())
    lines.append("")
    lines.append(
      f"**不一致**：比較可能 {comp_all} 件のうち **{nd} 件**"
      f"（`analysis.json` → `scorers.sentseq_section_triples.disagreements`）。"
    )
    lines.append("")

  lines.extend(["## 6. 入力ファイル", ""])
  path_labels = {
    "pairs": "ペア定義",
    "judgments": "人手判定",
    "items": "判定対象 60 節",
    "adapter_greedy": "推敲モデル・貪欲生成 jsonl",
    "sentseq_model": "文列型評価器",
    "bt_model": "BT 型評価器",
    "sentseq_section_triples_model": "工程 8-mid の文列型",
    "n_pairs": "判定件数",
    "n_items": "判定対象節数",
  }
  for k, v in analysis["inputs"].items():
    label = path_labels.get(k, k)
    lines.append(f"- **{label}**（`{k}`）：{v}")
  lines.append("")
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/blind_eval/pairs.jsonl")
  parser.add_argument("--judgments", default="data/blind_eval/judgments.jsonl")
  parser.add_argument("--items", default="data/blind_eval/items.jsonl")
  parser.add_argument(
    "--adapter-greedy",
    default="outputs/edit-sft-eval-v3/adapter_greedy.jsonl",
  )
  parser.add_argument(
    "--sentseq-model",
    default="outputs/pref-sentseq-keep-pairsplit",
  )
  parser.add_argument(
    "--bt-model",
    default="outputs/pref-bt-keep-pairsplit",
  )
  parser.add_argument(
    "--extra-sentseq-model",
    default="",
    help="追加の文列型。同じ 360 件を採点し、RESULTS の §5.5 に書く",
  )
  parser.add_argument("--analysis-json", default="data/blind_eval/analysis.json")
  parser.add_argument("--results-md", default="docs/RESULTS.md")
  args = parser.parse_args()

  pairs = load_jsonl(Path(args.pairs))
  judgments = load_jsonl(Path(args.judgments))
  items = load_jsonl(Path(args.items))
  merged = merge_rows(pairs, judgments)

  by_type: dict[str, list[dict]] = defaultdict(list)
  for r in merged:
    by_type[str(r["compare_type"])].append(r)

  model_metrics = {
    "compare1_base": compare_win_stats(
      by_type["1_draft_vs_base_greedy"], target_source="base_greedy"
    ),
    "compare2_adapter": compare_win_stats(
      by_type["2_draft_vs_adapter_greedy"], target_source="adapter_greedy"
    ),
    "compare4_adapter_selected": compare_win_stats(
      by_type["4_draft_vs_adapter_selected"], target_source="adapter_selected"
    ),
    "compare5_adapter_selected": compare_win_stats(
      by_type["5_gold_vs_adapter_selected"], target_source="adapter_selected"
    ),
    "compare6_gold": compare_win_stats(by_type["6_gold_vs_draft"], target_source="gold"),
  }

  analysis = {
    "inputs": {
      "pairs": args.pairs,
      "judgments": args.judgments,
      "items": args.items,
      "adapter_greedy": args.adapter_greedy,
      "sentseq_model": args.sentseq_model,
      "bt_model": args.bt_model,
      "sentseq_section_triples_model": args.extra_sentseq_model or None,
      "n_pairs": len(merged),
      "n_items": len(items),
    },
    "compare3": compare3_binomial(merged),
    "model_metrics": model_metrics,
    "adapter_absolute": adapter_absolute_rates(merged),
    "improvement_rate": improvement_rate(merged),
    "mechanical_noedit": mechanical_noedit(items, Path(args.adapter_greedy)),
    "source_flags": source_flag_rates(merged),
    "win_types_by_compare": win_type_by_compare(merged),
    "length_compare3": length_compare3(merged),
    "scorers": {},
  }

  print("scoring with sentseq...", flush=True)
  sentseq = score_all_pairs(merged, Path(args.sentseq_model))
  analysis["scorers"]["sentseq"] = agreement_stats(merged, sentseq)

  print("scoring with bt...", flush=True)
  bt = score_all_pairs(merged, Path(args.bt_model))
  analysis["scorers"]["bt"] = agreement_stats(merged, bt)

  extra_path = str(args.extra_sentseq_model or "").strip()
  if extra_path:
    print(f"scoring extra sentseq ({extra_path})...", flush=True)
    extra = score_all_pairs(merged, Path(extra_path))
    analysis["scorers"]["sentseq_section_triples"] = agreement_stats(merged, extra)

  analysis_path = Path(args.analysis_json)
  analysis_path.parent.mkdir(parents=True, exist_ok=True)
  analysis_path.write_text(
    json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )

  md_path = Path(args.results_md)
  md_path.parent.mkdir(parents=True, exist_ok=True)
  md_path.write_text(render_markdown(analysis), encoding="utf-8")

  c3 = analysis["compare3"]
  imp = analysis["improvement_rate"]
  ad = analysis["adapter_absolute"]
  print(
    json.dumps(
      {
        "compare3_adapter_wins": f"{c3['adapter_wins']}/{c3['n']}",
        "compare3_passes": c3["passes"],
        "broken_rate": ad["broken"]["rate"],
        "noedit_rate_human": ad["noedit"]["rate"],
        "improvement_rate": imp["rate"],
        "sentseq_agreement": analysis["scorers"]["sentseq"]["by_compare_type"],
        "bt_agreement": analysis["scorers"]["bt"]["by_compare_type"],
      },
      ensure_ascii=False,
      indent=2,
    )
  )
  print(f"wrote {analysis_path}")
  print(f"wrote {md_path}")


if __name__ == "__main__":
  main()
