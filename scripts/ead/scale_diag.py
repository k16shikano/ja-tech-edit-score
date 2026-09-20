#!/usr/bin/env python3
"""タスク 0-2: D 目盛り診断。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  POSITION_RANK,
  ead_reports,
  fmt_pct,
  fmt_rate,
  load_jsonl,
  md_table,
  rate_summary,
  repo_root,
  resolve_d_model_dir,
  write_json,
)
from pref_d_cross_encoder import load_cross_encoder, predict_f_delta, predict_vectors
from pref_d_pair_score import gpm_rank_scores

try:
  from pref_d_causal_reward import load_causal_reward_model, predict_f_delta as predict_f_delta_causal
  from pref_d_causal_reward import predict_vectors as predict_vectors_causal
except ImportError:
  load_causal_reward_model = None


MODEL_SPECS = [
  ("modernbert-d-interval-epoch7", "outputs/pref-d-modernbert-cv", 0, "interval", "outputs/pref-d-modernbert-cv"),
  ("pref-d-gpm-modernbert-k8", "outputs/pref-d-gpm-modernbert-k8", 0, "gpm", "outputs/pref-d-gpm-modernbert-k8"),
  ("pref-d-gpm-modernbert-k16", "outputs/pref-d-gpm-modernbert-k16", 0, "gpm", "outputs/pref-d-gpm-modernbert-k16"),
  ("pref-d-gpm-modernbert", "outputs/pref-d-gpm-modernbert", 0, "gpm", "outputs/pref-d-gpm-modernbert"),
  ("pref-d-bt-modernbert", "outputs/pref-d-bt-modernbert", 0, "bt", "outputs/pref-d-bt-modernbert"),
  ("pref-d-gpm-qwen3-8b-f0", "outputs/pref-d-gpm-qwen3-8b", 0, "gpm", None),
  ("pref-d-gpm-qwen3-8b-f1", "outputs/pref-d-gpm-qwen3-8b", 1, "gpm", None),
  ("pref-d-gpm-qwen3-8b-f2", "outputs/pref-d-gpm-qwen3-8b", 2, "gpm", None),
]

C_VS_D_ORIGIN_LABELS = {
  "gen_c_human_d": "(生成-c, 人間-d)",
  "gen_c_gen_d": "(生成-c, 生成-d)",
}

MIN_GEN_GEN_PAIRS_FOR_RANK = 5

PAIR_GROUPS = [
  ("a vs b", "a", "b"),
  ("a vs c", "a", "c"),
  ("a vs d", "a", "d"),
  ("b vs c", "b", "c"),
  ("b vs d", "b", "d"),
  ("c vs d", "c", "d"),
]


def is_causal(meta: dict) -> bool:
  backend = str(meta.get("backend") or "")
  return "causal" in backend


def load_scorer(model_dir: Path, device: torch.device):
  meta_path = model_dir / "meta.json"
  meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
  if is_causal(meta):
    if load_causal_reward_model is None:
      raise RuntimeError("pref_d_causal_reward unavailable")
    model, tokenizer, cfg, meta = load_causal_reward_model(model_dir, device=device)
    max_length = int(meta.get("max_length", cfg.max_length))
    mode = str(meta.get("mode") or "gpm")

    def score_rows(rows: list[dict]) -> dict[str, float | None]:
      if mode == "gpm":
        drafts = [r["draft"] for r in rows]
        ys = [r["y"] for r in rows]
        vecs = predict_vectors_causal(
          model, tokenizer, drafts, ys, max_length=max_length, device=device, batch_size=4
        )
        scores = gpm_rank_scores(vecs)
      else:
        drafts = [r["draft"] for r in rows]
        ys = [r["y"] for r in rows]
        vals = predict_f_delta_causal(
          model, tokenizer, drafts, ys, max_length=max_length, device=device, batch_size=4
        )
        scores = [float(v) for v in vals]
      return {r["row_id"]: s for r, s in zip(rows, scores)}

    return score_rows, meta

  model, tokenizer, cfg, meta = load_cross_encoder(model_dir, device=device)
  max_length = int(meta.get("max_length", cfg.max_length))
  mode = str(meta.get("mode") or "bt")

  def score_rows(rows: list[dict]) -> dict[str, float | None]:
    if mode == "gpm":
      by_item: dict[str, list[dict]] = defaultdict(list)
      for row in rows:
        by_item[str(row["item_id"])].append(row)
      out: dict[str, float | None] = {}
      for item_rows in by_item.values():
        drafts = [r["draft"] for r in item_rows]
        ys = [r["y"] for r in item_rows]
        vecs = predict_vectors(
          model, tokenizer, drafts, ys, max_length=max_length, device=device, batch_size=8
        )
        scores = gpm_rank_scores(vecs)
        for row, score in zip(item_rows, scores):
          out[row["row_id"]] = score
      return out
    drafts = [r["draft"] for r in rows]
    ys = [r["y"] for r in rows]
    vals = predict_f_delta(
      model, tokenizer, drafts, ys, max_length=max_length, device=device, batch_size=16
    )
    out: dict[str, float | None] = {}
    for row, val in zip(rows, vals):
      if math.isfinite(val):
        out[row["row_id"]] = float(val)
      else:
        out[row["row_id"]] = None
    return out

  return score_rows, meta


def pair_label_key(low: str, high: str) -> str:
  return f"{low} vs {high}"


def position_rank(position: str) -> int | None:
  if position == "eq":
    return POSITION_RANK["b"]
  return POSITION_RANK.get(position)


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


def load_oof_prediction_rows(cv_dir: Path) -> list[dict]:
  rows: list[dict] = []
  for fold in range(5):
    path = cv_dir / f"fold{fold}" / "best_valid_predictions.jsonl"
    if not path.is_file():
      raise FileNotFoundError(path)
    rows.extend(load_jsonl(path))
  return rows


def scores_from_prediction_rows(rows: list[dict], mode: str) -> dict[str, float | None]:
  if mode == "bt":
    out: dict[str, float | None] = {}
    for row in rows:
      delta = row.get("delta")
      out[row["row_id"]] = float(delta) if delta is not None and math.isfinite(float(delta)) else None
    return out

  if mode == "interval":
    out = {}
    for row in rows:
      val = row.get("f")
      out[row["row_id"]] = float(val) if val is not None and math.isfinite(float(val)) else None
    return out

  if mode != "gpm":
    raise ValueError(f"unsupported prediction mode: {mode}")

  by_item: dict[str, list[dict]] = defaultdict(list)
  for row in rows:
    by_item[str(row["item_id"])].append(row)

  out: dict[str, float | None] = {}
  for item_rows in by_item.values():
    vectors: list[list[float]] = []
    indexed: list[dict] = []
    for row in item_rows:
      vec = row.get("vector")
      if vec is None:
        continue
      vectors.append([float(v) for v in vec])
      indexed.append(row)
    if len(vectors) != len(item_rows):
      for row in item_rows:
        out[row["row_id"]] = None
      continue
    item_scores = gpm_rank_scores(vectors)
    for row, score in zip(indexed, item_scores):
      out[row["row_id"]] = score
  return out


def filter_eval_rows(rows: list[dict], eval_rows: str, *, root: Path) -> list[dict]:
  if eval_rows == "all":
    return rows
  split_path = root / ("data/d/valid.jsonl" if eval_rows == "valid" else "data/d/train.jsonl")
  allowed = {r["row_id"] for r in load_jsonl(split_path)}
  return [r for r in rows if r["row_id"] in allowed]


def label_pair_metrics(rows: list[dict], scores: dict[str, float | None]) -> dict:
  by_item: dict[str, list[dict]] = defaultdict(list)
  for row in rows:
    by_item[str(row["item_id"])].append(row)

  grouped: dict[str, dict] = {}
  origin_grouped: dict[str, dict] = {}
  total_pairs = 0
  for _, item_rows in by_item.items():
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
        }
      )
    for a, b in combinations(indexed, 2):
      if a["rank"] == b["rank"]:
        continue
      low, high = (a, b) if a["rank"] < b["rank"] else (b, a)
      key = pair_label_key(low["position"], high["position"])
      bucket = grouped.setdefault(key, {"correct": 0, "total": 0})
      bucket["total"] += 1
      total_pairs += 1
      correct = 0.0
      if high["score"] > low["score"]:
        correct = 1.0
      elif high["score"] == low["score"]:
        correct = 0.5
      bucket["correct"] += correct

      if key == "c vs d":
        origin = c_vs_d_origin_key(low["y_kind"], high["y_kind"])
        if origin is not None:
          ob = origin_grouped.setdefault(origin, {"correct": 0, "total": 0})
          ob["total"] += 1
          ob["correct"] += correct

  out: dict[str, dict] = {}
  for key, bucket in grouped.items():
    out[key] = rate_summary(int(round(bucket["correct"])), bucket["total"])
    out[key]["share_of_all_pairs"] = bucket["total"] / total_pairs if total_pairs else None

  c_vs_d_by_origin: dict[str, dict] = {}
  for origin, bucket in origin_grouped.items():
    c_vs_d_by_origin[origin] = rate_summary(int(round(bucket["correct"])), bucket["total"])

  return {
    "by_label_pair": out,
    "total_pairs": total_pairs,
    "c_vs_d_by_origin": c_vs_d_by_origin,
  }


def decide_rank_role(c_vs_d_gen_gen: dict) -> dict:
  n = int(c_vs_d_gen_gen.get("n") or 0)
  if n == 0:
    return {"role": "gate_only", "reason": "gen_c vs gen_d pairs missing"}
  if n < MIN_GEN_GEN_PAIRS_FOR_RANK:
    return {
      "role": "gate_only",
      "reason": f"gen_c vs gen_d n={n} < {MIN_GEN_GEN_PAIRS_FOR_RANK}",
    }
  lo = c_vs_d_gen_gen.get("wilson_95_lower")
  hi = c_vs_d_gen_gen.get("wilson_95_upper")
  if lo is None or math.isnan(lo):
    return {"role": "gate_only", "reason": "gen_c vs gen_d Wilson interval unavailable"}
  if lo > 0.5:
    return {"role": "gate_and_rank", "reason": "gen_c vs gen_d Wilson lower > 0.5"}
  if hi is not None and lo <= 0.5 <= hi:
    if lo >= 0.45:
      return {"role": "borderline", "reason": "gen_c vs gen_d Wilson interval crosses 0.5 with lower >= 0.45"}
    return {"role": "gate_only", "reason": "gen_c vs gen_d Wilson interval crosses 0.5"}
  return {"role": "gate_only", "reason": "gen_c vs gen_d Wilson upper <= 0.5 or lower <= 0.5"}


def eval_rows_label(eval_rows: str) -> str:
  if eval_rows == "valid":
    return "D valid"
  if eval_rows == "train":
    return "D train"
  return "D 全件 OOF"


def origin_table_rows(model_result: dict) -> list[list[str]]:
  by_origin = model_result.get("c_vs_d_by_origin") or {}
  rows: list[list[str]] = []
  for key in ("gen_c_human_d", "gen_c_gen_d"):
    stat = by_origin.get(key, rate_summary(0, 0))
    rows.append([C_VS_D_ORIGIN_LABELS[key], str(stat.get("n", "—")), fmt_rate(stat)])
  return rows


def run_model_eval(
  *,
  name: str,
  rel: str,
  fold: int,
  kind: str,
  oof_cv_rel: str | None,
  rows: list[dict],
  root: Path,
  device: torch.device,
  from_oof: bool,
  eval_rows_name: str,
) -> dict:
  if from_oof:
    if not oof_cv_rel:
      raise RuntimeError("OOF predictions unavailable for this model")
    cv_dir = root / oof_cv_rel if not Path(oof_cv_rel).is_absolute() else Path(oof_cv_rel)
    pred_rows = load_oof_prediction_rows(cv_dir)
    eval_rows = filter_eval_rows(pred_rows, eval_rows_name, root=root)
    scores = scores_from_prediction_rows(eval_rows, kind)
    meta = {"mode": kind, "backend": "oof_predictions", "source": str(cv_dir)}
    model_dir = cv_dir
  else:
    eval_rows = rows
    if kind == "direct":
      model_dir = Path(rel)
    else:
      cv_dir = root / rel if not Path(rel).is_absolute() else Path(rel)
      model_dir = resolve_d_model_dir(cv_dir, fold)
    score_rows_fn, meta = load_scorer(model_dir, device)
    scores = score_rows_fn(eval_rows)

  metrics = label_pair_metrics(eval_rows, scores)
  cvd = metrics["by_label_pair"].get("c vs d", rate_summary(0, 0))
  by_origin = metrics.get("c_vs_d_by_origin") or {}
  gen_gen = by_origin.get("gen_c_gen_d", rate_summary(0, 0))
  rank_role = decide_rank_role(gen_gen)
  return {
    "name": name,
    "model_dir": str(model_dir),
    "meta": {k: meta.get(k) for k in ("mode", "backend", "head_dim", "best_epoch", "source")},
    **metrics,
    "c_vs_d": cvd,
    "c_vs_d_gen_c_human_d": by_origin.get("gen_c_human_d", rate_summary(0, 0)),
    "c_vs_d_gen_c_gen_d": gen_gen,
    "d_role": rank_role,
  }


def eval_row_count(root: Path, eval_rows: str) -> int:
  if eval_rows == "all":
    return len(load_jsonl(root / "data/d/train.jsonl")) + len(load_jsonl(root / "data/d/valid.jsonl"))
  if eval_rows == "train":
    return len(load_jsonl(root / "data/d/train.jsonl"))
  return len(load_jsonl(root / "data/d/valid.jsonl"))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--valid", type=Path, default=Path("data/d/valid.jsonl"))
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument(
    "--from-oof",
    action="store_true",
    help="CV の best_valid_predictions.jsonl から採点（GPU 不要）",
  )
  parser.add_argument(
    "--eval-rows",
    choices=("valid", "train", "all"),
    default="valid",
    help="集計対象行（from-oof 時は OOF 800 行から抽出）",
  )
  args = parser.parse_args()

  root = args.root.resolve()
  valid_rows = load_jsonl(root / args.valid)
  device = torch.device(args.device)

  model_specs = list(MODEL_SPECS)
  lock_path = root / "outputs/d_epoch7/d_epoch7.lock.json"
  if lock_path.is_file() and not args.from_oof:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    interval_ckpt = Path(lock["checkpoint"]["path"])
    if not interval_ckpt.is_absolute():
      interval_ckpt = root / interval_ckpt
    model_specs[0] = (
      "modernbert-d-interval-epoch7",
      str(interval_ckpt),
      0,
      "direct",
      "outputs/pref-d-modernbert-cv",
    )

  results: list[dict] = []

  for name, rel, fold, kind, oof_cv_rel in model_specs:
    try:
      if kind == "direct" and args.from_oof:
        kind = "interval"
      results.append(
        run_model_eval(
          name=name,
          rel=rel,
          fold=fold,
          kind=kind,
          oof_cv_rel=oof_cv_rel,
          rows=valid_rows,
          root=root,
          device=device,
          from_oof=args.from_oof,
          eval_rows_name=args.eval_rows,
        )
      )
    except Exception as exc:
      results.append({"name": name, "error": repr(exc), "model_dir": str(root / rel)})

  overall_role = "gate_only"
  if any(r.get("d_role", {}).get("role") == "borderline" for r in results if "d_role" in r):
    overall_role = "borderline"
  elif any(r.get("d_role", {}).get("role") == "gate_and_rank" for r in results if "d_role" in r):
    overall_role = "gate_and_rank"

  eval_label = eval_rows_label(args.eval_rows)
  row_n = eval_row_count(root, args.eval_rows)

  summary = {
    "eval_rows": args.eval_rows,
    "eval_label": eval_label,
    "eval_n": row_n,
    "from_oof": args.from_oof,
    "models": results,
    "d_role_decision": overall_role,
    "d_role_note": "順位づけ併用は gen_c vs gen_d のみで判定。全体 c vs d は易しい境界混在の参考値。",
  }

  reports = ead_reports()
  write_json(reports / "ead-scale-diag.json", summary)

  lines = [
    "# ead-scale-diag",
    "",
    f"{eval_label} {row_n} 行。同一下書き内ラベル対別一致率。",
    "",
    "c vs d の全体一致率は (生成-c, 人間-d) が大半を占めるため、順位づけへの併用判断には使わない。",
    f"**D の順位づけ併用（gen_c vs gen_d のみ）**: {overall_role}",
    "",
    md_table(
      ["モデル", "c vs d 全体", "n", "(生成-c, 人間-d)", "n", "(生成-c, 生成-d)", "n"],
      [
        [
          r["name"],
          fmt_rate(r.get("c_vs_d", {})),
          str(r.get("c_vs_d", {}).get("n", "—")),
          fmt_rate(r.get("c_vs_d_gen_c_human_d", {})),
          str(r.get("c_vs_d_gen_c_human_d", {}).get("n", "—")),
          fmt_rate(r.get("c_vs_d_gen_c_gen_d", {})),
          str(r.get("c_vs_d_gen_c_gen_d", {}).get("n", "—")),
        ]
        for r in results
        if "c_vs_d" in r
      ],
    ),
    "",
    "## ラベル対別",
    "",
  ]
  for r in results:
    if "by_label_pair" not in r:
      lines.append(f"### {r['name']} — エラー: {r.get('error')}")
      continue
    lines.append(f"### {r['name']}")
    rows = []
    for label, _lo, _hi in PAIR_GROUPS:
      stat = r["by_label_pair"].get(label, {})
      rows.append([label, fmt_rate(stat), str(stat.get("n", "—"))])
    lines.append(md_table(["ラベル対", "一致率", "n"], rows))
    if r.get("c_vs_d_by_origin"):
      lines.append("")
      lines.append("c vs d 出自別:")
      lines.append(md_table(["出自", "n", "一致率"], origin_table_rows(r)))
    lines.append("")

  (reports / "ead-scale-diag.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-scale-diag.md"), "d_role": overall_role}, ensure_ascii=False))


if __name__ == "__main__":
  main()
