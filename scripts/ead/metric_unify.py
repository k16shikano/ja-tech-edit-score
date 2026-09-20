#!/usr/bin/env python3
"""タスク 0-5: 指標統一・床天井。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import human_pref, merge_rows, wilson_ci
from ead.common import (
  ead_reports,
  fmt_rate,
  load_jsonl,
  md_table,
  rate_summary,
  repo_root,
  resolve_d_model_dir,
  write_json,
)
from eval_pref_c_human_agreement import load_merged
from eval_pref_multigranular_blind60 import agreement_for_pairs, score_blind_rows
from eval_pref_valid50_gold_vs_composer import agreement_for_pairs as agreement_b
from pref_d_cross_encoder import load_cross_encoder, predict_f_delta, predict_vectors
from pref_d_pair_score import gpm_rank_scores, make_score_fn
from pref_scorer import load_scorer

try:
  from pref_d_causal_reward import load_causal_reward_model, predict_f_delta as predict_f_delta_causal
  from pref_d_causal_reward import predict_vectors as predict_vectors_causal
except ImportError:
  load_causal_reward_model = None

def make_d_score_fn(model_dir: Path, *, device: str):
  meta_path = model_dir / "meta.json"
  meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
  if "causal" in str(meta.get("backend") or ""):
    if load_causal_reward_model is None:
      raise RuntimeError("pref_d_causal_reward unavailable")
    dev = torch.device(device)
    model, tokenizer, cfg, meta = load_causal_reward_model(model_dir, device=dev)
    max_length = int(meta.get("max_length", cfg.max_length))
    mode = str(meta.get("mode") or "bt")
    batch_size = 4

    if mode == "gpm":

      def score(source: str, candidates: list[str]) -> list[float]:
        vecs = predict_vectors_causal(
          model, tokenizer, [source] * len(candidates), candidates,
          max_length=max_length, device=dev, batch_size=batch_size,
        )
        return gpm_rank_scores(vecs)

    else:

      def score(source: str, candidates: list[str]) -> list[float]:
        vals = predict_f_delta_causal(
          model, tokenizer, [source] * len(candidates), candidates,
          max_length=max_length, device=dev, batch_size=batch_size,
        )
        return [float(v) for v in vals]

    return score, meta
  return make_score_fn(model_dir, device=device)


C_COMPARE = "5_gold_vs_adapter_selected"
B_COMPARE = "gold_vs_composer"
FLOOR = 0.5

REQUIRED_MODELS = [
  ("pref-sentseq-section-triples", "outputs/pref-sentseq-section-triples", "sentseq"),
  ("pref-sentseq-a2b-modernbert-d", "outputs/pref-sentseq-a2b-modernbert-d", "sentseq"),
  ("pref-d-gpm-modernbert-k8", "outputs/pref-d-gpm-modernbert-k8", "d_pair"),
  ("pref-d-gpm-modernbert-k16", "outputs/pref-d-gpm-modernbert-k16", "d_pair"),
  ("pref-d-gpm-modernbert", "outputs/pref-d-gpm-modernbert", "d_pair"),
  ("pref-d-bt-modernbert", "outputs/pref-d-bt-modernbert", "d_pair"),
  ("a2-pdpo", "a2_pdpo_experiment_spec/checkpoints/pdpo_best", "pdpo"),
]

OPTIONAL_MODELS = [
  ("pref-bt-keep-pairsplit", "outputs/pref-bt-keep-pairsplit", "auto"),
  ("pref-sentseq-keep-pairsplit", "outputs/pref-sentseq-keep-pairsplit", "auto"),
  ("pref-nce-section", "outputs/pref-nce-section", "auto"),
  ("pref-detect-section", "outputs/pref-detect-section", "auto"),
  ("pref-detect-cd-section", "outputs/pref-detect-cd-section", "auto"),
  ("modernbert-d-interval-epoch7", "outputs/pref-d-modernbert-cv", "d_pair"),
]


def load_redo_judgments(blind_dir: Path) -> list[dict]:
  rows: list[dict] = []
  for path in sorted(blind_dir.glob("judgments.redo*.jsonl")):
    rows.extend(load_jsonl(path))
  return rows


def ceiling_from_redo(root: Path) -> dict:
  blind_dir = root / "data/blind_eval"
  pairs = [p for p in load_jsonl(blind_dir / "pairs_gold_vs_adapter_selected.jsonl")]
  primary = load_jsonl(blind_dir / "judgments_gold_vs_adapter_selected.jsonl")
  redo = [r for r in load_redo_judgments(blind_dir) if r.get("compare_type") == C_COMPARE]
  if not redo:
    return {"status": "unmeasured", "reason": "no redo judgments for compare5"}

  by_pair_primary = {str(r["pair_id"]): r for r in primary}
  by_pair_redo = {str(r["pair_id"]): r for r in redo}
  common = sorted(set(by_pair_primary) & set(by_pair_redo))
  if not common:
    return {"status": "unmeasured", "reason": "no overlapping pair_id between primary and redo"}

  missing_swapped = [
    pid for pid in common if "swapped" not in by_pair_primary[pid] or "swapped" not in by_pair_redo[pid]
  ]
  if missing_swapped:
    return {
      "status": "blocked",
      "reason": "swapped flag missing on redo or primary",
      "examples": missing_swapped[:5],
    }

  agree_n = 0
  comparable = 0
  excluded_tie = 0
  for pid in common:
    p = by_pair_primary[pid]
    r = by_pair_redo[pid]
    if p.get("choice") == "tie" or r.get("choice") == "tie":
      excluded_tie += 1
      continue
    hp = human_pref(p)
    hr = human_pref(r)
    if hp is None or hr is None:
      continue
    comparable += 1
    if hp == hr:
      agree_n += 1

  stat = rate_summary(agree_n, comparable)
  status = "measured" if comparable >= 20 else "unmeasured"
  return {
    "status": status,
    "n_redo_items": len(common),
    "excluded_tie": excluded_tie,
    "comparable_n": comparable,
    "agree_n": agree_n,
    **stat,
    "provisional": comparable < 20,
  }


def score_c(model_dir: Path, kind: str, device: str) -> dict:
  pairs_path = repo_root() / "data/blind_eval/pairs_gold_vs_adapter_selected.jsonl"
  judgments_path = repo_root() / "data/blind_eval/judgments_gold_vs_adapter_selected.jsonl"
  merged = load_merged(pairs_path, judgments_path)
  if kind == "d_pair":
    model_path = resolve_d_model_dir(model_dir, 0)
    score_fn, _ = make_d_score_fn(model_path, device=device)
  elif kind == "sentseq":
    loaded = load_scorer(model_dir, calibrate_draft_zero=False)
    score_fn = loaded.score
  elif kind == "auto":
    loaded = load_scorer(model_dir)
    score_fn = loaded.score
  elif kind == "pdpo":
    return score_c_pdpo(model_dir, merged, device)
  else:
    raise ValueError(kind)
  scored = score_blind_rows(merged, score_fn)
  stats = agreement_for_pairs(scored)
  human_tie = stats["human_tie"]
  return {
    "n_total": stats["n_total"],
    "human_tie_excluded": human_tie,
    "comparable_n": stats["comparable_n"],
    "agree_n": stats["agree_n"],
    **rate_summary(stats["agree_n"], stats["comparable_n"]),
  }


def score_c_pdpo(checkpoint: Path, merged: list[dict], device: str) -> dict:
  spec_root = checkpoint.parents[1]
  sys.path.insert(0, str(spec_root / "scripts"))
  try:
    from a2_common import load_config, load_prompt_files
    from evaluate_external import load_c_gold_vs_adapter_selected, s_theta
    from model_utils import load_policy_model, load_tokenizer
  except ImportError as exc:
    return {"error": f"pdpo import failed: {exc}"}

  config = load_config(spec_root / "config/experiment.yaml")
  system, user_tpl = load_prompt_files(config)
  training = config["training"]
  max_length = int(training.get("max_length", 2048))
  tokenizer = load_tokenizer(training["base_model"])
  model = load_policy_model(training["base_model"], training["lora"], adapter_path=checkpoint)
  model.eval()

  agree_n = 0
  comparable = 0
  human_tie = 0
  for row in merged:
    if row.get("choice") == "tie":
      human_tie += 1
      continue
    draft = str(row["context_draft"])
    left = str(row["a_text"])
    right = str(row["b_text"])
    with torch.no_grad():
      s_left = s_theta(model, tokenizer, draft, left, system, user_tpl, max_length)
      s_right = s_theta(model, tokenizer, draft, right, system, user_tpl, max_length)
    if s_left == s_right:
      continue
    model_pref = "a" if s_left > s_right else "b"
    hp = human_pref(row)
    if hp is None:
      continue
    comparable += 1
    if model_pref == hp:
      agree_n += 1
  return {
    "n_total": len(merged),
    "human_tie_excluded": human_tie,
    "comparable_n": comparable,
    "agree_n": agree_n,
    **rate_summary(agree_n, comparable),
  }


def score_b(model_dir: Path, kind: str, device: str) -> dict:
  pairs = load_jsonl(repo_root() / "data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl")
  judgments = load_jsonl(repo_root() / "data/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl")
  merged = merge_rows(pairs, judgments)
  if kind == "d_pair":
    score_fn, _ = make_d_score_fn(resolve_d_model_dir(model_dir, 0), device=device)
  else:
    score_fn = load_scorer(model_dir).score
  scored = score_blind_rows(merged, score_fn)
  stats = agreement_b(scored)
  return {
    "n_total": stats["n_total"],
    "human_tie_excluded": stats["human_tie"],
    "comparable_n": stats["comparable_n"],
    "agree_n": stats["agree_n"],
    **rate_summary(stats["agree_n"], stats["comparable_n"]),
  }


def eval_model(name: str, rel: str, kind: str, root: Path, device: str, *, optional: bool) -> dict:
  model_dir = root / rel
  row = {"name": name, "path": rel, "kind": kind}
  if kind == "pdpo":
    ckpt = model_dir
    if not ckpt.is_dir():
      row["status"] = "missing"
      return row
  elif kind == "d_pair":
    try:
      resolve_d_model_dir(model_dir, 0)
    except FileNotFoundError:
      row["status"] = "missing"
      return row
  elif not model_dir.is_dir():
    row["status"] = "missing"
    return row

  try:
    row["C"] = score_c(model_dir if kind != "pdpo" else model_dir, kind, device)
    if kind in ("sentseq", "d_pair", "auto"):
      row["B"] = score_b(model_dir, kind, device)
    row["status"] = "ok"
  except Exception as exc:
    row["status"] = "error"
    row["error"] = str(exc)
  return row


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--skip-pdpo", action="store_true")
  parser.add_argument("--skip-gpu-models", action="store_true")
  args = parser.parse_args()
  root = args.root.resolve()

  floor_path = ead_reports() / "ead-floor.json"
  floor_rate = None
  if floor_path.is_file():
    floor_rate = json.loads(floor_path.read_text(encoding="utf-8")).get("valid", {}).get("accuracy_4")

  ceiling = ceiling_from_redo(root)
  models: list[dict] = []
  for spec in REQUIRED_MODELS:
    if spec[0] == "a2-pdpo" and args.skip_pdpo:
      models.append({"name": spec[0], "status": "skipped"})
      continue
    if args.skip_gpu_models and spec[2] in ("d_pair", "pdpo", "sentseq"):
      models.append({"name": spec[0], "status": "skipped_gpu"})
      continue
    models.append(eval_model(*spec, root, args.device, optional=False))

  for spec in OPTIONAL_MODELS:
    if args.skip_gpu_models:
      continue
    row = eval_model(*spec, root, args.device, optional=True)
    if row.get("status") != "missing":
      models.append(row)

  summary = {
    "floor_random": FLOOR,
    "floor_surface": floor_rate,
    "ceiling": ceiling,
    "models": models,
    "c_denominator_expected": 56,
    "b_denominator_expected": 47,
  }

  reports = ead_reports()
  write_json(reports / "ead-metric-unify.json", summary)

  table_rows = []
  for m in models:
    c = m.get("C") or {}
    b = m.get("B") or {}
    table_rows.append(
      [
        m["name"],
        m.get("status", ""),
        fmt_rate(c) if c.get("n") else "—",
        fmt_rate(b) if b.get("n") else "—",
      ]
    )

  ceil_line = "未測定"
  if ceiling.get("status") == "measured":
    ceil_line = fmt_rate(ceiling)
  elif ceiling.get("status") == "blocked":
    ceil_line = f"停止: {ceiling.get('reason')}"

  lines = [
    "# ead-metric-unify",
    "",
    f"- 床（偶然）: {FLOOR}",
    f"- 床（表層）: {fmt_pct(floor_rate)}",
    f"- 天井（再判定自己一致）: {ceil_line}",
    "",
    "一致率 = 評価器選好符号と当人ブラインド判定の符号一致。同等は除外。",
    "",
    md_table(["モデル", "状態", "C（分母 56 想定）", "B valid（分母 47 想定）"], table_rows),
  ]
  (reports / "ead-metric-unify.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-metric-unify.md")}, ensure_ascii=False))


def fmt_pct(x):
  from ead.common import fmt_pct as _fp
  return _fp(x)


if __name__ == "__main__":
  main()
