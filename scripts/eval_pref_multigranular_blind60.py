#!/usr/bin/env python3
"""工程6の人間の推敲対選抜生成60件を段階1–7で採点し、人手選択との一致を集計する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from analyze_blind_judgments import human_pref, merge_rows
from eval_pref_multigranular import (
  _gated_score_fn,
  _multigranular_score_fn,
  _sentseq_score_fn,
)
from pref_multigranular_runtime import load_multigranular_model, score_candidates_multigranular
from pref_scorer import load_scorer
from pref_static_utils import load_jsonl

COMPARE_TYPE = "5_gold_vs_adapter_selected"

STAGE_SPECS: list[dict] = [
  {
    "stage": 1,
    "kind": "sentseq",
    "dir_pattern": "pref-sentseq-keep-pairsplit",
    "seeds": (None,),
  },
  {"stage": 2, "kind": "multigranular", "dir_pattern": "pref-pair-draft-seed{seed}"},
  {"stage": 3, "kind": "multigranular", "dir_pattern": "pref-pair-humantop-seed{seed}"},
  {
    "stage": 4,
    "kind": "multigranular",
    "dir_pattern": "pref-pair-humantop-hunk-then-section-seed{seed}",
  },
  {
    "stage": 5,
    "kind": "multigranular",
    "dir_pattern": "pref-pair-humantop-hunk-replay-seed{seed}",
  },
  {
    "stage": 6,
    "kind": "multigranular",
    "dir_pattern": "pref-joint-humantop-hunk-replay-seed{seed}",
  },
  {
    "stage": 7,
    "kind": "gated",
    "dir_pattern": "pref-pair-humantop-hunk-replay-seed{seed}",
  },
]

SEEDS = (0, 1, 2)


def scorer_pref_from_scores(score_a: float, score_b: float) -> str | None:
  if score_a > score_b:
    return "a"
  if score_b > score_a:
    return "b"
  return None


def gold_scores_higher(row: dict, score_a: float, score_b: float) -> bool:
  a_src = str(row.get("a_source") or "")
  b_src = str(row.get("b_source") or "")
  if a_src == "gold" and b_src == "adapter_selected":
    return score_a > score_b
  if b_src == "gold" and a_src == "adapter_selected":
    return score_b > score_a
  raise ValueError(
    f"expected gold vs adapter_selected for pair {row.get('pair_id')!r}: "
    f"a_source={a_src!r} b_source={b_src!r}"
  )


def agreement_for_pairs(rows: list[dict]) -> dict:
  """rows には choice, pair_id, a_source, b_source, score_a, score_b が必要。"""
  n_total = len(rows)
  human_tie = 0
  human_pref_n = 0
  scorer_tie = 0
  comparable_n = 0
  agree_n = 0
  gold_higher_n = 0
  items: list[dict] = []

  for row in rows:
    score_a = float(row["score_a"])
    score_b = float(row["score_b"])
    hp = human_pref(row)
    sp = scorer_pref_from_scores(score_a, score_b)

    if row.get("choice") == "tie":
      human_tie += 1
    if hp is not None:
      human_pref_n += 1
    if hp is not None and sp is None:
      scorer_tie += 1

    agree: bool | None = None
    if hp is not None and sp is not None:
      comparable_n += 1
      agree = hp == sp
      if agree:
        agree_n += 1

    if gold_scores_higher(row, score_a, score_b):
      gold_higher_n += 1

    items.append(
      {
        "pair_id": row["pair_id"],
        "human_pref": hp,
        "scorer_pref": sp,
        "score_a": score_a,
        "score_b": score_b,
        "agree": agree,
      }
    )

  return {
    "n_total": n_total,
    "human_tie": human_tie,
    "human_pref_n": human_pref_n,
    "scorer_tie": scorer_tie,
    "comparable_n": comparable_n,
    "agree_n": agree_n,
    "agree_rate": (agree_n / comparable_n) if comparable_n else None,
    "gold_higher_n": gold_higher_n,
    "items": items,
  }


def _multigranular_score_fn_device(model_dir: Path, device: str) -> Callable:
  if not device:
    return _multigranular_score_fn(model_dir)
  loaded = load_multigranular_model(model_dir, device=device)

  def score(source: str, candidates: list[str]) -> list[float]:
    return score_candidates_multigranular(loaded, source, candidates)["logits"]

  return score


def _gated_score_fn_device(
  pair_model_dir: Path,
  gate_model_dir: Path,
  *,
  device: str,
  gate_min_margin: float = 0.0,
) -> Callable:
  if not device:
    return _gated_score_fn(pair_model_dir, gate_model_dir, gate_min_margin)
  pair_score = _multigranular_score_fn_device(pair_model_dir, device)
  gate = load_scorer(gate_model_dir)

  def score(source: str, candidates: list[str]) -> list[float]:
    pair_logits = pair_score(source, candidates)
    gate_raw = gate.score(source, candidates + [source], batch_size=4)
    self_g = gate_raw[-1]
    out: list[float] = []
    for i, cand_score in enumerate(pair_logits):
      margin = float(gate_raw[i] - self_g)
      if margin < gate_min_margin:
        out.append(float("-inf"))
      else:
        out.append(cand_score)
    return out

  return score


def score_blind_rows(rows: list[dict], score_fn: Callable) -> list[dict]:
  scored: list[dict] = []
  for row in rows:
    draft = str(row.get("context_draft") or "")
    a_text = str(row.get("a_text") or "")
    b_text = str(row.get("b_text") or "")
    logits = score_fn(draft, [a_text, b_text])
    scored.append(
      {
        **row,
        "score_a": float(logits[0]),
        "score_b": float(logits[1]),
      }
    )
  return scored


def filter_compare5_rows(pairs: list[dict], judgments: list[dict]) -> list[dict]:
  compare5_pairs = [p for p in pairs if p.get("compare_type") == COMPARE_TYPE]
  compare5_ids = {str(p["pair_id"]) for p in compare5_pairs}
  compare5_judgments = [j for j in judgments if str(j.get("pair_id") or "") in compare5_ids]
  merged = merge_rows(compare5_pairs, compare5_judgments)
  if len(merged) != len(compare5_pairs):
    raise SystemExit(
      f"compare5 merge mismatch: pairs={len(compare5_pairs)} merged={len(merged)}"
    )
  return merged


def model_dir_for_spec(root: Path, spec: dict, seed: int | None) -> Path:
  pattern = str(spec["dir_pattern"])
  if seed is None:
    rel = pattern.replace("-seed{seed}", "").replace("{seed}", "")
  else:
    rel = pattern.format(seed=seed)
  return root / "outputs" / "pref-multigranular" / rel


def evaluate_run(
  *,
  stage: int,
  seed: int | None,
  kind: str,
  model_dir: Path,
  rows: list[dict],
  gate_model_dir: Path | None,
  device: str,
) -> dict:
  model_pt = model_dir / "model.pt"
  if not model_pt.is_file():
    return {
      "stage": stage,
      "seed": seed,
      "kind": kind,
      "model": str(model_dir),
      "error": f"model not found: {model_pt}",
    }

  if kind == "sentseq":
    score_fn = _sentseq_score_fn(model_dir)
  elif kind == "multigranular":
    score_fn = _multigranular_score_fn_device(model_dir, device)
  elif kind == "gated":
    if gate_model_dir is None:
      raise ValueError("gate_model_dir required for gated kind")
    score_fn = _gated_score_fn_device(model_dir, gate_model_dir, device=device)
  else:
    raise ValueError(f"unknown kind: {kind!r}")

  scored_rows = score_blind_rows(rows, score_fn)
  stats = agreement_for_pairs(scored_rows)
  return {
    "stage": stage,
    "seed": seed,
    "kind": kind,
    "model": str(model_dir),
    **{k: stats[k] for k in stats if k != "items"},
    "items": stats["items"],
  }


def stdout_summary(run: dict) -> dict:
  if "error" in run:
    return {
      "stage": run["stage"],
      "seed": run["seed"],
      "kind": run["kind"],
      "model": run["model"],
      "error": run["error"],
    }
  comparable_n = run["comparable_n"]
  agree_n = run["agree_n"]
  return {
    "stage": run["stage"],
    "seed": run["seed"],
    "kind": run["kind"],
    "model": run["model"],
    "agree": f"{agree_n}/{comparable_n}",
    "agree_rate": run["agree_rate"],
    "gold_higher_n": run["gold_higher_n"],
    "human_tie": run["human_tie"],
    "scorer_tie": run["scorer_tie"],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default="data/blind_eval/pairs.jsonl")
  parser.add_argument("--judgments", default="data/blind_eval/judgments.jsonl")
  parser.add_argument("--gate-model", default="outputs/pref-bt-keep")
  parser.add_argument("--root", default=".")
  parser.add_argument(
    "--out",
    default="outputs/pref-multigranular/blind60_gold_vs_adapter_selected.json",
  )
  parser.add_argument("--device", default="")
  parser.add_argument("--only-stage", type=int, default=0)
  args = parser.parse_args()

  root = Path(args.root).resolve()
  pairs = load_jsonl(str(root / args.pairs))
  judgments = load_jsonl(str(root / args.judgments))
  rows = filter_compare5_rows(pairs, judgments)

  gate_model_dir = root / args.gate_model
  runs: list[dict] = []

  for spec in STAGE_SPECS:
    stage = int(spec["stage"])
    if args.only_stage and stage != args.only_stage:
      continue
    for seed in spec.get("seeds", SEEDS):
      model_dir = model_dir_for_spec(root, spec, seed)
      run = evaluate_run(
        stage=stage,
        seed=seed,
        kind=str(spec["kind"]),
        model_dir=model_dir,
        rows=rows,
        gate_model_dir=gate_model_dir if spec["kind"] == "gated" else None,
        device=str(args.device or ""),
      )
      runs.append(run)
      print(json.dumps(stdout_summary(run), ensure_ascii=False), flush=True)

  out_path = root / args.out
  if args.only_stage and out_path.is_file():
    prev = json.loads(out_path.read_text(encoding="utf-8"))
    kept = [r for r in prev.get("runs", []) if int(r.get("stage") or 0) != int(args.only_stage)]
    runs = kept + runs
    runs.sort(key=lambda r: (int(r.get("stage") or 0), r.get("seed") is None, r.get("seed") or 0))

  payload = {
    "meta": {
      "compare_type": COMPARE_TYPE,
      "pairs": str(root / args.pairs),
      "judgments": str(root / args.judgments),
      "gate_model": str(gate_model_dir),
      "n_pairs": len(rows),
      "device": args.device or None,
      "only_stage": args.only_stage or None,
    },
    "runs": runs,
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
  main()
