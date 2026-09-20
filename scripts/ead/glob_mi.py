#!/usr/bin/env python3
"""Phase 1: 大局性（条件付き相互情報量）の検証。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  ead_reports,
  ead_work,
  fmt_pct,
  id_from_row,
  load_jsonl,
  md_table,
  normalized_levenshtein,
  repo_root,
  surface_features,
  write_json,
  write_jsonl,
)

try:
  from scipy import stats as scipy_stats
except ImportError:
  scipy_stats = None

MODEL_8B = "Qwen/Qwen3-8B"
MODEL_4B = "Qwen/Qwen3-4B"
K_VALUES_DEFAULT = (4, 8, 16)
C_SEEDS = (0, 1, 2)
SOURCES = ("human", "composer", "draft")
COND_SEEDS = {"A": (0,), "B": (0,), "C": C_SEEDS, "D": (0,)}
BUDGET_HOURS = 24.0
DRAFT_LABEL = "下書き"
CONTEXT_LABEL = "文脈"
REVISION_LABEL = "推敲"


@dataclass
class SectionItem:
  item_id: str
  draft: str
  human: str
  composer: str
  split: str
  delta_chars: float
  edit_distance_norm: float


@dataclass
class RunConfig:
  k_values: tuple[int, ...] = K_VALUES_DEFAULT
  model_id: str = MODEL_8B
  cond_seeds: dict[str, tuple[int, ...]] = field(default_factory=lambda: dict(COND_SEEDS))
  degenerate_steps: list[str] = field(default_factory=list)


def split_paragraphs(text: str) -> list[str]:
  parts = [p.strip() for p in text.split("\n\n") if p.strip()]
  return parts if parts else ([text.strip()] if text.strip() else [])


def group_into_k_blocks(paragraphs: list[str], k_target: int) -> list[str]:
  n = len(paragraphs)
  if n == 0:
    return []
  k_eff = min(k_target, n)
  blocks: list[str] = []
  start = 0
  for i in range(k_eff):
    end = start + (n - start) // (k_eff - i)
    blocks.append("\n\n".join(paragraphs[start:end]))
    start = end
  return blocks


def load_sections(root: Path) -> tuple[list[SectionItem], dict[str, str]]:
  keep_rows = load_jsonl(root / "data/revision_corpus/keep_section.jsonl")
  composer_rows = load_jsonl(root / "data/section_middle/revisions.jsonl")
  composer_by_id = {str(r["id"]): str(r["text"]) for r in composer_rows if r.get("id")}
  split_path = ead_work() / "split_map.json"
  split_map: dict[str, str] = {}
  if split_path.is_file():
    split_map = json.loads(split_path.read_text(encoding="utf-8"))

  items: list[SectionItem] = []
  for row in keep_rows:
    item_id = id_from_row(row)
    if not item_id:
      continue
    draft = str(row.get("source_text") or "")
    human = str(row.get("edited_text") or "")
    composer = composer_by_id.get(item_id, "")
    if not draft or not human or not composer:
      continue
    feats = surface_features(draft, human)
    items.append(
      SectionItem(
        item_id=item_id,
        draft=draft,
        human=human,
        composer=composer,
        split=split_map.get(item_id, "unknown"),
        delta_chars=feats["delta_chars"],
        edit_distance_norm=feats["edit_distance_norm"],
      )
    )
  return items, split_map


def source_text(item: SectionItem, source: str) -> str:
  if source == "human":
    return item.human
  if source == "composer":
    return item.composer
  return item.draft


class CausalScorer:
  def __init__(self, model_id: str, device: torch.device, *, max_length: int = 8192):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    self.device = device
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained(
      model_id,
      dtype=torch.bfloat16,
      attn_implementation="sdpa",
      low_cpu_mem_usage=True,
    )
    self.model.to(device)
    self.model.eval()
    self.model.config.use_cache = True
    self.model_id = model_id

  def encode(self, text: str) -> list[int]:
    return self.tokenizer.encode(text, add_special_tokens=False)

  def _draft_part(self, draft: str) -> str:
    return f"{DRAFT_LABEL}:\n{draft}\n\n"

  def _suffix_part(self, context_blocks: list[str] | None) -> str:
    if context_blocks:
      ctx = "\n\n".join(context_blocks)
      return f"{CONTEXT_LABEL}:\n{ctx}\n\n{REVISION_LABEL}:\n"
    return f"{REVISION_LABEL}:\n"

  @torch.inference_mode()
  def score_target(
    self,
    draft: str,
    context_blocks: list[str] | None,
    target_text: str,
    *,
    draft_cache: dict[str, Any] | None = None,
  ) -> tuple[float, int]:
    target_ids = self.encode(target_text)
    if not target_ids:
      return float("nan"), 0

    suffix_ids = self.encode(self._suffix_part(context_blocks))
    cont_ids = suffix_ids + target_ids
    max_cont = self.max_length - 32
    if len(cont_ids) > max_cont:
      cont_ids = cont_ids[-max_cont:]
      suffix_ids = cont_ids[: max(0, len(cont_ids) - len(target_ids))]
      target_ids = cont_ids[len(suffix_ids) :]

    draft_ids = self.encode(self._draft_part(draft))
    if draft_cache is not None and draft_cache.get("draft") == draft:
      past = draft_cache["past"]
    else:
      if draft_ids:
        out = self.model(
          torch.tensor([draft_ids], device=self.device),
          use_cache=True,
        )
        past = out.past_key_values
      else:
        past = None
      if draft_cache is not None:
        draft_cache["draft"] = draft
        draft_cache["past"] = past

    out = self.model(
      torch.tensor([cont_ids], device=self.device),
      past_key_values=past,
      use_cache=False,
    )
    logits = out.logits
    start = len(suffix_ids) - 1
    pred = logits[0, start : start + len(target_ids), :]
    target_t = torch.tensor(target_ids, device=self.device)
    tok_lp = F.log_softmax(pred, dim=-1).gather(1, target_t.unsqueeze(1)).squeeze(1)
    return float(tok_lp.sum().item()), len(target_ids)

def token_len(scorer: CausalScorer, text: str) -> int:
  return len(scorer.encode(text))


def pick_cross_blocks_scored(
  rng: random.Random,
  scorer: CausalScorer,
  *,
  pool: list[tuple[str, list[str]]],
  block_idx: int,
  target_token_len: int,
  self_id: str,
) -> list[str]:
  candidates: list[tuple[list[str], int]] = []
  lo = int(target_token_len * 0.9)
  hi = int(target_token_len * 1.1) + 1
  for doc_id, blocks in pool:
    if doc_id == self_id:
      continue
    if block_idx >= len(blocks):
      continue
    other = [blocks[j] for j in range(len(blocks)) if j != block_idx]
    if not other:
      continue
    ctx = "\n\n".join(other)
    tlen = token_len(scorer, ctx)
    candidates.append((other, tlen))
  if not candidates:
    for doc_id, blocks in pool:
      if doc_id == self_id:
        continue
      other = [blocks[j] for j in range(len(blocks)) if j != block_idx]
      if other:
        return other
    return []
  matched = [c for c in candidates if lo <= c[1] <= hi]
  pick_from = matched if matched else candidates
  return rng.choice(pick_from)[0]


def cache_key(item_id: str, k: int, source: str, block_idx: int, cond: str, seed: int) -> str:
  return f"{item_id}|K{k}|{source}|b{block_idx}|{cond}|s{seed}"


def load_cache(path: Path) -> dict[str, dict]:
  if not path.is_file():
    return {}
  out: dict[str, dict] = {}
  for row in load_jsonl(path):
    key = row.get("cache_key") or cache_key(
      row["item_id"], row["K"], row["source"], row["block_idx"], row["cond"], row["seed"]
    )
    out[key] = row
  return out


def estimate_forwards(
  n_sections: int,
  avg_blocks: float,
  k_values: tuple[int, ...],
  cond_seeds: dict[str, tuple[int, ...]],
) -> int:
  per_block = sum(len(cond_seeds[c]) for c in ("A", "B", "C", "D"))
  return int(n_sections * len(k_values) * avg_blocks * len(SOURCES) * per_block)


def apply_degenerates(cfg: RunConfig, extrap_hours: float) -> RunConfig:
  if extrap_hours <= BUDGET_HOURS:
    return cfg
  cfg = RunConfig(
    k_values=cfg.k_values,
    model_id=cfg.model_id,
    cond_seeds=dict(cfg.cond_seeds),
    degenerate_steps=list(cfg.degenerate_steps),
  )
  if "A/D 1 seed" not in cfg.degenerate_steps:
    cfg.cond_seeds = {"A": (0,), "B": (0,), "C": C_SEEDS, "D": (0,)}
    cfg.degenerate_steps.append("A/D 1 seed")
  extrap_hours = extrap_hours * (2.0 / 3.0)
  if extrap_hours <= BUDGET_HOURS:
    return cfg
  if cfg.model_id == MODEL_8B:
    cfg.model_id = MODEL_4B
    cfg.degenerate_steps.append(f"model fallback {MODEL_4B}")
    extrap_hours *= 0.6
  if extrap_hours <= BUDGET_HOURS:
    return cfg
  if len(cfg.k_values) > 2:
    cfg.k_values = tuple(k for k in cfg.k_values if k != 16)
    cfg.degenerate_steps.append("K set {4,8}")
  return cfg


def quartiles(vals: list[float]) -> dict[str, float | None]:
  if not vals:
    return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None, "mean": None}
  arr = np.asarray(vals, dtype=np.float64)
  return {
    "n": len(vals),
    "min": float(np.min(arr)),
    "q1": float(np.percentile(arr, 25)),
    "median": float(np.median(arr)),
    "q3": float(np.percentile(arr, 75)),
    "max": float(np.max(arr)),
    "mean": float(np.mean(arr)),
  }


def cliffs_delta_paired(x: list[float], y: list[float]) -> float | None:
  if len(x) != len(y) or not x:
    return None
  wins = sum(1 for a, b in zip(x, y) if a > b)
  losses = sum(1 for a, b in zip(x, y) if a < b)
  return (wins - losses) / len(x)


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> float | None:
  if len(x) < 4 or scipy_stats is None:
    return None
  rx = scipy_stats.rankdata(x)
  ry = scipy_stats.rankdata(y)
  rz = scipy_stats.rankdata(z)
  slope_x, intercept_x, _, _, _ = scipy_stats.linregress(rz, rx)
  slope_y, intercept_y, _, _, _ = scipy_stats.linregress(rz, ry)
  res_x = rx - (slope_x * rz + intercept_x)
  res_y = ry - (slope_y * rz + intercept_y)
  r, _ = scipy_stats.pearsonr(res_x, res_y)
  return float(r)


def decile_strata(items: list[dict], *, key: str = "delta_chars") -> list[dict]:
  if not items:
    return []
  vals = [it[key] for it in items]
  arr = np.asarray(vals, dtype=np.float64)
  if arr.size == 0:
    return []
  n_bins = min(10, len(vals))
  if n_bins <= 1:
    diffs = [
      it["G_human"] - it["G_composer"]
      for it in items
      if it.get("G_human") is not None and it.get("G_composer") is not None
    ]
    return [
      {
        "decile": 0,
        "n": len(items),
        "mean_G_diff_human_minus_composer": float(np.mean(diffs)) if diffs else None,
      }
    ]
  edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
  edges = np.unique(edges)
  out: list[dict] = []
  for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    bucket = [it for it in items if lo <= it[key] <= hi or (i == len(edges) - 2 and it[key] == hi)]
    if not bucket:
      continue
    diffs = [it["G_human"] - it["G_composer"] for it in bucket if it.get("G_human") is not None and it.get("G_composer") is not None]
    mean_diff = float(np.mean(diffs)) if diffs else None
    out.append({"decile": i, "n": len(bucket), "mean_G_diff_human_minus_composer": mean_diff})
  return out


def count_excluded(items: list[SectionItem]) -> int:
  n = 0
  for item in items:
    if len(split_paragraphs(item.human)) < 4:
      n += 1
  return n


def score_sections(
  scorer: CausalScorer,
  items: list[SectionItem],
  cfg: RunConfig,
  *,
  cache: dict[str, dict],
  cache_path: Path,
  timing: dict[str, float],
) -> tuple[list[dict], int]:
  aggregates: list[dict] = []
  forward_count = int(timing.get("forward_count", 0))
  t0 = time.perf_counter()

  pool_by_k: dict[int, list[tuple[str, list[str]]]] = {}
  for k in cfg.k_values:
    pool: list[tuple[str, list[str]]] = []
    for item in items:
      paras = split_paragraphs(item.human)
      if len(paras) < 4:
        continue
      blocks = group_into_k_blocks(paras, k)
      if len(blocks) >= 4:
        pool.append((item.item_id, blocks))
    pool_by_k[k] = pool

  for item in items:
    paras_human = split_paragraphs(item.human)
    if len(paras_human) < 4:
      continue
    composer_paras = split_paragraphs(item.composer)
    draft_paras = split_paragraphs(item.draft)
    for k in cfg.k_values:
      blocks_by_source = {
        "human": group_into_k_blocks(paras_human, k),
        "composer": group_into_k_blocks(composer_paras, k),
        "draft": group_into_k_blocks(draft_paras, k),
      }
      n_blocks = min(len(blocks_by_source[s]) for s in SOURCES)
      if n_blocks < 4:
        continue

      for source in SOURCES:
        blocks = blocks_by_source[source]
        comp_blocks = blocks_by_source["composer"]
        for block_idx in range(n_blocks):
          draft_cache: dict[str, Any] = {}
          y_k = blocks[block_idx]
          y_neg = [blocks[j] for j in range(n_blocks) if j != block_idx]
          comp_neg = [comp_blocks[j] for j in range(n_blocks) if j != block_idx]
          c_ctx_token_len = token_len(scorer, "\n\n".join(y_neg)) if y_neg else 0
          cond_logps: dict[str, list[float]] = {c: [] for c in ("A", "B", "C", "D")}

          for cond in ("A", "B", "C", "D"):
            seeds = cfg.cond_seeds[cond]
            for seed in seeds:
              ck = cache_key(item.item_id, k, source, block_idx, cond, seed)
              if ck in cache:
                cond_logps[cond].append(float(cache[ck]["logp"]))
                continue
              if cond == "A":
                ctx_blocks: list[str] = []
              elif cond == "B":
                ctx_blocks = y_neg
              elif cond == "D":
                ctx_blocks = comp_neg
              else:
                rng = random.Random(seed)
                ctx_blocks = pick_cross_blocks_scored(
                  rng,
                  scorer,
                  pool=pool_by_k[k],
                  block_idx=block_idx,
                  target_token_len=c_ctx_token_len,
                  self_id=item.item_id,
                )
              t_fwd = time.perf_counter()
              logp, n_tok = scorer.score_target(
                item.draft,
                ctx_blocks,
                y_k,
                draft_cache=draft_cache,
              )
              timing["forward_seconds"] = timing.get("forward_seconds", 0.0) + (time.perf_counter() - t_fwd)
              forward_count += 1
              cache[ck] = {
                "cache_key": ck,
                "item_id": item.item_id,
                "K": k,
                "source": source,
                "block_idx": block_idx,
                "cond": cond,
                "seed": seed,
                "logp": logp,
                "n_tokens": n_tok,
                "split": item.split,
              }
              cond_logps[cond].append(logp)

          n_tok = int(cache.get(cache_key(item.item_id, k, source, block_idx, "A", 0), {}).get("n_tokens") or 0)
          if not n_tok:
            n_tok = token_len(scorer, y_k)
          logp_b = float(np.mean(cond_logps["B"])) if cond_logps["B"] else float("nan")
          logp_c = float(np.mean(cond_logps["C"])) if cond_logps["C"] else float("nan")
          g_val = (logp_b - logp_c) / n_tok if n_tok else float("nan")
          aggregates.append(
            {
              "item_id": item.item_id,
              "K": k,
              "source": source,
              "block_idx": block_idx,
              "logp_A": float(np.mean(cond_logps["A"])) if cond_logps["A"] else float("nan"),
              "logp_B": logp_b,
              "logp_C": logp_c,
              "logp_D": float(np.mean(cond_logps["D"])) if cond_logps["D"] else float("nan"),
              "G": g_val,
              "n_tokens": n_tok,
              "delta_chars": item.delta_chars,
              "edit_distance_norm": item.edit_distance_norm,
              "split": item.split,
            }
          )

  timing["score_wall_seconds"] = timing.get("score_wall_seconds", 0.0) + (time.perf_counter() - t0)
  timing["forward_count"] = forward_count
  write_jsonl(cache_path, list(cache.values()))
  return aggregates, count_excluded(items)


def aggregate_doc_g(score_rows: list[dict]) -> list[dict]:
  by_doc: dict[tuple[str, int, str], list[float]] = defaultdict(list)
  meta: dict[tuple[str, int, str], dict] = {}
  for row in score_rows:
    key = (row["item_id"], row["K"], row["source"])
    by_doc[key].append(row["G"])
    meta[key] = row
  out: list[dict] = []
  for key, gs in by_doc.items():
    item_id, k, source = key
    m = meta[key]
    out.append(
      {
        "item_id": item_id,
        "K": k,
        "source": source,
        "G_doc": float(np.mean(gs)),
        "n_blocks": len(gs),
        "delta_chars": m["delta_chars"],
        "edit_distance_norm": m["edit_distance_norm"],
        "split": m["split"],
      }
    )
  return out


def build_report(
  *,
  doc_rows: list[dict],
  cfg: RunConfig,
  excluded_k_lt4: int,
  n_items: int,
  timing: dict[str, float],
  extrap: dict[str, Any],
  pilot_n: int | None,
) -> tuple[str, dict]:
  by_source: dict[str, list[float]] = defaultdict(list)
  for row in doc_rows:
    if row["source"] == "human":
      by_source["human"].append(row["G_doc"])
    elif row["source"] == "composer":
      by_source["composer"].append(row["G_doc"])
    else:
      by_source["draft"].append(row["G_doc"])

  paired: dict[int, list[tuple[float, float, float, str]]] = defaultdict(list)
  for row in doc_rows:
    if row["source"] not in ("human", "composer"):
      continue
    paired[row["K"]].append((row["item_id"], row["G_doc"], row["delta_chars"], row["source"]))

  human_by_k: dict[int, dict[str, float]] = defaultdict(dict)
  comp_by_k: dict[int, dict[str, float]] = defaultdict(dict)
  for row in doc_rows:
    if row["source"] == "human":
      human_by_k[row["K"]][row["item_id"]] = row["G_doc"]
    if row["source"] == "composer":
      comp_by_k[row["K"]][row["item_id"]] = row["G_doc"]

  wilcoxon_by_k: dict[int, dict] = {}
  cliff_by_k: dict[int, float | None] = {}
  sign_by_k: dict[int, str] = {}
  for k in cfg.k_values:
    ids = sorted(set(human_by_k[k]) & set(comp_by_k[k]))
    hx = [human_by_k[k][i] for i in ids]
    cy = [comp_by_k[k][i] for i in ids]
    cliff_by_k[k] = cliffs_delta_paired(hx, cy)
    if scipy_stats and len(hx) >= 5:
      stat = scipy_stats.wilcoxon(hx, cy, alternative="greater", zero_method="wilcox")
      wilcoxon_by_k[k] = {"statistic": float(stat.statistic), "pvalue": float(stat.pvalue), "n": len(hx)}
      sign_by_k[k] = "human>composer" if float(np.mean(hx)) > float(np.mean(cy)) else "composer>=human"
    else:
      wilcoxon_by_k[k] = {"statistic": None, "pvalue": None, "n": len(hx)}
      sign_by_k[k] = "inconclusive"

  merged = []
  for row in doc_rows:
    if row["source"] != "human":
      continue
    cid = row["item_id"]
    k = row["K"]
    gc = comp_by_k[k].get(cid)
    merged.append(
      {
        "item_id": cid,
        "K": k,
        "delta_chars": row["delta_chars"],
        "edit_distance_norm": row["edit_distance_norm"],
        "G_human": row["G_doc"],
        "G_composer": gc,
      }
    )

  partial = {}
  for k in cfg.k_values:
    sub = [m for m in merged if m["K"] == k and m.get("G_composer") is not None]
    if len(sub) < 4:
      partial[str(k)] = None
      continue
    partial[str(k)] = partial_spearman(
      [m["G_human"] for m in sub],
      [m["delta_chars"] for m in sub],
      [m["edit_distance_norm"] for m in sub],
    )

  strata = decile_strata([m for m in merged if m.get("G_composer") is not None and m["K"] == cfg.k_values[0]])

  signs = [1 if (s.get("mean_G_diff_human_minus_composer") or 0) > 0 else -1 for s in strata if s.get("mean_G_diff_human_minus_composer") is not None]
  sign_consistency = sum(1 for s in signs if s > 0) / len(signs) if signs else None

  k_signs = [sign_by_k[k] for k in cfg.k_values]
  k_sign_match = len(set(k_signs)) == 1 and "inconclusive" not in k_signs

  primary_k = cfg.k_values[0]
  pval = wilcoxon_by_k.get(primary_k, {}).get("pvalue")
  cliff = cliff_by_k.get(primary_k)
  partial_primary = partial.get(str(primary_k))

  accept = {
    "G_human_gt_composer_p001": bool(pval is not None and pval < 0.01 and (cliff or 0) > 0.3),
    "partial_corr_abs_lt_02": bool(partial_primary is not None and abs(partial_primary) < 0.2),
    "decile_sign_consistency_ge_07": bool(sign_consistency is not None and sign_consistency >= 0.7),
    "K_sign_stable": bool(k_sign_match),
  }
  n_accept = sum(accept.values())
  if n_accept == 4:
    verdict = "support"
  elif n_accept == 0:
    verdict = "reject"
  else:
    verdict = "inconclusive"

  summary = {
    "n_items_scored": n_items,
    "excluded_k_lt4": excluded_k_lt4,
    "model_id": cfg.model_id,
    "k_values": list(cfg.k_values),
    "degenerate_steps": cfg.degenerate_steps,
    "distribution_quartiles": {s: quartiles(by_source[s]) for s in SOURCES},
    "wilcoxon_human_vs_composer": wilcoxon_by_k,
    "cliffs_delta": cliff_by_k,
    "partial_spearman_G_vs_delta_chars_control_ed": partial,
    "decile_strata": strata,
    "decile_sign_consistency": sign_consistency,
    "K_sign_stability": sign_by_k,
    "acceptance": accept,
    "verdict": verdict,
    "timing": timing,
    "extrapolation": extrap,
    "pilot_n": pilot_n,
  }

  qrows = []
  for s in SOURCES:
    q = summary["distribution_quartiles"][s]
    qrows.append([s, q["n"], fmt_pct(q["mean"]), fmt_pct(q["median"]), fmt_pct(q["q1"]), fmt_pct(q["q3"])])

  lines = [
    "# ead-glob-mi",
    "",
    "Phase 1: 大局性（条件付き相互情報量）の検証。",
    "",
    "## 実行設定",
    "",
    md_table(
      ["項目", "値"],
      [
        ["モデル", cfg.model_id],
        ["K", ", ".join(str(k) for k in cfg.k_values)],
        ["縮退", ", ".join(cfg.degenerate_steps) if cfg.degenerate_steps else "なし"],
        ["対象節数", str(n_items)],
        ["K<4 除外", str(excluded_k_lt4)],
      ],
    ),
    "",
    "## パイロットと外挿",
    "",
    md_table(
      ["項目", "値"],
      [
        ["パイロット節数", str(pilot_n) if pilot_n else "—"],
        ["実測 forward 数", str(timing.get("forward_count", "—"))],
        ["forward 合計秒", fmt_pct(timing.get("forward_seconds"))],
        ["外挿総時間 h", fmt_pct(extrap.get("hours"))],
        ["予算 h", str(BUDGET_HOURS)],
      ],
    ),
    "",
    "## G の分布（文書平均）",
    "",
    md_table(["source", "n", "mean", "median", "q1", "q3"], qrows),
    "",
    "## Wilcoxon（human vs composer）と Cliff's delta",
    "",
    md_table(
      ["K", "n", "p", "Cliff's delta", "符号"],
      [
        [
          k,
          wilcoxon_by_k[k]["n"],
          fmt_pct(wilcoxon_by_k[k]["pvalue"]),
          fmt_pct(cliff_by_k[k]),
          sign_by_k[k],
        ]
        for k in cfg.k_values
      ],
    ),
    "",
    "## 偏相関 Spearman: G vs Δ文字数 | edit_distance",
    "",
    md_table(["K", "partial_r"], [[k, fmt_pct(partial.get(str(k)))] for k in cfg.k_values]),
    "",
    "## Δ文字数十分位の群間差",
    "",
    md_table(
      ["decile", "n", "mean(G_human-G_composer)"],
      [
        [s["decile"], s["n"], fmt_pct(s.get("mean_G_diff_human_minus_composer"))]
        for s in strata
      ],
    ),
    "",
    f"符号一致率: {fmt_pct(sign_consistency)}",
    "",
    "## K 別符号安定性",
    "",
    md_table(["K", "符号"], [[k, sign_by_k[k]] for k in cfg.k_values]),
    "",
    "## Acceptance criteria (P1)",
    "",
    md_table(
      ["条件", "結果"],
      [
        ["G(人間)>G(Composer) p<0.01, Cliff's δ>0.3", "pass" if accept["G_human_gt_composer_p001"] else "fail"],
        ["|偏相関(G, Δ文字)|<0.2", "pass" if accept["partial_corr_abs_lt_02"] else "fail"],
        ["十分位符号一致 ≥7/10", "pass" if accept["decile_sign_consistency_ge_07"] else "fail"],
        ["K=4,8,16 符号一致", "pass" if accept["K_sign_stable"] else "fail"],
      ],
    ),
    "",
    f"**判定**: {verdict}",
  ]
  return "\n".join(lines) + "\n", summary


def try_load_scorer(model_id: str, device: torch.device) -> tuple[CausalScorer, str | None]:
  import gc

  try:
    return CausalScorer(model_id, device), None
  except torch.cuda.OutOfMemoryError:
    if model_id != MODEL_4B:
      gc.collect()
      if device.type == "cuda":
        torch.cuda.empty_cache()
      return CausalScorer(MODEL_4B, device), f"OOM on {model_id}"
    raise


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--pilot", type=int, default=0, help="パイロット節数。--pilot-only と併用。")
  parser.add_argument("--pilot-only", action="store_true", help="パイロットのみで終了。")
  parser.add_argument("--model", default=MODEL_8B)
  args = parser.parse_args()

  root = args.root.resolve()
  device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
  items, _ = load_sections(root)
  items.sort(key=lambda x: x.item_id)

  cfg = RunConfig(model_id=args.model)
  cache_path = ead_work() / "glob_mi_scores.jsonl"
  cache = load_cache(cache_path)
  timing: dict[str, float] = {}
  reports = ead_reports()
  reports.mkdir(parents=True, exist_ok=True)

  pilot_only = args.pilot_only or args.pilot > 0
  estimate_n = args.pilot if args.pilot > 0 else 20
  run_items = items[: args.pilot] if pilot_only else items
  estimate_items = run_items if pilot_only else items[:estimate_n]

  scorer, oom_reason = try_load_scorer(cfg.model_id, device)
  if oom_reason:
    cfg.model_id = MODEL_4B
    cfg.degenerate_steps.append(oom_reason)

  timing_estimate: dict[str, float] = {}
  score_rows, _ = score_sections(
    scorer, estimate_items, cfg, cache=cache, cache_path=cache_path, timing=timing_estimate
  )
  fwd = max(int(timing_estimate.get("forward_count", 0)), 1)
  sec_per_fwd = timing_estimate.get("forward_seconds", 0.0) / fwd
  avg_blocks = 6.0
  total_forwards = estimate_forwards(len(items), avg_blocks, cfg.k_values, cfg.cond_seeds)
  extrap_hours = total_forwards * sec_per_fwd / 3600.0
  extrap: dict[str, Any] = {
    "hours": extrap_hours,
    "total_forwards_est": total_forwards,
    "sec_per_forward": sec_per_fwd,
    "estimate_sections": len(estimate_items),
  }

  cfg_after = apply_degenerates(cfg, extrap_hours)
  if cfg_after.model_id != scorer.model_id:
    if pilot_only:
      cfg_after.degenerate_steps.append(f"pilot kept {scorer.model_id} for timing")
    else:
      import gc

      del scorer
      gc.collect()
      if device.type == "cuda":
        torch.cuda.empty_cache()
      scorer, oom2 = try_load_scorer(cfg_after.model_id, device)
      if oom2:
        cfg_after.degenerate_steps.append(oom2)
  cfg = cfg_after

  recalc_forwards = estimate_forwards(len(items), avg_blocks, cfg.k_values, cfg.cond_seeds)
  extrap_hours2 = recalc_forwards * sec_per_fwd / 3600.0
  extrap["hours_after_degenerate"] = extrap_hours2
  extrap["total_forwards_after_degenerate"] = recalc_forwards

  if extrap_hours2 > BUDGET_HOURS and not pilot_only:
    msg = {
      "error": "extrapolated_hours_exceeds_budget",
      "hours": extrap_hours2,
      "budget_hours": BUDGET_HOURS,
      "degenerate_steps": cfg.degenerate_steps,
      "action": "停止。縮退規則3段適用後も24h超。§10に従い報告のみ。",
    }
    write_json(reports / "ead-glob-mi.json", msg)
    (reports / "ead-glob-mi.md").write_text(
      "# ead-glob-mi\n\n外挿 "
      f"{extrap_hours2:.1f}h > {BUDGET_HOURS}h のため全件実行を停止。\n",
      encoding="utf-8",
    )
    print(json.dumps(msg, ensure_ascii=False))
    raise SystemExit(2)

  timing = timing_estimate
  if not pilot_only:
    score_rows, _ = score_sections(
      scorer, items, cfg, cache=cache, cache_path=cache_path, timing=timing
    )

  doc_rows = aggregate_doc_g(score_rows)
  md, summary = build_report(
    doc_rows=doc_rows,
    cfg=cfg,
    excluded_k_lt4=count_excluded(items),
    n_items=len(run_items),
    timing=timing,
    extrap=extrap,
    pilot_n=estimate_n if pilot_only else estimate_n,
  )
  summary["block_aggregates"] = score_rows
  summary["doc_aggregates"] = doc_rows
  write_json(reports / "ead-glob-mi.json", summary)
  (reports / "ead-glob-mi.md").write_text(md, encoding="utf-8")

  sample_g = [
    {"item_id": r["item_id"], "K": r["K"], "source": r["source"], "G_doc": r["G_doc"]}
    for r in doc_rows[:6]
  ]
  print(
    json.dumps(
      {
        "wrote_md": str(reports / "ead-glob-mi.md"),
        "wrote_json": str(reports / "ead-glob-mi.json"),
        "verdict": summary["verdict"],
        "forwards": timing.get("forward_count"),
        "wall_s": timing.get("score_wall_seconds"),
        "sample_G": sample_g,
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
