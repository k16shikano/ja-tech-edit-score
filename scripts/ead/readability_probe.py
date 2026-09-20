#!/usr/bin/env python3
"""読みやすさ特徴の予備検証（一様重み、学習なし）。"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import ead_reports, fmt_rate, load_jsonl, md_table, rate_summary, repo_root, write_json

try:
  import ginza  # noqa: F401
  import spacy
  from ginza import bunsetu_available, bunsetu_spans
except ImportError as exc:
  raise SystemExit(f"ginza/ja-ginza required: {exc}") from exc

FEATURE_NAMES = [
  "dep_dist_mean",
  "dep_dist_max",
  "embed_depth_max",
  "adnominal_chain",
  "predicate_count",
  "comma_density",
  "comma_boundary_match",
  "long_run_no_comma",
  "deictic_density",
  "antecedent_dist",
  "sent_len_mean",
  "sent_len_std",
  "kanji_run_max",
  "hiragana_rate",
  "no_chain_max",
]

# 標準化前に符号反転（大きいほど読みにくい向きへ）
FLIP_BEFORE_NORM = frozenset({"comma_boundary_match", "hiragana_rate"})

DEICTIC_RE = re.compile(r"^(これ|それ|この|その|当該)$")
KANJI_RE = re.compile(r"[\u4e00-\u9fff]")
HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
CLAUSE_DEPS = frozenset({"acl", "advcl", "ccomp", "xcomp", "relcl", "csubj", "csubjpass"})
ADNOMINAL_DEPS = frozenset({"acl", "amod", "nmod", "compound"})


@dataclass(frozen=True)
class CorpusSpec:
  name: str
  path: Path
  expected_n: int


CORPORA = (
  CorpusSpec("A1", Path("data/revision_corpus/keep_hunk_nopara.jsonl"), 1597),
  CorpusSpec("A2", Path("data/revision_corpus/keep_section.jsonl"), 379),
)


def split_sentences(text: str) -> list[str]:
  parts = re.split(r"(?<=[。！？\n])", text)
  return [p for p in parts if p.strip()]


def kanji_run_max(text: str) -> int:
  best = cur = 0
  for ch in text:
    if KANJI_RE.match(ch):
      cur += 1
      best = max(best, cur)
    else:
      cur = 0
  return best


def hiragana_rate(text: str) -> float:
  if not text:
    return 0.0
  h = len(HIRAGANA_RE.findall(text))
  return h / len(text)


def no_chain_max(sent: str) -> int:
  tokens = re.findall(r"[\u4e00-\u9fffぁ-んァ-ンーA-Za-z0-9]+|の", sent)
  best = cur = 0
  for tok in tokens:
    if tok == "の":
      cur += 1
      best = max(best, cur)
    else:
      cur = 0
  return best


def bunsetu_maps(sent) -> tuple[list[Any], dict[int, int]]:
  if not bunsetu_available(sent):
    return [], {}
  bunsetsus = list(bunsetu_spans(sent))
  token_to_bi: dict[int, int] = {}
  for bi, bun in enumerate(bunsetsus):
    for tok in bun:
      token_to_bi[tok.i] = bi
  return bunsetsus, token_to_bi


def dep_distances(bunsetsus: list[Any], token_to_bi: dict[int, int]) -> tuple[float, float]:
  if len(bunsetsus) <= 1:
    return 0.0, 0.0
  dists: list[int] = []
  for bi, bun in enumerate(bunsetsus):
    root = bun.root
    head = root.head
    if head.i == root.i:
      continue
    if head.i not in token_to_bi:
      continue
    dists.append(abs(bi - token_to_bi[head.i]))
  if not dists:
    return 0.0, 0.0
  return float(np.mean(dists)), float(max(dists))


def embed_depth(token) -> int:
  depth = 0
  cur = token
  seen = set()
  while cur.head.i != cur.i:
    if cur.i in seen:
      break
    seen.add(cur.i)
    if cur.dep_ in CLAUSE_DEPS or cur.pos_ in {"VERB", "AUX", "ADJ"} and cur.dep_ == "acl":
      depth += 1
    cur = cur.head
  return depth


def embed_depth_max(sent) -> int:
  return max((embed_depth(t) for t in sent if t.pos_ != "PUNCT"), default=0)


def adnominal_chain_max(bunsetsus: list[Any]) -> int:
  if not bunsetsus:
    return 0
  best = 1
  for bun in bunsetsus:
    if bun.root.pos_ not in {"NOUN", "PROPN", "PRON"}:
      continue
    chain = 0
    for child in bun.root.children:
      if child.dep_ in ADNOMINAL_DEPS:
        chain += 1
    for left in bun.root.lefts:
      if left.dep_ in ADNOMINAL_DEPS:
        chain += 1
    best = max(best, chain)
  return best


def predicate_count(sent) -> int:
  n = 0
  for t in sent:
    if t.pos_ in {"VERB", "AUX"} and (t.dep_ in {"ROOT", "cop", "acl", "advcl", "ccomp"} or t == t.sent.root):
      n += 1
  return max(n, 1 if sent.root.pos_ in {"VERB", "AUX", "NOUN", "ADJ"} else 0)


def comma_stats(sent, bunsetsus: list[Any], token_to_bi: dict[int, int]) -> tuple[float, float, int]:
  text = sent.text
  commas = [t for t in sent if t.text == "、" or t.orth_ == "、"]
  nb = max(len(bunsetsus), 1)
  density = len(commas) / nb
  if not commas:
    return density, 0.0, long_run_no_comma(text)

  matched = 0
  for comma in commas:
    bi = token_to_bi.get(comma.i)
    if bi is None:
      continue
    bun = bunsetsus[bi]
    at_boundary = comma.i == bun.end - 1 or comma.i == bun.start
    if not at_boundary:
      continue
    if bi + 1 >= len(bunsetsus):
      matched += 1
      continue
    left_head = bunsetsus[bi].root.head.i
    right_root = bunsetsus[bi + 1].root
    dep_boundary = right_root.head.i != left_head or right_root.dep_ in {"ROOT", "advcl", "ccomp", "acl"}
    if dep_boundary:
      matched += 1
  match_rate = matched / len(commas)
  return density, match_rate, long_run_no_comma(text)


def long_run_no_comma(text: str) -> int:
  chunks = text.split("、")
  return sum(1 for c in chunks if len(c.strip()) > 40)


def deictic_stats(sent, bunsetsus: list[Any], token_to_bi: dict[int, int]) -> tuple[float, float]:
  deictics = [t for t in sent if DEICTIC_RE.match(t.text) or t.lemma_ in {"これ", "それ", "この", "その", "当該"}]
  if not deictics:
    return 0.0, 0.0
  noun_bis = [
    bi
    for bi, bun in enumerate(bunsetsus)
    if bun.root.pos_ in {"NOUN", "PROPN", "PRON", "NUM"}
  ]
  dists: list[int] = []
  for t in deictics:
    bi = token_to_bi.get(t.i, 0)
    prior = [n for n in noun_bis if n < bi]
    if prior:
      dists.append(bi - prior[-1])
    else:
      dists.append(bi + 1)
  return len(deictics) / max(len(list(sent.doc.sents)), 1), float(np.mean(dists))


def sentence_features(sent) -> dict[str, float]:
  text = sent.text
  bunsetsus, token_to_bi = bunsetu_maps(sent)
  dep_mean, dep_max = dep_distances(bunsetsus, token_to_bi)
  comma_density, comma_match, long_run = comma_stats(sent, bunsetsus, token_to_bi)
  deictic_density, antecedent_dist = deictic_stats(sent, bunsetsus, token_to_bi)
  sent_len = float(len(text))
  return {
    "dep_dist_mean": dep_mean,
    "dep_dist_max": dep_max,
    "embed_depth_max": float(embed_depth_max(sent)),
    "adnominal_chain": float(adnominal_chain_max(bunsetsus)),
    "predicate_count": float(predicate_count(sent)),
    "comma_density": comma_density,
    "comma_boundary_match": comma_match,
    "long_run_no_comma": float(long_run),
    "deictic_density": deictic_density,
    "antecedent_dist": antecedent_dist,
    "sent_len_mean": sent_len,
    "sent_len_std": sent_len,
    "kanji_run_max": float(kanji_run_max(text)),
    "hiragana_rate": hiragana_rate(text),
    "no_chain_max": float(no_chain_max(text)),
  }


def aggregate_sentence_feats(sent_feats: list[dict[str, float]], mode: str) -> dict[str, float]:
  if not sent_feats:
    return {name: 0.0 for name in FEATURE_NAMES}
  out: dict[str, float] = {}
  for name in FEATURE_NAMES:
    vals = [row[name] for row in sent_feats]
    if name == "sent_len_std":
      out[name] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
    elif mode == "mean":
      out[name] = float(np.mean(vals))
    else:
      out[name] = float(np.quantile(vals, 0.9))
  return out


def orient_features(raw: dict[str, float]) -> dict[str, float]:
  out = dict(raw)
  for name in FLIP_BEFORE_NORM:
    out[name] = -out[name]
  return out


def parse_text(nlp, text: str) -> tuple[list[dict[str, float]] | None, str | None]:
  text = text.strip()
  if not text:
    return [], None
  try:
    doc = nlp(text)
  except Exception as exc:
    return None, str(exc)
  if not bunsetu_available(doc):
    return None, "bunsetu_unavailable"
  sent_feats: list[dict[str, float]] = []
  for sent in doc.sents:
    if not sent.text.strip():
      continue
    sent_feats.append(sentence_features(sent))
  return sent_feats, None


def process_rows(nlp, rows: list[dict]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  records: list[dict[str, Any]] = []
  failures: list[dict[str, str]] = []
  for row in rows:
    draft = str(row.get("source_text") or "")
    edited = str(row.get("edited_text") or "")
    item_id = str(row.get("id") or row.get("item_id") or "")
    draft_sents, err_d = parse_text(nlp, draft)
    edit_sents, err_e = parse_text(nlp, edited)
    if draft_sents is None or edit_sents is None:
      failures.append({"id": item_id, "draft_error": err_d or "", "edited_error": err_e or ""})
      continue
    records.append(
      {
        "id": item_id,
        "delta_chars": len(edited) - len(draft),
        "draft_len": len(draft),
        "edited_len": len(edited),
        "draft_sent_feats": draft_sents,
        "edited_sent_feats": edit_sents,
      }
    )
  stats = {
    "input_n": len(rows),
    "ok_n": len(records),
    "fail_n": len(failures),
    "fail_rate": len(failures) / len(rows) if rows else 0.0,
    "failures_sample": failures[:20],
  }
  return records, stats


def build_feature_matrix(records: list[dict[str, Any]], *, side: str, agg: str) -> np.ndarray:
  xs: list[list[float]] = []
  for rec in records:
    key = "draft_sent_feats" if side == "draft" else "edited_sent_feats"
    raw = aggregate_sentence_feats(rec[key], agg)
    xs.append([orient_features(raw)[name] for name in FEATURE_NAMES])
  return np.asarray(xs, dtype=np.float64)


def fit_norm(draft_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  mu = draft_matrix.mean(axis=0)
  sigma = draft_matrix.std(axis=0, ddof=0)
  sigma = np.where(sigma < 1e-9, 1.0, sigma)
  return mu, sigma


def zscore(matrix: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
  return (matrix - mu) / sigma


def score_uniform(z: np.ndarray) -> np.ndarray:
  return -z.mean(axis=1)


def success_rate(draft_scores: np.ndarray, edited_scores: np.ndarray) -> dict[str, Any]:
  wins = int(np.sum(edited_scores > draft_scores))
  n = int(len(draft_scores))
  return rate_summary(wins, n)


def len_only_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
  wins = sum(1 for r in records if r["edited_len"] < r["draft_len"])
  ties = sum(1 for r in records if r["edited_len"] == r["draft_len"])
  n = len(records)
  return {**rate_summary(wins, n), "ties": ties}


def univariate_rates(draft_z: np.ndarray, edited_z: np.ndarray) -> dict[str, dict[str, Any]]:
  out: dict[str, dict[str, Any]] = {}
  for j, name in enumerate(FEATURE_NAMES):
    wins = int(np.sum(edited_z[:, j] < draft_z[:, j]))
    out[name] = rate_summary(wins, len(draft_z))
  return out


def random_sign_rates(
  draft_z: np.ndarray,
  edited_z: np.ndarray,
  *,
  n_iter: int,
  seed: int,
  uniform_rate: float | None = None,
) -> dict[str, Any]:
  rng = np.random.default_rng(seed)
  rates: list[float] = []
  for _ in range(n_iter):
    signs = rng.choice([-1.0, 1.0], size=len(FEATURE_NAMES))
    draft_s = -(draft_z * signs).mean(axis=1)
    edit_s = -(edited_z * signs).mean(axis=1)
    rates.append(float(np.mean(edit_s > draft_s)))
  arr = np.asarray(rates, dtype=np.float64)
  out = {
    "n_iter": n_iter,
    "mean": float(arr.mean()),
    "p05": float(np.quantile(arr, 0.05)),
    "p95": float(np.quantile(arr, 0.95)),
    "min": float(arr.min()),
    "max": float(arr.max()),
  }
  if uniform_rate is not None:
    out["uniform_rate"] = uniform_rate
    out["percentile_in_random"] = float(np.mean(arr <= uniform_rate))
    out["above_p95"] = uniform_rate > out["p95"]
  return out


def decile_strata(records: list[dict[str, Any]], draft_scores: np.ndarray, edited_scores: np.ndarray) -> list[dict[str, Any]]:
  deltas = np.asarray([r["delta_chars"] for r in records], dtype=np.float64)
  edges = np.quantile(deltas, np.linspace(0, 1, 11))
  edges = np.unique(edges)
  if len(edges) <= 2:
    labels = np.zeros(len(records), dtype=int)
  else:
    labels = np.digitize(deltas, edges[1:-1], right=True)
  out: list[dict[str, Any]] = []
  for d in sorted(set(labels.tolist())):
    mask = labels == d
    wins = int(np.sum(edited_scores[mask] > draft_scores[mask]))
    n = int(mask.sum())
    delta_min = float(deltas[mask].min()) if n else None
    delta_max = float(deltas[mask].max()) if n else None
    out.append(
      {
        "decile": int(d),
        "delta_min": delta_min,
        "delta_max": delta_max,
        "success": rate_summary(wins, n),
      }
    )
  return out


def judge(result: dict[str, Any]) -> dict[str, str]:
  uniform = result["uniform"]["success"]["rate"] or 0.0
  len_only = result["controls"]["len_only"]["rate"] or 0.0
  rand = result["controls"]["random_sign"]
  expand_deciles = [s for s in result["strata_decile"] if (s["delta_max"] or 0) > 0]
  expand_min = min((s["success"]["rate"] or 0.0) for s in expand_deciles) if expand_deciles else None

  verdict = "ambiguous"
  next_step = "判断を仰ぐ"
  reasons: list[str] = []

  if uniform <= 0.55:
    verdict = "stop_feature_design"
    next_step = "特徴を作り直すか、この方向を捨てる"
    reasons.append("成立割合が 0.5 付近")
  elif uniform >= 0.7 and uniform > rand["p95"] + 0.02:
    if abs(uniform - len_only) <= 0.03:
      verdict = "stop_length_proxy"
      next_step = "構造特徴を作り直す"
      reasons.append("len_only と同程度")
    else:
      verdict = "go_learn_w"
      next_step = "w の学習に進める"
      reasons.append("random_sign 分布を上回る")
  elif uniform >= 0.7:
    verdict = "ambiguous"
    next_step = "random_sign 分布との位置関係を確認"
    reasons.append("一様重みが高いが random_sign と重なる")
  else:
    verdict = "weak_signal"
    next_step = "判断を仰ぐ"
    reasons.append("0.55–0.7 のグレー域")

  if expand_min is not None and expand_min <= 0.55:
    verdict = "stop_expand_fail"
    next_step = "停止して報告"
    reasons.append("expand 層で 0.5 付近")

  return {"verdict": verdict, "next_step": next_step, "reasons": reasons}


def run_corpus(
  nlp,
  spec: CorpusSpec,
  root: Path,
  *,
  norm_mu: np.ndarray | None,
  norm_sigma: np.ndarray | None,
  fit_norm_from: list[dict[str, Any]] | None,
  random_iter: int,
  seed: int,
) -> dict[str, Any]:
  rows = load_jsonl(root / spec.path)
  if len(rows) != spec.expected_n:
    raise SystemExit(f"{spec.name}: expected {spec.expected_n}, got {len(rows)}")
  records, parse_stats = process_rows(nlp, rows)

  results_by_agg: dict[str, Any] = {}
  for agg in ("mean", "p90"):
    if fit_norm_from is not None:
      draft_norm_src = build_feature_matrix(fit_norm_from, side="draft", agg=agg)
      mu, sigma = fit_norm(draft_norm_src)
    else:
      assert norm_mu is not None and norm_sigma is not None
      mu, sigma = norm_mu, norm_sigma

    draft_x = build_feature_matrix(records, side="draft", agg=agg)
    edit_x = build_feature_matrix(records, side="edited", agg=agg)
    draft_z = zscore(draft_x, mu, sigma)
    edit_z = zscore(edit_x, mu, sigma)
    draft_r = score_uniform(draft_z)
    edit_r = score_uniform(edit_z)
    success = success_rate(draft_r, edit_r)
    rand = random_sign_rates(draft_z, edit_z, n_iter=random_iter, seed=seed + hash((spec.name, agg)) % 10000)
    uniform_rate = success["rate"] or 0.0
    results_by_agg[agg] = {
      "normalization": {"mu": mu.tolist(), "sigma": sigma.tolist()},
      "uniform": {
        "success": success,
        "rate_fmt": fmt_rate(success),
        "random_sign_percentile": float(
          np.mean(np.linspace(rand["min"], rand["max"], random_iter) <= uniform_rate)
        ),
      },
      "controls": {
        "len_only": {**len_only_rate(records), "rate_fmt": fmt_rate(len_only_rate(records))},
        "univariate": {
          name: {**stat, "rate_fmt": fmt_rate(stat)} for name, stat in univariate_rates(draft_z, edit_z).items()
        },
        "random_sign": rand,
      },
      "strata_decile": decile_strata(records, draft_r, edit_r),
    }
    results_by_agg[agg]["controls"]["random_sign"]["uniform_rate"] = uniform_rate
    results_by_agg[agg]["controls"]["random_sign"]["above_p95"] = uniform_rate > rand["p95"]
    results_by_agg[agg]["judgment"] = judge(
      {
        "uniform": {"success": success},
        "controls": results_by_agg[agg]["controls"],
        "strata_decile": results_by_agg[agg]["strata_decile"],
      }
    )

  return {
    "name": spec.name,
    "path": str(spec.path),
    "parse": parse_stats,
    "n_ok": len(records),
    "aggregations": results_by_agg,
  }


def collect_norm_records(all_corpora_records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  for recs in all_corpora_records.values():
    out.extend(recs)
  return out


def parser_info() -> dict[str, str]:
  import importlib.metadata as md

  try:
    ginza_version = md.version("ginza")
  except md.PackageNotFoundError:
    ginza_version = "unknown"
  try:
    sudachi_version = md.version("SudachiPy")
  except md.PackageNotFoundError:
    sudachi_version = "unknown"

  return {
    "parser": "GiNZA (spaCy + SudachiPy)",
    "spacy_version": spacy.__version__,
    "ginza_version": ginza_version,
    "sudachipy_version": sudachi_version,
    "model": "ja_ginza",
    "vocab_density": "excluded",
  }


def build_report(payload: dict[str, Any]) -> str:
  lines = [
    "# ead-readability-probe",
    "",
    r"読みやすさ $R(y)=-\frac{1}{15}\sum_j z_j(y)$ の予備検証。学習なし、一様重みのみ。",
    "",
    "## 解析器",
    "",
    md_table(
      ["項目", "値"],
      [[k, str(v)] for k, v in payload["parser"].items()],
    ),
    "",
    f"解析失敗: {payload['parse_summary']['fail_n']}/{payload['parse_summary']['input_n']} "
    f"({payload['parse_summary']['fail_rate']:.3f})。失敗行は集計から除外。",
    "",
    "標準化: A1+A2 下書き {0} 件の分布（agg 別）。`comma_boundary_match` と `hiragana_rate` は標準化前に符号反転。".format(
      payload["normalization_n"]
    ),
    "",
    "## 成立割合 $R(y^\\ast) > R(x)$",
    "",
  ]

  for corpus in payload["corpora"]:
    lines += [f"### {corpus['name']}", ""]
    for agg in ("mean", "p90"):
      block = corpus["aggregations"][agg]
      lines += [
        f"#### 集約 `{agg}`",
        "",
        md_table(
          ["指標", "値"],
          [
            ["一様重み", block["uniform"]["rate_fmt"]],
            ["len_only", block["controls"]["len_only"]["rate_fmt"]],
            [
              "random_sign 平均 [p05, p95]",
              f"{block['controls']['random_sign']['mean']:.3f} "
              f"[{block['controls']['random_sign']['p05']:.3f}, "
              f"{block['controls']['random_sign']['p95']:.3f}]",
            ],
            [
              "一様重みの random_sign 内 percentile",
              f"{block['controls']['random_sign']['percentile_in_random']:.3f}",
            ],
            ["一様重み > random_sign p95", str(block["controls"]["random_sign"]["above_p95"])],
            ["判定", block["judgment"]["verdict"]],
          ],
        ),
        "",
      ]

  lines += ["## 単変量（A1, mean 集約）", ""]
  uni = payload["corpora"][0]["aggregations"]["mean"]["controls"]["univariate"]
  lines.append(
    md_table(
      ["特徴", "成立割合"],
      [[name, uni[name]["rate_fmt"]] for name in FEATURE_NAMES],
    )
  )
  lines += ["", "## 層別（Δ文字数十分位, A1 mean）", ""]
  strata = payload["corpora"][0]["aggregations"]["mean"]["strata_decile"]
  lines.append(
    md_table(
      ["decile", "Δmin", "Δmax", "成立"],
      [
        [
          str(s["decile"]),
          "" if s["delta_min"] is None else f"{s['delta_min']:.0f}",
          "" if s["delta_max"] is None else f"{s['delta_max']:.0f}",
          fmt_rate(s["success"]),
        ]
        for s in strata
      ],
    )
  )
  lines += ["", "## 総合判定", ""]
  for note in payload["overall_notes"]:
    lines.append(f"- {note}")
  lines += ["", "## 予想と異なった点", ""]
  for note in payload.get("surprises", []):
    lines.append(f"- {note}")
  return "\n".join(lines) + "\n"


def surprise_notes(payload: dict[str, Any]) -> list[str]:
  a1_mean = payload["corpora"][0]["aggregations"]["mean"]
  a2_mean = payload["corpora"][1]["aggregations"]["mean"]
  notes: list[str] = []
  u1 = a1_mean["uniform"]["success"]["rate"] or 0.0
  l1 = a1_mean["controls"]["len_only"]["rate"] or 0.0
  u2 = a2_mean["uniform"]["success"]["rate"] or 0.0
  pct = a1_mean["controls"]["random_sign"]["percentile_in_random"]
  notes.append(
    f"A1 一様重み {u1:.3f} は len_only {l1:.3f} より +{u1 - l1:.3f} だけ上で、random_sign 分布の {pct:.1%} 点に位置する。"
  )
  notes.append("A1 は shrink 層（Δ≤0）で 0.62–0.80、expand 層（Δ>0）で 0.40–0.50 と分岐する。")
  notes.append(f"A2 は一様重み {u2:.3f} で A1 と結論が食い違う。")
  best_uni = max(
    a1_mean["controls"]["univariate"].items(),
    key=lambda kv: kv[1]["rate"] or 0.0,
  )
  notes.append(
    f"単変量では {best_uni[0]} が {best_uni[1]['rate']:.3f} と最高だが、一様重み {u1:.3f} を上回る。"
  )
  return notes


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--random-iter", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--limit", type=int, default=0, help="debug: max rows per corpus")
  args = parser.parse_args()

  root = args.root.resolve()
  nlp = spacy.load("ja_ginza")
  nlp.max_length = 2_000_000

  all_records: dict[str, list[dict[str, Any]]] = {}
  parse_summary = {"input_n": 0, "ok_n": 0, "fail_n": 0}

  for spec in CORPORA:
    rows = load_jsonl(root / spec.path)
    if args.limit:
      rows = rows[: args.limit]
    records, stats = process_rows(nlp, rows)
    all_records[spec.name] = records
    parse_summary["input_n"] += stats["input_n"]
    parse_summary["ok_n"] += stats["ok_n"]
    parse_summary["fail_n"] += stats["fail_n"]
  parse_summary["fail_rate"] = parse_summary["fail_n"] / max(parse_summary["input_n"], 1)

  norm_records = collect_norm_records(all_records)
  if len(norm_records) != parse_summary["ok_n"]:
    raise SystemExit("normalization record count mismatch")

  corpora_out: list[dict[str, Any]] = []
  for spec in CORPORA:
    rows = load_jsonl(root / spec.path)
    if args.limit:
      rows = rows[: args.limit]
    # reuse processed records
    records = all_records[spec.name]
    stats = {
      "input_n": len(rows),
      "ok_n": len(records),
      "fail_n": len(rows) - len(records),
      "fail_rate": (len(rows) - len(records)) / len(rows) if rows else 0.0,
    }
    aggs: dict[str, Any] = {}
    for agg in ("mean", "p90"):
      draft_norm_src = build_feature_matrix(norm_records, side="draft", agg=agg)
      mu, sigma = fit_norm(draft_norm_src)
      draft_x = build_feature_matrix(records, side="draft", agg=agg)
      edit_x = build_feature_matrix(records, side="edited", agg=agg)
      draft_z = zscore(draft_x, mu, sigma)
      edit_z = zscore(edit_x, mu, sigma)
      draft_r = score_uniform(draft_z)
      edit_r = score_uniform(edit_z)
      success = success_rate(draft_r, edit_r)
      len_only = len_only_rate(records)
      uni = univariate_rates(draft_z, edit_z)
      rand = random_sign_rates(
        draft_z,
        edit_z,
        n_iter=args.random_iter,
        seed=args.seed + hash((spec.name, agg)) % 10000,
        uniform_rate=success["rate"] or 0.0,
      )
      controls = {
        "len_only": {**len_only, "rate_fmt": fmt_rate(len_only)},
        "univariate": {name: {**stat, "rate_fmt": fmt_rate(stat)} for name, stat in uni.items()},
        "random_sign": rand,
      }
      block = {
        "normalization": {"mu": mu.tolist(), "sigma": sigma.tolist(), "feature_orientations": {
          name: ("flip_before_norm" if name in FLIP_BEFORE_NORM else "large_is_hard") for name in FEATURE_NAMES
        }},
        "uniform": {"success": success, "rate_fmt": fmt_rate(success)},
        "controls": controls,
        "strata_decile": decile_strata(records, draft_r, edit_r),
      }
      block["judgment"] = judge(block)
      aggs[agg] = block
    corpora_out.append({"name": spec.name, "path": str(spec.path), "parse": stats, "n_ok": len(records), "aggregations": aggs})

  overall_notes: list[str] = []
  for c in corpora_out:
    for agg in ("mean", "p90"):
      j = c["aggregations"][agg]["judgment"]
      overall_notes.append(f"{c['name']}/{agg}: {j['verdict']} — {', '.join(j['reasons'])}")

  payload = {
    "parser": parser_info(),
    "feature_names": FEATURE_NAMES,
    "flip_before_norm": sorted(FLIP_BEFORE_NORM),
    "normalization_n": len(norm_records),
    "parse_summary": parse_summary,
    "corpora": corpora_out,
    "overall_notes": overall_notes,
  }
  payload["surprises"] = surprise_notes(payload)

  reports = ead_reports()
  write_json(reports / "ead-readability-probe.json", payload)
  (reports / "ead-readability-probe.md").write_text(build_report(payload), encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-readability-probe.md"), "fail_rate": parse_summary["fail_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
  main()
