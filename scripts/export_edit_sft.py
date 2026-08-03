#!/usr/bin/env python3
"""系統1フェーズ0: 推敲ペアを編集モデル SFT 形式へ書き出す。

節でも hunk でもよい。要件は壊れた教師を入れないこと
（対応破綻・比較点の誤り）。段落境界の有無それ自体は除外理由にしない。

- 既定入力: data/examples.section.raw.jsonl
- --include-hunk: revision_pairs 等を合流（対応が健全なものを前提）
- 出力: chat messages 形式の train / heldout JSONL と stats / filter_report

節ペアでも、見出しは同じで中身がほぼ無変更のものや、短すぎる節は
教師として弱いので落とす。revised の地の文は落とさず、コードと図は
一律プレースホルダに置換して推敲学習の対象外にする。
同一下書きに複数の edit ブランチ由来が並ぶ場合は 1 件に潰す（SFT は一対一）。
選好用の dpo_curated / pref_dataset は変更しない。
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from align_nonprose_to_draft import align_nonprose_to_draft
from markdown_sections import paragraph_count
from mask_code_figures import mask_pair
from steering_utils import strip_reference_block

INSTRUCTION = "次の下書きを、意味を保ったまま日本語の技術文書として推敲せよ。"

# held-out は編集の重さが偏らないように選ぶ（節データに存在する書籍）:
#   what-is-monad / computer-arch-revisit / ir-system
DEFAULT_HOLDOUT = ["what-is-monad", "computer-arch-revisit", "ir-system"]

# 節主軸の編集 SFT 用フィルタ
DEFAULT_MIN_SIM = 0.15
DEFAULT_MAX_SIM = 0.985
DEFAULT_MIN_JACCARD = 0.05
DEFAULT_MIN_LEN_RATIO = 0.4
DEFAULT_MAX_LEN_RATIO = 2.5
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 8000
DEFAULT_MIN_PARAGRAPHS = 2


def _has_near_dup_prose(text: str, *, threshold: float = 0.72) -> bool:
  """同一 revised 内に、ほぼ同じ地の文が二重でないか。"""
  parts = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 40]
  # コード・画像だけの塊は除外
  prose = []
  for p in parts:
    if p.startswith("```") or p.startswith("!["):
      continue
    prose.append(re.sub(r"\s+", " ", p))
  for i, a in enumerate(prose):
    for b in prose[i + 1 :]:
      if difflib.SequenceMatcher(None, a, b).ratio() >= threshold:
        return True
  return False


def char_bigram_jaccard(a: str, b: str) -> float:
  def grams(s: str) -> set[str]:
    s = re.sub(r"\s+", " ", s)
    if len(s) < 2:
      return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}

  A, B = grams(a), grams(b)
  if not A or not B:
    return 0.0
  return len(A & B) / len(A | B)


def is_heading_only(text: str) -> bool:
  lines = [ln for ln in text.strip().splitlines() if ln.strip()]
  return len(lines) == 1 and lines[0].lstrip().startswith("#")


def looks_footnote(text: str) -> bool:
  s = text.strip()
  if s.startswith("[^"):
    return True
  if re.match(r"^\[[^\]]+\]:", s):
    return True
  return False


def looks_image_only(text: str) -> bool:
  s = text.strip()
  if not s.startswith("!["):
    return False
  return s.count("\n") <= 1


def pair_metrics(draft: str, revised: str) -> dict[str, float]:
  return {
    "char_similarity": float(difflib.SequenceMatcher(None, draft, revised).ratio()),
    "bigram_jaccard": float(char_bigram_jaccard(draft, revised)),
    "length_ratio": float(len(revised) / max(len(draft), 1)),
  }


def accept_edit_pair(
  draft: str,
  revised: str,
  *,
  min_sim: float,
  max_sim: float,
  min_jaccard: float,
  min_len_ratio: float,
  max_len_ratio: float,
  min_chars: int,
  max_chars: int,
  min_paragraphs: int,
) -> tuple[bool, str, dict[str, float]]:
  """編集 SFT 用に pair を残すか。戻り値: (採用?, 理由, 指標)。"""
  d = draft.strip()
  r = revised.strip()
  metrics = pair_metrics(d, r) if d and r else {
    "char_similarity": 0.0,
    "bigram_jaccard": 0.0,
    "length_ratio": 0.0,
  }
  if not d or not r:
    return False, "empty", metrics
  if d == r:
    return False, "identical", metrics
  if _has_near_dup_prose(r):
    return False, "revised_internal_dup", metrics
  if len(d) < min_chars or len(r) < min_chars:
    return False, "too_short", metrics
  if len(d) > max_chars or len(r) > max_chars:
    return False, "too_long", metrics
  n_para = min(paragraph_count(d), paragraph_count(r))
  if n_para < min_paragraphs:
    return False, "too_few_paragraphs", metrics
  if looks_footnote(d) or looks_footnote(r):
    return False, "footnote", metrics
  if looks_image_only(d) or looks_image_only(r):
    return False, "image", metrics
  if is_heading_only(d) or is_heading_only(r):
    return False, "heading_only", metrics
  lr = metrics["length_ratio"]
  if lr < min_len_ratio or lr > max_len_ratio:
    return False, "length_ratio", metrics
  if metrics["char_similarity"] < min_sim:
    return False, "similarity_low", metrics
  if metrics["char_similarity"] > max_sim:
    return False, "similarity_high", metrics
  if metrics["bigram_jaccard"] < min_jaccard:
    return False, "jaccard", metrics
  return True, "ok", metrics


def _default_pair_rank(pair: dict) -> tuple:
  """export 時の代表選好: 変更量→revised 長→id。"""
  d = (pair.get("draft") or "").strip()
  r = (pair.get("revised") or "").strip()
  return (abs(len(r) - len(d)), len(r), str(pair.get("id") or ""))


def _norm_for_dedupe(text: str) -> str:
  t = re.sub(r"【コード】|【図】", "", text or "")
  return re.sub(r"\s+", " ", t).strip()


def _text_sim(a: str, b: str) -> float:
  if not a or not b:
    return 0.0
  if a == b:
    return 1.0
  if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.45:
    return 0.0
  if difflib.SequenceMatcher(None, a[:120], b[:120]).ratio() < 0.75:
    return 0.0
  return difflib.SequenceMatcher(None, a, b).ratio()


def dedupe_sft_pairs(
  pairs: list[dict],
  *,
  rank: Callable[[dict], tuple] | None = None,
  near_draft_sim: float = 0.85,
  chain_sim: float = 0.95,
) -> tuple[list[dict], dict]:
  """学習阻害になる重複・連鎖を潰す。

  1. (draft, revised) 完全一致 → 1 件
  2. 同一 draft → 1 件
  3. 同一 project+path 内で、draft が近い／一方の revised≈他方の draft
     （複数 edit ブランチの同節・推敲連鎖）→ クラスタ 1 件
  """
  rank_fn = rank or _default_pair_rank
  stats: dict[str, int] = {
    "before": len(pairs),
    "dropped_exact": 0,
    "dropped_same_draft": 0,
    "dropped_near_or_chain": 0,
    "near_draft_sim": near_draft_sim,
    "chain_sim": chain_sim,
  }

  by_exact: dict[tuple[str, str], dict] = {}
  for p in pairs:
    key = ((p.get("draft") or "").strip(), (p.get("revised") or "").strip())
    prev = by_exact.get(key)
    if prev is None or rank_fn(p) > rank_fn(prev):
      if prev is not None:
        stats["dropped_exact"] += 1
      by_exact[key] = p
    else:
      stats["dropped_exact"] += 1

  by_draft: dict[str, dict] = {}
  for p in by_exact.values():
    d = (p.get("draft") or "").strip()
    prev = by_draft.get(d)
    if prev is None or rank_fn(p) > rank_fn(prev):
      if prev is not None:
        stats["dropped_same_draft"] += 1
      by_draft[d] = p
    else:
      stats["dropped_same_draft"] += 1

  stage = list(by_draft.values())
  n = len(stage)
  parent = list(range(n))

  def find(i: int) -> int:
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i

  def union(i: int, j: int) -> None:
    ri, rj = find(i), find(j)
    if ri != rj:
      parent[rj] = ri

  norms = [
    (_norm_for_dedupe(p.get("draft") or ""), _norm_for_dedupe(p.get("revised") or ""))
    for p in stage
  ]
  groups: dict[tuple[str, str], list[int]] = {}
  for i, p in enumerate(stage):
    key = (str(p.get("project_id") or ""), str(p.get("path") or ""))
    groups.setdefault(key, []).append(i)

  for idxs in groups.values():
    for a in range(len(idxs)):
      ia = idxs[a]
      da, ra = norms[ia]
      for b in range(a + 1, len(idxs)):
        ib = idxs[b]
        db, rb = norms[ib]
        if _text_sim(da, db) >= near_draft_sim:
          union(ia, ib)
          continue
        if _text_sim(ra, rb) >= near_draft_sim:
          union(ia, ib)
          continue
        if _text_sim(da, rb) >= chain_sim or _text_sim(db, ra) >= chain_sim:
          union(ia, ib)

  clusters: dict[int, list[dict]] = {}
  for i, p in enumerate(stage):
    clusters.setdefault(find(i), []).append(p)

  winners: dict[str, dict] = {}
  for members in clusters.values():
    w = max(members, key=rank_fn)
    winners[str(w.get("id"))] = w
    stats["dropped_near_or_chain"] += len(members) - 1

  ordered: list[dict] = []
  seen: set[str] = set()
  for p in pairs:
    pid = str(p.get("id") or "")
    w = winners.get(pid)
    if w is None:
      # 代表が別 id のクラスタに吸収された
      continue
    wid = str(w.get("id") or "")
    if wid in seen:
      continue
    # 入力順ではじめてクラスタに触れたとき代表を出す
    ordered.append(w)
    seen.add(wid)
  stats["after"] = len(ordered)
  stats["clusters"] = len(clusters)
  return ordered, stats


def load_section_raw(path: Path) -> list[dict]:
  """examples.section.raw.jsonl → draft/revised。"""
  if not path.is_file():
    return []
  out: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      draft = strip_reference_block(str(rec.get("source_text") or ""))
      revised = strip_reference_block(str(rec.get("edited_text") or ""))
      pid = str(rec.get("project_id") or "")
      rid = str(rec.get("id") or "")
      if not draft or not revised or not pid or not rid:
        continue
      meta = rec.get("meta") or {}
      section_key = str(meta.get("section_key") or "")
      # 本文類似だけで結んだ節対応のうち、見出し末葉が食い違うものは除外
      if "<=>" in section_key:
        left, right = section_key.split("<=>", 1)
        leaf_l = left.split(">")[-1].strip()
        leaf_r = right.split(">")[-1].strip()
        if difflib.SequenceMatcher(None, leaf_l, leaf_r).ratio() < 0.45:
          continue
        lr = len(revised) / max(len(draft), 1)
        if lr < 0.4 or lr > 2.5:
          continue
      out.append(
        {
          "id": f"section:{rid}",
          "project_id": pid,
          "draft": draft,
          "revised": revised,
          "source": "section",
          "section_key": section_key,
          "path": meta.get("path") or "",
        }
      )
  return out


def load_section_dpo(path: Path) -> list[dict]:
  """dpo_section_curated.jsonl 互換（任意）。"""
  if not path.is_file():
    return []
  out: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      draft = strip_reference_block(str(rec.get("rejected") or ""))
      revised = strip_reference_block(str(rec.get("chosen") or ""))
      meta = rec.get("meta") or {}
      pid = str(meta.get("project_id") or "")
      rid = str(rec.get("id") or "")
      if not draft or not revised or not pid:
        continue
      out.append(
        {
          "id": f"section-dpo:{rid}",
          "project_id": pid,
          "draft": draft,
          "revised": revised,
          "source": "section",
        }
      )
  return out


def to_chat_row(pair: dict) -> dict:
  meta = {
    "id": pair["id"],
    "project_id": pair["project_id"],
    "source": pair.get("source", "section"),
    "char_similarity": pair.get("char_similarity"),
    "bigram_jaccard": pair.get("bigram_jaccard"),
    "length_ratio": pair.get("length_ratio"),
    "draft_chars": len(pair["draft"]),
    "revised_chars": len(pair["revised"]),
    "draft_paragraphs": paragraph_count(pair["draft"]),
    "revised_paragraphs": paragraph_count(pair["revised"]),
  }
  if pair.get("section_key"):
    meta["section_key"] = pair["section_key"]
  if pair.get("path"):
    meta["path"] = pair["path"]
  return {
    "messages": [
      {"role": "user", "content": f"{INSTRUCTION}\n\n{pair['draft']}"},
      {"role": "assistant", "content": pair["revised"]},
    ],
    "meta": meta,
  }


def pair_stats(pairs: list[dict]) -> dict:
  if not pairs:
    return {
      "n": 0,
      "length_ratio_revised_over_draft": {},
      "char_similarity_ratio": {},
      "bigram_jaccard": {},
      "draft_chars": {},
      "draft_paragraphs": {},
      "projects": {},
      "sources": {},
    }

  def pct(a: list[float]) -> dict:
    arr = np.asarray(a, dtype=float)
    return {
      "mean": float(arr.mean()),
      "p10": float(np.percentile(arr, 10)),
      "p50": float(np.percentile(arr, 50)),
      "p90": float(np.percentile(arr, 90)),
    }

  return {
    "n": len(pairs),
    "length_ratio_revised_over_draft": pct([p["length_ratio"] for p in pairs]),
    "char_similarity_ratio": pct([p["char_similarity"] for p in pairs]),
    "bigram_jaccard": pct([p["bigram_jaccard"] for p in pairs]),
    "draft_chars": pct([float(len(p["draft"])) for p in pairs]),
    "draft_paragraphs": pct([float(paragraph_count(p["draft"])) for p in pairs]),
    "projects": dict(Counter(p["project_id"] for p in pairs)),
    "sources": dict(Counter(p.get("source", "section") for p in pairs)),
  }


def filter_pairs(
  pairs: list[dict],
  *,
  min_sim: float,
  max_sim: float,
  min_jaccard: float,
  min_len_ratio: float,
  max_len_ratio: float,
  min_chars: int,
  max_chars: int,
  min_paragraphs: int,
) -> tuple[list[dict], Counter, list[dict]]:
  kept: list[dict] = []
  reject_counts: Counter = Counter()
  reject_examples: list[dict] = []
  for p in pairs:
    ok, reason, metrics = accept_edit_pair(
      p["draft"],
      p["revised"],
      min_sim=min_sim,
      max_sim=max_sim,
      min_jaccard=min_jaccard,
      min_len_ratio=min_len_ratio,
      max_len_ratio=max_len_ratio,
      min_chars=min_chars,
      max_chars=max_chars,
      min_paragraphs=min_paragraphs,
    )
    if not ok:
      reject_counts[reason] += 1
      if len(reject_examples) < 40:
        reject_examples.append(
          {
            "id": p.get("id"),
            "project_id": p.get("project_id"),
            "reason": reason,
            "metrics": metrics,
            "draft_chars": len(p["draft"]),
            "revised_chars": len(p["revised"]),
            "draft_paragraphs": paragraph_count(p["draft"]),
          }
        )
      continue
    kept.append({**p, "source": p.get("source", "section"), **metrics})
  return kept, reject_counts, reject_examples


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--section-raw",
    default="data/examples.section.raw.jsonl",
    help="節単位 raw（mine_section_pairs の出力）",
  )
  parser.add_argument(
    "--section-dpo",
    default="",
    help="任意: dpo_section_curated.jsonl を追加合流（通常は不要）",
  )
  parser.add_argument(
    "--include-hunk",
    action="store_true",
    help="健全な hunk revision_pairs も合流する",
  )
  parser.add_argument("--pairs", default="data/revision_pairs.jsonl", help="--include-hunk 時のみ")
  parser.add_argument("--out-dir", default="data/edit_sft")
  parser.add_argument(
    "--holdout-projects",
    default=",".join(DEFAULT_HOLDOUT),
    help="held-out にする project_id（カンマ区切り）",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
  parser.add_argument("--max-sim", type=float, default=DEFAULT_MAX_SIM)
  parser.add_argument("--min-jaccard", type=float, default=DEFAULT_MIN_JACCARD)
  parser.add_argument("--min-len-ratio", type=float, default=DEFAULT_MIN_LEN_RATIO)
  parser.add_argument("--max-len-ratio", type=float, default=DEFAULT_MAX_LEN_RATIO)
  parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
  parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
  parser.add_argument("--min-paragraphs", type=int, default=DEFAULT_MIN_PARAGRAPHS)
  parser.add_argument("--no-filter", action="store_true")
  parser.add_argument(
    "--no-align-nonprose",
    action="store_true",
    help="aside 等の非段落揃えをしない",
  )
  parser.add_argument(
    "--no-mask-code-figures",
    action="store_true",
    help="コード・図をプレースホルダに置換しない",
  )
  args = parser.parse_args()

  pairs = load_section_raw(Path(args.section_raw))
  if args.section_dpo:
    for sp in load_section_dpo(Path(args.section_dpo)):
      pairs.append(sp)

  if args.include_hunk:
    from steering_utils import load_revision_pairs

    for p in load_revision_pairs(Path(args.pairs)):
      p.setdefault("source", "hunk")
      pairs.append(p)

  # id 重複除去（先勝ち）
  deduped: list[dict] = []
  seen_ids: set[str] = set()
  for p in pairs:
    if p["id"] in seen_ids:
      continue
    seen_ids.add(p["id"])
    deduped.append(p)
  pairs = deduped

  align_stats: Counter = Counter()
  if not args.no_align_nonprose:
    aligned: list[dict] = []
    for p in pairs:
      new_d, new_r, st = align_nonprose_to_draft(p["draft"], p["revised"])
      for k, v in st.items():
        align_stats[k] += int(v)
      # revised の地の文が消えていないか（文字が大きく減っていたら異常）
      if len(new_r.strip()) < len(p["revised"].strip()) * 0.7:
        align_stats["align_reverted_shrink"] += 1
        aligned.append(p)
        continue
      if new_d != p["draft"] or new_r != p["revised"]:
        align_stats["pairs_changed"] += 1
      aligned.append({**p, "draft": new_d, "revised": new_r})
    pairs = aligned
  align_stats["enabled"] = 0 if args.no_align_nonprose else 1

  mask_stats: Counter = Counter()
  if not args.no_mask_code_figures:
    masked: list[dict] = []
    for p in pairs:
      new_d, new_r, st = mask_pair(p["draft"], p["revised"])
      for k, v in st.items():
        if isinstance(v, bool):
          mask_stats[k] += int(v)
        else:
          mask_stats[k] += int(v) if isinstance(v, int) else 0
      if st["changed"]:
        mask_stats["pairs_changed"] += 1
      masked.append({**p, "draft": new_d, "revised": new_r})
    pairs = masked
  mask_stats["enabled"] = 0 if args.no_mask_code_figures else 1

  if not pairs:
    raise SystemExit(
      f"no section pairs: run make mine-sections first ({args.section_raw})"
    )

  n_raw = len(pairs)
  filter_cfg = {
    "min_sim": args.min_sim,
    "max_sim": args.max_sim,
    "min_jaccard": args.min_jaccard,
    "min_len_ratio": args.min_len_ratio,
    "max_len_ratio": args.max_len_ratio,
    "min_chars": args.min_chars,
    "max_chars": args.max_chars,
    "min_paragraphs": args.min_paragraphs,
    "enabled": not args.no_filter,
    "primary_source": "section",
    "include_hunk": bool(args.include_hunk),
  }

  if args.no_filter:
    kept = []
    for p in pairs:
      m = pair_metrics(p["draft"], p["revised"])
      kept.append({**p, "source": p.get("source", "section"), **m})
    reject_counts: Counter = Counter()
    reject_examples: list[dict] = []
  else:
    kept, reject_counts, reject_examples = filter_pairs(
      pairs,
      min_sim=args.min_sim,
      max_sim=args.max_sim,
      min_jaccard=args.min_jaccard,
      min_len_ratio=args.min_len_ratio,
      max_len_ratio=args.max_len_ratio,
      min_chars=args.min_chars,
      max_chars=args.max_chars,
      min_paragraphs=args.min_paragraphs,
    )

  if not kept:
    raise SystemExit("no pairs left after filter")

  n_after_filter = len(kept)
  kept, dedupe_stats = dedupe_sft_pairs(kept)
  if dedupe_stats["dropped_exact"] or dedupe_stats["dropped_same_draft"]:
    print(
      f"dedupe: {dedupe_stats['before']} -> {dedupe_stats['after']} "
      f"(exact -{dedupe_stats['dropped_exact']}, "
      f"same_draft -{dedupe_stats['dropped_same_draft']})",
      flush=True,
    )

  if not kept:
    raise SystemExit("no pairs left after dedupe")

  holdout_projects = {p.strip() for p in args.holdout_projects.split(",") if p.strip()}
  known = {p["project_id"] for p in kept}
  missing = holdout_projects - known
  if missing:
    # heldout 指定が欠ける場合は、存在する分だけ heldout にし警告
    print(
      f"WARNING: holdout projects missing after filter: {sorted(missing)}; "
      f"using intersection with known={sorted(known)}",
      flush=True,
    )
    holdout_projects = holdout_projects & known
  if not holdout_projects:
    raise SystemExit(f"no holdout projects left (known={sorted(known)})")

  train = [p for p in kept if p["project_id"] not in holdout_projects]
  heldout = [p for p in kept if p["project_id"] in holdout_projects]
  if not train or not heldout:
    raise SystemExit(f"empty split: train={len(train)} heldout={len(heldout)}")
  random.Random(args.seed).shuffle(train)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  for name, subset in (("train", train), ("heldout", heldout)):
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
      for p in subset:
        f.write(json.dumps(to_chat_row(p), ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(subset)} rows)")

  stats = {
    "instruction": INSTRUCTION,
    "holdout_projects": sorted(holdout_projects),
    "seed": args.seed,
    "filter": filter_cfg,
    "align_nonprose": dict(align_stats),
    "mask_code_figures": dict(mask_stats),
    "dedupe": dict(dedupe_stats),
    "n_raw": n_raw,
    "n_after_filter": n_after_filter,
    "n_kept": len(kept),
    "train": pair_stats(train),
    "heldout": pair_stats(heldout),
  }
  stats_path = out_dir / "stats.json"
  stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {stats_path}")

  report = {
    "note": "本文なし。節主軸の編集 SFT 用フィルタ点検結果。",
    "filter": filter_cfg,
    "align_nonprose": dict(align_stats),
    "mask_code_figures": dict(mask_stats),
    "dedupe": dict(dedupe_stats),
    "n_raw": n_raw,
    "n_after_filter": n_after_filter,
    "n_kept": len(kept),
    "n_rejected": n_raw - n_after_filter,
    "reject_counts": dict(reject_counts),
    "reject_examples_no_text": reject_examples,
    "train_n": len(train),
    "heldout_n": len(heldout),
    "train_draft_chars_p50": stats["train"]["draft_chars"].get("p50"),
    "train_draft_paragraphs_p50": stats["train"]["draft_paragraphs"].get("p50"),
    "train_sim_p10": stats["train"]["char_similarity_ratio"].get("p10"),
    "train_sim_p50": stats["train"]["char_similarity_ratio"].get("p50"),
  }
  report_path = out_dir / "filter_report.json"
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {report_path}")
  print(
    f"raw={n_raw} kept={len(kept)} "
    f"train={len(train)} heldout={len(heldout)} "
    f"({len(heldout) / (len(train) + len(heldout)):.1%} held out)"
  )
  if reject_counts:
    print("reject_counts:", dict(reject_counts))
  tr = stats["train"]
  print(
    f"train draft_chars p50={tr['draft_chars'].get('p50'):.0f} "
    f"paragraphs p50={tr['draft_paragraphs'].get('p50'):.1f} "
    f"sim p50={tr['char_similarity_ratio'].get('p50'):.3f}"
  )


if __name__ == "__main__":
  main()
