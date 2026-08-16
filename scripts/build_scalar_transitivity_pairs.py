#!/usr/bin/env python3
"""スカラー仮説の検査用に、下書きと選ばれなかった生成案の総当たり対を作る。

既定は評価用データ C から 10 件。各件の候補は下書き 1 本と、SFT アダプタの
8 本のうち当時の評価器が最高点を付けた 1 本を除いた 2 本。総当たりは 3 対、
全体で 30 対。人間の推敲本文は出さない。評価器が選んだ文は出さない。
"""
from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from build_blind_pairs import index_samples, load_jsonl, pick_scored_best
from generation_integrity import structure_preserved

COMPARE_TYPE = "scalar_transitivity"
SRC_DRAFT = "draft"
QUESTION = "A と B のどちらが、自分の推敲に近いか"


def _sample_text(row: dict) -> str:
  return str(row.get("generated") or "")


def unelected_rows(
  draft: str,
  samples: list[dict],
  selected_text: str | None,
) -> list[dict]:
  out: list[dict] = []
  for row in samples:
    text = _sample_text(row)
    if not text:
      continue
    if selected_text is not None and text == selected_text:
      continue
    if not structure_preserved(draft, text):
      continue
    out.append(row)
  return out


def pick_item_ids(
  eligible: list[str],
  *,
  n_items: int,
  seed: int,
) -> list[str]:
  ordered = sorted(eligible)
  rng = random.Random(seed)
  if len(ordered) <= n_items:
    rng.shuffle(ordered)
    return ordered
  return sorted(rng.sample(ordered, n_items))


def _selected_text(
  draft: str,
  samples: list[dict],
  *,
  primary_model: Path | None,
  scorer,
) -> str | None:
  if scorer is None:
    return pick_scored_best(draft, samples, primary_model=primary_model, fallback=None)
  texts = [str(s.get("generated") or "") for s in samples if s.get("generated")]
  texts = [t for t in texts if structure_preserved(draft, t)]
  if not texts:
    return None
  scores = scorer.score(draft, texts)
  best_i = max(range(len(texts)), key=lambda i: scores[i])
  return texts[best_i]


def remaining_for_item(
  item: dict,
  samples: list[dict],
  *,
  primary_model: Path | None,
  scorer=None,
) -> tuple[str, list[dict]] | None:
  draft = str(item.get("draft") or "")
  if not draft:
    return None
  selected = _selected_text(
    draft,
    samples,
    primary_model=primary_model,
    scorer=scorer,
  )
  remaining = unelected_rows(draft, samples, selected)
  return draft, remaining


def _candidates_from_remaining(
  draft: str,
  remaining: list[dict],
  *,
  n_unselected: int,
  rng: random.Random,
) -> list[tuple[str, str]] | None:
  if len(remaining) < n_unselected:
    return None
  chosen = rng.sample(remaining, n_unselected)
  chosen.sort(key=lambda r: int(r.get("sample_index") or 0))
  cands = [(SRC_DRAFT, draft)]
  for row in chosen:
    idx = int(row.get("sample_index") or 0)
    cands.append((f"adapter_unselected:{idx}", _sample_text(row)))
  return cands


def build_item_candidates(
  item: dict,
  samples: list[dict],
  *,
  n_unselected: int,
  rng: random.Random,
  primary_model: Path | None,
  scorer=None,
) -> list[tuple[str, str]] | None:
  got = remaining_for_item(
    item,
    samples,
    primary_model=primary_model,
    scorer=scorer,
  )
  if got is None:
    return None
  draft, remaining = got
  return _candidates_from_remaining(
    draft,
    remaining,
    n_unselected=n_unselected,
    rng=rng,
  )


def _load_scorer(primary_model: Path | None):
  if primary_model is None or not primary_model.is_dir():
    return None
  from pref_scorer import load_scorer

  return load_scorer(primary_model)


def build_pairs(
  items: list[dict],
  samples_by_id: dict[str, list[dict]],
  *,
  n_items: int,
  n_unselected: int,
  seed: int,
  primary_model: Path | None,
) -> tuple[list[dict], dict]:
  scorer = _load_scorer(primary_model)
  remaining_by_id: dict[str, tuple[str, list[dict]]] = {}
  for item in items:
    iid = str(item.get("id") or "")
    if not iid:
      continue
    got = remaining_for_item(
      item,
      samples_by_id.get(iid) or [],
      primary_model=primary_model,
      scorer=scorer,
    )
    if got is None:
      continue
    draft, remaining = got
    if len(remaining) < n_unselected:
      continue
    remaining_by_id[iid] = (draft, remaining)

  eligible = list(remaining_by_id)
  chosen_ids = pick_item_ids(eligible, n_items=n_items, seed=seed)
  pair_rng = random.Random(seed + 1)
  item_rng = random.Random(seed + 2)
  shuffle_rng = random.Random(seed + 3)
  pairs: list[dict] = []
  skipped = {
    "n_source_items": len(items),
    "n_eligible": len(eligible),
    "n_selected_items": len(chosen_ids),
  }

  for iid in chosen_ids:
    draft, remaining = remaining_by_id[iid]
    cands = _candidates_from_remaining(
      draft,
      remaining,
      n_unselected=n_unselected,
      rng=item_rng,
    )
    if not cands:
      continue
    for (src_a, text_a), (src_b, text_b) in combinations(cands, 2):
      swap = pair_rng.random() < 0.5
      left_src, right_src = src_a, src_b
      left_text, right_text = text_a, text_b
      if swap:
        left_src, right_src = src_b, src_a
        left_text, right_text = text_b, text_a
      pairs.append(
        {
          "pair_id": f"{iid}::{COMPARE_TYPE}::{src_a}::{src_b}",
          "item_id": iid,
          "compare_type": COMPARE_TYPE,
          "context_draft": draft,
          "a_text": left_text,
          "b_text": right_text,
          "a_source": left_src,
          "b_source": right_src,
          "swapped": swap,
          "question": QUESTION,
        }
      )

  shuffle_rng.shuffle(pairs)
  for i, row in enumerate(pairs):
    row["order"] = i
  skipped["n_pairs"] = len(pairs)
  skipped["item_ids"] = chosen_ids
  return pairs, skipped


def write_protocol(path: Path, *, n_items: int, n_unselected: int, seed: int, stats: dict) -> None:
  protocol = {
    "n_items": n_items,
    "n_unselected": n_unselected,
    "n_candidates_per_item": 1 + n_unselected,
    "n_pairs_per_item": (1 + n_unselected) * n_unselected // 2,
    "seed": seed,
    "source_items": "data/blind_eval/items.jsonl",
    "source_samples": "outputs/edit-sft-eval-v3/adapter_samples.jsonl",
    "compare": "draft and two adapter samples the evaluator did not select",
    "judgment": {
      "view": "左右の役割を見せない。下書きは文脈として出す。人間の推敲本文は出さない。",
      "question": QUESTION,
      "options": ["a", "b", "tie", "incomparable"],
    },
    "aggregation": {
      "primary": "件ごとに選択が推移的か。循環した件数。比較できない対がある件数",
      "denominator": "3 対すべてを人が付けた件",
      "incomparable": "同程度ではない。比較できない対がある件は推移的とも循環とも数えない",
    },
    "pairs_file": "data/blind_eval/pairs_scalar_transitivity.jsonl",
    "judgments_file": "data/blind_eval/judgments_scalar_transitivity.jsonl",
    "stats": {
      "n_source_items": stats.get("n_source_items"),
      "n_eligible": stats.get("n_eligible"),
      "n_selected_items": stats.get("n_selected_items"),
      "n_pairs": stats.get("n_pairs"),
    },
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--items", default="data/blind_eval/items.jsonl")
  parser.add_argument(
    "--adapter-samples",
    default="outputs/edit-sft-eval-v3/adapter_samples.jsonl",
  )
  parser.add_argument(
    "--primary-model",
    default="outputs/pref-sentseq-section-triples",
  )
  parser.add_argument(
    "--out",
    default="data/blind_eval/pairs_scalar_transitivity.jsonl",
  )
  parser.add_argument(
    "--protocol-out",
    default="data/blind_eval/scalar_transitivity_protocol.json",
  )
  parser.add_argument("--n-items", type=int, default=10)
  parser.add_argument("--n-unselected", type=int, default=2)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  items = load_jsonl(Path(args.items))
  if not items:
    raise SystemExit(f"no items in {args.items}")
  samples_by_id = index_samples(load_jsonl(Path(args.adapter_samples)))
  primary = Path(args.primary_model)
  primary_model = primary if primary.is_dir() else None
  pairs, stats = build_pairs(
    items,
    samples_by_id,
    n_items=args.n_items,
    n_unselected=args.n_unselected,
    seed=args.seed,
    primary_model=primary_model,
  )
  if not pairs:
    raise SystemExit("no pairs: not enough unelected samples")
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for row in pairs:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  write_protocol(
    Path(args.protocol_out),
    n_items=args.n_items,
    n_unselected=args.n_unselected,
    seed=args.seed,
    stats=stats,
  )
  n_swap = sum(1 for row in pairs if row["swapped"])
  print(
    f"wrote {out} items={stats['n_selected_items']} pairs={len(pairs)} "
    f"swapped={n_swap} eligible={stats['n_eligible']} seed={args.seed}",
    flush=True,
  )


if __name__ == "__main__":
  main()
