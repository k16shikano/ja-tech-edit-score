#!/usr/bin/env python3
"""ブラインド判定用の比較ペア一覧を作る。

生成結果が揃っていれば 6 種すべてを埋める。
未揃いの比較はスキップし、揃っているものだけで pairs.jsonl を書く
（アプリ開発・途中再開用）。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def index_by_id(rows: list[dict], *, sample_index: int | None = None) -> dict[str, dict]:
  out: dict[str, dict] = {}
  for r in rows:
    if sample_index is not None and int(r.get("sample_index") or 0) != sample_index:
      continue
    rid = str(r.get("id") or "")
    if rid:
      out[rid] = r
  return out


def index_samples(rows: list[dict]) -> dict[str, list[dict]]:
  out: dict[str, list[dict]] = {}
  for r in rows:
    rid = str(r.get("id") or "")
    if not rid:
      continue
    out.setdefault(rid, []).append(r)
  for rid in out:
    out[rid].sort(key=lambda x: int(x.get("sample_index") or 0))
  return out


def pick_scored_best(
  draft: str,
  samples: list[dict],
  *,
  primary_model: Path | None,
  fallback: str | None = None,
) -> str | None:
  """構成要素を保ったサンプルだけを評価器で順位付けし、最高点を返す。

  健全なサンプルが 1 本もなければ fallback（貪欲生成）を返す。
  評価器は壊れた候補を弾けないことが分かっているので、
  整合性の検査は機械（generation_integrity.structure_preserved）で先に行う。
  """
  import sys

  root = Path(__file__).resolve().parent
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  from generation_integrity import structure_preserved

  texts = [str(s.get("generated") or "") for s in samples if s.get("generated")]
  texts = [t for t in texts if structure_preserved(draft, t)]
  if not texts:
    return fallback
  if primary_model is None or not primary_model.is_dir():
    return texts[0]
  try:
    from pref_scorer import load_scorer

    scorer = load_scorer(primary_model)
    scores = scorer.score(draft, texts)
    best_i = max(range(len(texts)), key=lambda i: scores[i])
    return texts[best_i]
  except Exception as exc:  # noqa: BLE001
    print(f"WARNING: scorer unavailable ({exc}); using sample 0", flush=True)
    return texts[0]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--items", default="data/blind_eval/items.jsonl")
  parser.add_argument("--out", default="data/blind_eval/pairs.jsonl")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--base-greedy", default="outputs/edit-sft-eval-v3/base_greedy.jsonl")
  parser.add_argument(
    "--adapter-greedy",
    default="outputs/edit-sft-eval-v3/adapter_greedy.jsonl",
  )
  parser.add_argument(
    "--adapter-samples",
    default="outputs/edit-sft-eval-v3/adapter_samples.jsonl",
  )
  parser.add_argument(
    "--primary-model",
    default="outputs/pref-sentseq-section-triples",
    help="サンプル選抜に使う評価器。無ければ先頭サンプル",
  )
  args = parser.parse_args()

  items = load_jsonl(Path(args.items))
  if not items:
    raise SystemExit(f"no items in {args.items}")

  base_g = index_by_id(load_jsonl(Path(args.base_greedy)))
  ad_g = index_by_id(load_jsonl(Path(args.adapter_greedy)))
  ad_s = index_samples(load_jsonl(Path(args.adapter_samples)))
  primary = Path(args.primary_model)

  rng = random.Random(args.seed)
  pairs: list[dict] = []
  skipped = 0

  for item in items:
    iid = str(item["id"])
    draft = str(item["draft"])
    gold = str(item["gold"])
    base_text = (base_g.get(iid) or {}).get("generated")
    ad_text = (ad_g.get(iid) or {}).get("generated")
    samples = ad_s.get(iid) or []
    selected = (
      pick_scored_best(draft, samples, primary_model=primary, fallback=ad_text)
      if samples
      else None
    )

    candidates = [
      ("1_draft_vs_base_greedy", draft, base_text, "draft", "base_greedy"),
      ("2_draft_vs_adapter_greedy", draft, ad_text, "draft", "adapter_greedy"),
      ("3_base_vs_adapter_greedy", base_text, ad_text, "base_greedy", "adapter_greedy"),
      ("4_draft_vs_adapter_selected", draft, selected, "draft", "adapter_selected"),
      ("5_gold_vs_adapter_selected", gold, selected, "gold", "adapter_selected"),
      ("6_gold_vs_draft", gold, draft, "gold", "draft"),
    ]
    for ctype, left, right, left_src, right_src in candidates:
      if not left or not right:
        skipped += 1
        continue
      swap = rng.random() < 0.5
      a_text, b_text = (right, left) if swap else (left, right)
      a_src, b_src = (right_src, left_src) if swap else (left_src, right_src)
      pairs.append(
        {
          "pair_id": f"{iid}::{ctype}",
          "item_id": iid,
          "compare_type": ctype,
          "context_draft": draft,
          "a_text": a_text,
          "b_text": b_text,
          "a_source": a_src,
          "b_source": b_src,
          "swapped": swap,
        }
      )

  rng.shuffle(pairs)
  for i, p in enumerate(pairs):
    p["order"] = i

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for p in pairs:
      f.write(json.dumps(p, ensure_ascii=False) + "\n")

  print(
    f"wrote {out} pairs={len(pairs)} items={len(items)} "
    f"skipped_incomplete={skipped} seed={args.seed}"
  )


if __name__ == "__main__":
  main()
