#!/usr/bin/env python3
"""corpus（source_text/edited_text）をレビュー UI 用 chat JSONL に書き出す。

レビューキューはディレクトリを分けて持つ（例: hunk 用、節追加用）。
学習への合流は export_reviewed_keeps 側。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from export_edit_sft import INSTRUCTION, pair_metrics
from mask_code_figures import mask_pair
from align_nonprose_to_draft import align_nonprose_to_draft

DEFAULT_HOLDOUT = ("what-is-monad", "computer-arch-revisit", "ir-system")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="canonical / hunk_trainready JSONL")
  parser.add_argument("--out-dir", required=True)
  parser.add_argument(
    "--holdout-projects",
    default=",".join(DEFAULT_HOLDOUT),
  )
  parser.add_argument(
    "--quality",
    default="",
    help="空なら全件。keep / candidate など",
  )
  parser.add_argument("--mask", action="store_true", help="未マスクなら mask_pair を適用")
  parser.add_argument("--align", action="store_true", help="align_nonprose を適用")
  parser.add_argument("--source-label", default="corpus")
  parser.add_argument(
    "--reset-state",
    action="store_true",
    help="review_state.json を空にする（旧 keep の carry 禁止）",
  )
  args = parser.parse_args()

  holdout = {p.strip() for p in args.holdout_projects.split(",") if p.strip()}
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  train_rows: list[dict] = []
  held_rows: list[dict] = []
  skip = Counter()

  with Path(args.input).open(encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      obj = json.loads(line)
      if args.quality and obj.get("quality") != args.quality:
        skip["quality"] += 1
        continue
      draft = str(obj.get("source_text") or "").strip()
      revised = str(obj.get("edited_text") or "").strip()
      if not draft or not revised or draft == revised:
        skip["empty_or_identical"] += 1
        continue
      if args.align:
        nd, nr, _ = align_nonprose_to_draft(draft, revised)
        if len(nr.strip()) >= len(revised) * 0.7:
          draft, revised = nd, nr
      if args.mask:
        draft, revised, _ = mask_pair(draft, revised)
        if draft.strip() == revised.strip():
          skip["identical_after_mask"] += 1
          continue
      metrics = pair_metrics(draft, revised)
      pid = str(obj.get("project_id") or "unknown")
      rid = str(obj.get("id") or "")
      chat = {
        "messages": [
          {"role": "user", "content": f"{INSTRUCTION}\n\n{draft}"},
          {"role": "assistant", "content": revised},
        ],
        "meta": {
          "id": rid,
          "project_id": pid,
          "source": args.source_label,
          "char_similarity": metrics["char_similarity"],
          "bigram_jaccard": metrics["bigram_jaccard"],
          "length_ratio": metrics["length_ratio"],
          "draft_chars": len(draft),
          "revised_chars": len(revised),
          "unit": obj.get("unit"),
          "compare_kind": obj.get("compare_kind"),
          "corpus_source": obj.get("corpus_source"),
          "review_status": "pending",
        },
      }
      if pid in holdout:
        held_rows.append(chat)
      else:
        train_rows.append(chat)

  train_path = out_dir / "train.jsonl"
  held_path = out_dir / "heldout.jsonl"
  with train_path.open("w", encoding="utf-8") as f:
    for row in train_rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  with held_path.open("w", encoding="utf-8") as f:
    for row in held_rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  # 空の state（carry しない。--reset-state または未作成時のみ）
  state_path = out_dir / "review_state.json"
  if args.reset_state or not state_path.exists():
    state_path.write_text("{}\n", encoding="utf-8")

  stats = {
    "train": len(train_rows),
    "heldout": len(held_rows),
    "skip": dict(skip),
    "holdout_projects": sorted(holdout),
    "out_dir": str(out_dir),
  }
  (out_dir / "stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
