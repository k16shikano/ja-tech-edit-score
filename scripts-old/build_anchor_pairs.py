#!/usr/bin/env python3
"""基準=候補（コピー）を含むアンカー付き選好ペアを既存ペアから作る。

運用時のマージン（margin = s(基準, 案) - s(基準, 基準)）は、基準と同一の
候補という学習にない入力を含むため、「変更しただけで加点」の偏りがある。
これを直すため、実ペア（下書き ≺ 人間編集）から次の 2 種類を機械的に作る。

- anchor-fwd: 基準=下書き、人間編集 ≻ 下書きのコピー
  （下書きに対して、実編集は「何もしない」に勝つ）
- anchor-rev: 基準=人間編集、人間編集のコピー ≻ 下書き
  （すでに良い版なら、「何もしない」が下書きへ戻すのに勝つ）

捏造した劣化例ではなく、実在ペアの役割の組み替えだけを使う。
入力には分割済みの train.jsonl を渡し、valid はそのまま使うことで
評価のリークを避ける。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def anchor_rows(row: dict) -> list[dict]:
  meta = row.get("meta", {})
  if meta.get("pair_order", "chosen_first") != "chosen_first":
    return []
  if int(row["label"]) != 1:
    return []
  edit = row["candidate_a"]
  draft = row["candidate_b"]
  if edit == draft:
    return []
  base_meta = {**meta, "base_id": meta.get("base_id", row["id"])}
  return [
    {
      "id": row["id"] + "-anchor-fwd",
      "source_text": draft,
      "candidate_a": edit,
      "candidate_b": draft,
      "label": 1,
      "meta": {**base_meta, "pair_order": "chosen_first", "anchor": "draft_base"},
    },
    {
      "id": row["id"] + "-anchor-rev",
      "source_text": edit,
      "candidate_a": edit,
      "candidate_b": draft,
      "label": 1,
      "meta": {**base_meta, "pair_order": "chosen_first", "anchor": "edit_base"},
    },
  ]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", default="data/pref_split/train.jsonl")
  parser.add_argument("--valid", default="data/pref_split/valid.jsonl")
  parser.add_argument("--out-dir", default="data/pref_split_anchor")
  parser.add_argument("--train-name", default="train.jsonl", help="出力する train ファイル名")
  parser.add_argument(
    "--anchor-fraction",
    type=float,
    default=1.0,
    help="アンカーを付ける実ペアの割合（0-1）。同じペアの fwd/rev は常にセット",
  )
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  rng = random.Random(args.seed)

  n_orig = n_anchor = 0
  with (out_dir / args.train_name).open("w", encoding="utf-8") as dst:
    for line in Path(args.train).open(encoding="utf-8"):
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      dst.write(json.dumps(row, ensure_ascii=False) + "\n")
      n_orig += 1
      extras = anchor_rows(row)
      if extras and rng.random() >= args.anchor_fraction:
        continue
      for extra in extras:
        dst.write(json.dumps(extra, ensure_ascii=False) + "\n")
        n_anchor += 1

  valid_text = Path(args.valid).read_text(encoding="utf-8")
  (out_dir / "valid.jsonl").write_text(valid_text, encoding="utf-8")

  print(f"train: original={n_orig} anchor={n_anchor} -> {out_dir / args.train_name}")
  print(f"valid: copied unchanged -> {out_dir / 'valid.jsonl'}")


if __name__ == "__main__":
  main()
