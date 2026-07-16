#!/usr/bin/env python3
"""BT 報酬モデルで候補の絶対スコア s(source, candidate) を出す。"""
from __future__ import annotations

import argparse
from pathlib import Path

from pref_bt_runtime import load_bt_model, score_candidates_bt


def read_text(value: str | None, file_path: str | None) -> str:
  if value:
    return value
  if file_path:
    return Path(file_path).read_text(encoding="utf-8")
  raise SystemExit("text or file input is required")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, help="pref-bt model directory")
  parser.add_argument("--source-text")
  parser.add_argument("--source-file")
  parser.add_argument("--candidate-text")
  parser.add_argument("--candidate-file")
  args = parser.parse_args()

  source = read_text(args.source_text, args.source_file)
  candidate = read_text(args.candidate_text, args.candidate_file)
  loaded = load_bt_model(Path(args.model))
  score = score_candidates_bt(loaded, source, [candidate], batch_size=2)[0]
  print(f"score: {score:.6f}")


if __name__ == "__main__":
  main()
