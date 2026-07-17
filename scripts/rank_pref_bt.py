#!/usr/bin/env python3
"""Best-of-N: 報酬モデル（pref-bt / pref-ce 自動判別）で複数候補をランク付けし最良を選ぶ。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pref_scorer import load_scorer


def read_text(value: str | None, file_path: str | None) -> str:
  if value:
    return value
  if file_path:
    return Path(file_path).read_text(encoding="utf-8")
  raise SystemExit("source text or file is required")


def load_candidates(args: argparse.Namespace) -> list[tuple[str, str]]:
  """Return list of (label, text)."""
  pairs: list[tuple[str, str]] = []
  for path in args.candidate_file or []:
    p = Path(path)
    pairs.append((str(p), p.read_text(encoding="utf-8")))
  for i, text in enumerate(args.candidate_text or [], start=1):
    pairs.append((f"text-{i}", text))
  if args.candidates_dir:
    directory = Path(args.candidates_dir)
    if not directory.is_dir():
      raise SystemExit(f"not a directory: {directory}")
    for path in sorted(directory.iterdir()):
      if path.is_file() and not path.name.startswith("."):
        pairs.append((str(path), path.read_text(encoding="utf-8")))
  if not pairs:
    raise SystemExit("at least one candidate is required")
  return pairs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, help="pref-bt or pref-ce model directory")
  parser.add_argument("--source-text")
  parser.add_argument("--source-file")
  parser.add_argument(
    "--candidate-file",
    action="append",
    default=[],
    help="candidate file (repeatable)",
  )
  parser.add_argument(
    "--candidate-text",
    action="append",
    default=[],
    help="candidate text (repeatable)",
  )
  parser.add_argument("--candidates-dir", help="directory of candidate files")
  parser.add_argument(
    "--include-source",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="候補に下書き自身を含め、改善なしも選べるようにする（既定: true）",
  )
  parser.add_argument(
    "--min-margin",
    type=float,
    default=None,
    help="source 自己スコアからの最小改善幅。未達なら reject",
  )
  parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
  parser.add_argument("--batch-size", type=int, default=32)
  args = parser.parse_args()

  source = read_text(args.source_text, args.source_file)
  candidates = load_candidates(args)
  labels = [label for label, _ in candidates]
  texts = [text for _, text in candidates]
  if args.include_source:
    labels = ["__source__"] + labels
    texts = [source] + texts

  scorer = load_scorer(Path(args.model))
  scores = scorer.score(source, texts, batch_size=args.batch_size)
  source_score = scores[0] if args.include_source else float(
    scorer.score(source, [source], batch_size=1)[0]
  )

  ranked = sorted(
    [
      {
        "rank": 0,
        "index": i,
        "label": labels[i],
        "score": scores[i],
        "margin_vs_source": scores[i] - source_score,
        "is_source": labels[i] == "__source__",
        "preview": texts[i].replace("\n", " ")[:120],
      }
      for i in range(len(texts))
    ],
    key=lambda row: (-row["score"], row["index"]),
  )
  for i, row in enumerate(ranked, start=1):
    row["rank"] = i

  best = ranked[0]
  accepted = True
  reject_reason = ""
  if args.min_margin is not None and best["margin_vs_source"] < args.min_margin:
    accepted = False
    reject_reason = f"margin_vs_source {best['margin_vs_source']:.4f} < min_margin {args.min_margin}"

  payload = {
    "scorer": scorer.kind,
    "accepted": accepted,
    "reject_reason": reject_reason,
    "winner_label": best["label"] if accepted else None,
    "winner_index": best["index"] if accepted else None,
    "winner_score": best["score"] if accepted else None,
    "source_score": source_score,
    "n_candidates": len(texts),
    "ranked": ranked,
  }

  if args.format == "json":
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return

  if args.format == "markdown":
    status = "accepted" if accepted else "rejected"
    print(f"**Best-of-N**: {status}")
    if accepted:
      print(f"- winner: `{best['label']}`")
      print(f"- score: `{best['score']:.4f}` (margin vs source: `{best['margin_vs_source']:.4f}`)")
    else:
      print(f"- reason: {reject_reason}")
    print("")
    print("| rank | label | score | margin |")
    print("|-----:|-------|------:|-------:|")
    for row in ranked:
      print(
        f"| {row['rank']} | `{row['label']}` | {row['score']:.4f} | "
        f"{row['margin_vs_source']:.4f} |"
      )
    return

  print(f"accepted: {str(accepted).lower()}")
  if accepted:
    print(f"winner: {best['label']}")
    print(f"score: {best['score']:.6f}")
    print(f"margin_vs_source: {best['margin_vs_source']:.6f}")
  else:
    print(f"reject_reason: {reject_reason}")
  print(f"source_score: {source_score:.6f}")
  print("ranked:")
  for row in ranked:
    mark = " *" if row["rank"] == 1 else ""
    print(
      f"  {row['rank']}. {row['label']}: {row['score']:.6f} "
      f"(margin={row['margin_vs_source']:.6f}){mark}"
    )


if __name__ == "__main__":
  main()
