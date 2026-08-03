#!/usr/bin/env python3
"""hunk 系 examples.raw を学習用に整形する。

人手レビュー版（節 SFT）と同じ処理を機械適用する。
- align_nonprose_to_draft（aside 等）
- コード／図のマスク、行頭 ★ コメント行の削除（mask_pair）
- 対応破綻・英文のみ・マスク後の非散文・同一／ほぼ同一などを除去

入力:
  - 本リポの pre-merge examples.raw.jsonl
  - wwtawwta-systems の pre-merge examples.wwtawwta.premerge.raw.jsonl
    （editor-ai-starter の branch_mined 全履歴は使わない）

出力:
  - data/revision_corpus/hunk_trainready.jsonl（examples スキーマ + corpus メタ）
  - data/revision_corpus/hunk_trainready.report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from align_nonprose_to_draft import align_nonprose_to_draft
from export_edit_sft import accept_edit_pair, pair_metrics
from mask_code_figures import mask_pair

# 節 SFT に近いが、hunk 用に段落要件は外す。対応の下限は袋詰め除去のためやや厳しめ。
DEFAULT_MIN_SIM = 0.15
DEFAULT_MAX_SIM = 0.985
DEFAULT_MIN_JACCARD = 0.25
DEFAULT_MIN_LEN_RATIO = 0.4
DEFAULT_MAX_LEN_RATIO = 2.5
DEFAULT_MIN_CHARS = 40
DEFAULT_MAX_CHARS = 8000


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ja_count(text: str) -> int:
  return len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", text))


def prose_after_mask(text: str) -> str:
  t = re.sub(r"【コード】|【図】", "", text or "")
  return re.sub(r"\s+", "", t)


def load_raw(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      rows.append(json.loads(line))
  return rows


def content_id(project_id: str, draft: str, revised: str) -> str:
  h = hashlib.sha256((draft + "\n---\n" + revised).encode("utf-8")).hexdigest()[:12]
  return f"{project_id}-{h}"


def process_rows(
  rows: list[dict],
  *,
  corpus_source: str,
  unit: str,
  compare_kind: str,
  min_sim: float,
  max_sim: float,
  min_jaccard: float,
  min_len_ratio: float,
  max_len_ratio: float,
  min_chars: int,
  max_chars: int,
  do_align: bool,
) -> tuple[list[dict], Counter, Counter]:
  kept: list[dict] = []
  reject: Counter = Counter()
  stage: Counter = Counter()

  for obj in rows:
    stage["input"] += 1
    draft = str(obj.get("source_text") or "")
    revised = str(obj.get("edited_text") or "")
    if not draft.strip() or not revised.strip():
      reject["empty"] += 1
      continue

    if do_align:
      new_d, new_r, st = align_nonprose_to_draft(draft, revised)
      if len(new_r.strip()) < len(revised.strip()) * 0.7:
        stage["align_reverted_shrink"] += 1
      else:
        if new_d != draft or new_r != revised:
          stage["align_changed"] += 1
        draft, revised = new_d, new_r

    draft, revised, mst = mask_pair(draft, revised)
    if mst.get("changed"):
      stage["mask_changed"] += 1

    ok, reason, metrics = accept_edit_pair(
      draft,
      revised,
      min_sim=min_sim,
      max_sim=max_sim,
      min_jaccard=min_jaccard,
      min_len_ratio=min_len_ratio,
      max_len_ratio=max_len_ratio,
      min_chars=min_chars,
      max_chars=max_chars,
      min_paragraphs=0,
    )
    if not ok:
      reject[reason] += 1
      continue

    # マスク後に日本語の地の文がほぼ無いものは学習価値が薄い
    if ja_count(prose_after_mask(draft)) < 12 and ja_count(prose_after_mask(revised)) < 12:
      reject["non_prose_after_mask"] += 1
      continue

    pid = str(obj.get("project_id") or "unknown")
    oid = str(obj.get("id") or content_id(pid, draft, revised))
    kept.append(
      {
        "id": f"hunk:{oid}",
        "project_id": pid,
        "source_text": draft.strip(),
        "edited_text": revised.strip(),
        "source_reference": str(obj.get("source_reference") or ""),
        "rationale": str(obj.get("rationale") or "masked hunk pair"),
        "labels": list(obj.get("labels") or []) + ["hunk_trainready", "code_figure_masked"],
        "author": str(obj.get("author") or "human"),
        "review_result": "pending",
        "created_at": _now(),
        "unit": unit,
        "compare_kind": compare_kind,
        "quality": "candidate",
        "has_paragraph_break": "\n\n" in draft or "\n\n" in revised,
        "corpus_source": corpus_source,
        "char_similarity": metrics["char_similarity"],
        "bigram_jaccard": metrics["bigram_jaccard"],
        "length_ratio": metrics["length_ratio"],
      }
    )
  return kept, reject, stage


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo-raw", default="data/examples.raw.jsonl")
  parser.add_argument(
    "--ww-raw",
    default="data/examples.wwtawwta.premerge.raw.jsonl",
    help="wwtawwta-systems の pre-merge hunk raw（starter の branch_mined は使わない）",
  )
  parser.add_argument("--out", default="data/revision_corpus/hunk_trainready.jsonl")
  parser.add_argument("--report", default="data/revision_corpus/hunk_trainready.report.json")
  parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
  parser.add_argument("--max-sim", type=float, default=DEFAULT_MAX_SIM)
  parser.add_argument("--min-jaccard", type=float, default=DEFAULT_MIN_JACCARD)
  parser.add_argument("--min-len-ratio", type=float, default=DEFAULT_MIN_LEN_RATIO)
  parser.add_argument("--max-len-ratio", type=float, default=DEFAULT_MAX_LEN_RATIO)
  parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
  parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
  parser.add_argument("--no-align-nonprose", action="store_true")
  parser.add_argument("--skip-ww", action="store_true")
  parser.add_argument("--skip-repo", action="store_true")
  args = parser.parse_args()

  all_kept: list[dict] = []
  report: dict = {
    "built_at": _now(),
    "filter": {
      "min_sim": args.min_sim,
      "max_sim": args.max_sim,
      "min_jaccard": args.min_jaccard,
      "min_len_ratio": args.min_len_ratio,
      "max_len_ratio": args.max_len_ratio,
      "min_chars": args.min_chars,
      "max_chars": args.max_chars,
      "min_paragraphs": 0,
      "align_nonprose": not args.no_align_nonprose,
      "mask_code_figures": True,
    },
    "stages": {},
  }

  sources: list[tuple[str, Path, str, str]] = []
  if not args.skip_repo:
    sources.append(
      (
        "repo_premerge",
        Path(args.repo_raw),
        "examples.raw.jsonl (premerge)",
        "pre_merge",
      )
    )
  if not args.skip_ww:
    sources.append(
      (
        "wwtawwta",
        Path(args.ww_raw),
        "examples.wwtawwta.premerge.raw.jsonl",
        "pre_merge",
      )
    )

  seen_text: set[tuple[str, str]] = set()
  for key, path, corpus_source, compare_kind in sources:
    if not path.is_file():
      raise SystemExit(f"missing {path}")
    rows = load_raw(path)
    kept, reject, stage = process_rows(
      rows,
      corpus_source=corpus_source,
      unit="hunk",
      compare_kind=compare_kind,
      min_sim=args.min_sim,
      max_sim=args.max_sim,
      min_jaccard=args.min_jaccard,
      min_len_ratio=args.min_len_ratio,
      max_len_ratio=args.max_len_ratio,
      min_chars=args.min_chars,
      max_chars=args.max_chars,
      do_align=not args.no_align_nonprose,
    )
    deduped = []
    dup = 0
    for r in kept:
      t = (r["source_text"], r["edited_text"])
      if t in seen_text:
        dup += 1
        continue
      seen_text.add(t)
      deduped.append(r)
    all_kept.extend(deduped)
    report["stages"][key] = {
      "input": stage["input"],
      "kept": len(deduped),
      "dup_dropped": dup,
      "align_changed": stage.get("align_changed", 0),
      "mask_changed": stage.get("mask_changed", 0),
      "reject": dict(reject),
    }

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for row in all_kept:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  report["n_keep"] = len(all_kept)
  report["projects"] = dict(Counter(r["project_id"] for r in all_kept))
  report["has_paragraph_break"] = sum(1 for r in all_kept if r["has_paragraph_break"])
  Path(args.report).write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps({
    "n_keep": report["n_keep"],
    "has_paragraph_break": report["has_paragraph_break"],
    "projects": report["projects"],
    "stages": {k: {"input": v["input"], "kept": v["kept"], "reject_top": dict(Counter(v["reject"]).most_common(6))} for k, v in report["stages"].items()},
    "out": str(out),
  }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
