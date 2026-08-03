#!/usr/bin/env python3
"""正しい推敲コーパスの正本を組み立てる。

方針: docs/EDIT-SFT-CORPUS.md

学習採用は人手 keep のみ。空行の有無で二系統に分ける
（scripts/export_reviewed_keeps.py の出力を正とする）。

  keep_section.jsonl      — 推敲前か後に空行あり（8310+8312+空行あり8311）
  keep_hunk_nopara.jsonl  — どちらにも空行なし
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  if not path.is_file():
    return rows
  with path.open(encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      rows.append(json.loads(line))
  return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", default="data/revision_corpus")
  parser.add_argument(
    "--keep-section",
    default="data/revision_corpus/keep_section.jsonl",
  )
  parser.add_argument(
    "--keep-hunk-nopara",
    default="data/revision_corpus/keep_hunk_nopara.jsonl",
  )
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  section = load_jsonl(Path(args.keep_section))
  hunk = load_jsonl(Path(args.keep_hunk_nopara))
  if not section and not hunk:
    raise SystemExit(
      "missing keep exports; run: python scripts/export_reviewed_keeps.py"
    )

  all_rows = section + hunk
  # 本文重複は section 優先（先に並べている）
  seen: set[tuple[str, str]] = set()
  deduped: list[dict] = []
  dup = 0
  for r in all_rows:
    key = (r.get("source_text") or "", r.get("edited_text") or "")
    if key in seen:
      dup += 1
      continue
    seen.add(key)
    deduped.append(r)

  write_jsonl(out_dir / "canonical.jsonl", deduped)

  quality = Counter(r.get("quality", "keep") for r in deduped)
  unit = Counter(r.get("unit", "") for r in deduped)
  bucket = Counter(
    (r.get("meta") or {}).get("bucket")
    or ("section" if r.get("has_paragraph_break") else "hunk_nopara")
    for r in deduped
  )
  keep_only = [r for r in deduped if r.get("quality") == "keep"]
  keep_para = sum(1 for r in keep_only if r.get("has_paragraph_break"))
  stats = {
    "built_at": _now(),
    "n_total": len(deduped),
    "n_keep": len(keep_only),
    "n_dup_dropped": dup,
    "quality": dict(quality),
    "unit": dict(unit),
    "bucket": dict(bucket),
    "n_section": len(section),
    "n_hunk_nopara": len(hunk),
    "has_paragraph_break_keep": keep_para,
    "paragraph_rate_keep": (keep_para / len(keep_only)) if keep_only else 0.0,
    "by_corpus_source": dict(Counter(r.get("corpus_source", "") for r in deduped)),
    "projects_section": dict(Counter(r["project_id"] for r in section)),
    "projects_hunk_nopara": dict(Counter(r["project_id"] for r in hunk)),
  }
  report = {
    "built_at": _now(),
    "stages": {
      "keep_section": {"n": len(section), "path": args.keep_section},
      "keep_hunk_nopara": {"n": len(hunk), "path": args.keep_hunk_nopara},
      "canonical_dedupe_dropped": dup,
    },
    "stats": stats,
  }
  (out_dir / "stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  (out_dir / "build_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(stats, ensure_ascii=False, indent=2))
  print(f"wrote {out_dir / 'canonical.jsonl'}")


if __name__ == "__main__":
  main()
