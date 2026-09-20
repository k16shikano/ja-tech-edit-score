#!/usr/bin/env python3
"""A1 三群スモークの生成 jsonl を読み、同じ下書きに三群が出たかと、生成文の漏れを見る。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODES = ("base", "base_norms", "adapter")
PROMPT_LEAK_PREFIXES = (
  "あなたは日本語技術文書の編集者である",
  "【文章規範】",
  "【作業】",
  "【出力規則",
  "【下書き】",
  "次の下書きを、意味を保ったまま日本語の技術文書として推敲せよ",
  "次の下書きを、上記の規範に沿って",
)


def load_rows(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def looks_like_prompt_leak(text: str) -> bool:
  t = (text or "").lstrip()
  return any(t.startswith(p) for p in PROMPT_LEAK_PREFIXES)


def leak_ids(out_dir: Path) -> list[str]:
  found: list[str] = []
  seen: set[str] = set()
  for mode in MODES:
    path = out_dir / f"{mode}_samples.jsonl"
    if not path.is_file():
      continue
    for row in load_rows(path):
      rid = str(row.get("id") or "")
      if not rid or rid in seen:
        continue
      if looks_like_prompt_leak(str(row.get("generated") or "")):
        seen.add(rid)
        found.append(rid)
  return found


def drop_ids(out_dir: Path, ids: set[str]) -> list[dict]:
  dropped: list[dict] = []
  for mode in MODES:
    path = out_dir / f"{mode}_samples.jsonl"
    if not path.is_file():
      continue
    kept: list[dict] = []
    for row in load_rows(path):
      if str(row.get("id") or "") in ids:
        dropped.append(row)
      else:
        kept.append(row)
    with path.open("w", encoding="utf-8") as f:
      for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
  return dropped


def check_dir(out_dir: Path) -> list[str]:
  errors: list[str] = []
  by_mode: dict[str, list[dict]] = {}
  for mode in MODES:
    path = out_dir / f"{mode}_samples.jsonl"
    if not path.is_file():
      errors.append(f"missing {path}")
      continue
    rows = load_rows(path)
    if not rows:
      errors.append(f"empty {path}")
      continue
    for i, row in enumerate(rows):
      if row.get("mode") != mode:
        errors.append(f"{path}:{i} mode={row.get('mode')!r} expected {mode}")
      gen = str(row.get("generated") or "").strip()
      if not gen:
        errors.append(f"{path}:{i} empty generated id={row.get('id')}")
      elif looks_like_prompt_leak(gen):
        errors.append(f"{path}:{i} generated looks like the prompt id={row.get('id')}")
    by_mode[mode] = rows

  if len(by_mode) == 3:
    ids = {mode: [r.get("id") for r in rows] for mode, rows in by_mode.items()}
    n = {mode: len(v) for mode, v in ids.items()}
    if n["base"] != n["base_norms"] or n["base"] != n["adapter"]:
      errors.append(
        f"n differs across modes: base={n['base']} "
        f"base_norms={n['base_norms']} adapter={n['adapter']}"
      )
    elif ids["base"] != ids["base_norms"] or ids["base"] != ids["adapter"]:
      errors.append("id order differs across modes")
  return errors


def dump_texts(out_dir: Path, *, n: int | None = None) -> str:
  by_id: dict[str, dict[str, dict]] = {}
  order: list[str] = []
  for mode in MODES:
    path = out_dir / f"{mode}_samples.jsonl"
    if not path.is_file():
      continue
    for row in load_rows(path):
      rid = str(row.get("id") or "")
      if rid not in by_id:
        by_id[rid] = {}
        order.append(rid)
      by_id[rid][mode] = row
  if n is not None:
    order = order[: max(0, n)]
  parts: list[str] = []
  for rid in order:
    parts.append(f"== {rid}")
    draft = ""
    for mode in MODES:
      row = by_id[rid].get(mode)
      if row and not draft:
        draft = str(row.get("draft") or "")
    parts.append("--- draft")
    parts.append(draft)
    for mode in MODES:
      row = by_id[rid].get(mode) or {}
      parts.append(f"--- {mode}")
      parts.append(str(row.get("generated") or ""))
    parts.append("")
  return "\n".join(parts).rstrip() + "\n"


def count_rows(out_dir: Path) -> dict[str, int]:
  out: dict[str, int] = {}
  for mode in MODES:
    path = out_dir / f"{mode}_samples.jsonl"
    if not path.is_file():
      out[mode] = 0
      continue
    out[mode] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="outputs/a1-probe-smoke")
  parser.add_argument(
    "--dump-n",
    type=int,
    default=-1,
    help="先頭何件の本文を出すか。-1 は全件、0 は出さない",
  )
  parser.add_argument("--expect-n", type=int, default=0, help="0 なら件数を検査しない")
  parser.add_argument(
    "--drop-leaks",
    action="store_true",
    help="指示が本文に出た id を三群から外す",
  )
  args = parser.parse_args()
  out_dir = Path(args.dir)
  if not out_dir.is_dir():
    raise SystemExit(f"missing {out_dir}")
  if args.drop_leaks:
    ids = leak_ids(out_dir)
    dropped = drop_ids(out_dir, set(ids))
    report = {
      "ids": ids,
      "n_dropped_items": len(ids),
      "n_dropped_rows": len(dropped),
    }
    report_path = out_dir / "dropped_prompt_leaks.json"
    report_path.write_text(
      json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"dropped {len(ids)} ids -> {report_path}", file=sys.stderr)
  errors = check_dir(out_dir)
  counts = count_rows(out_dir)
  if args.expect_n > 0:
    for mode, n in counts.items():
      if n != args.expect_n:
        errors.append(f"{mode} n={n} expected {args.expect_n}")
  dump_n = None if args.dump_n < 0 else args.dump_n
  if dump_n != 0:
    sys.stdout.write(dump_texts(out_dir, n=dump_n))
  print(f"n: {counts}", file=sys.stderr)
  if errors:
    print(f"CHECK FAIL: {out_dir}", file=sys.stderr)
    for e in errors:
      print(f"  {e}", file=sys.stderr)
    raise SystemExit(1)
  print(f"CHECK OK: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
  main()
