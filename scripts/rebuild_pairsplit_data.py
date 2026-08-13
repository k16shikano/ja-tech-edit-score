#!/usr/bin/env python3
"""書籍単位の分割を捨て、ペア単位の層化乱択で学習/検証を作り直す。

入力: data/revision_corpus/canonical.jsonl（推敲前後ペア正本）
出力:
  - data/pairsplit/assignment.jsonl … id → split（再現用）
  - data/pairsplit/report.json … 件数内訳
  - canonical / keep_*.jsonl の meta.split を上書き
  - data/edit_sft_{section,hunk_nopara,all}/{train,heldout}.jsonl
  - pref-keep 分割（--also-pref 時）

旧書籍単位の id 一覧は data/pairsplit/booksplit_ids.json に退避する。
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from export_edit_sft import INSTRUCTION  # noqa: E402


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratify_assign(
  rows: list[dict],
  *,
  heldout_n: int,
  seed: int,
) -> dict[str, str]:
  """unit（section/hunk）の比率を保って heldout を取り、残りを train。"""
  by_unit: dict[str, list[dict]] = {"section": [], "hunk": []}
  for r in rows:
    unit = str(r.get("unit") or "hunk")
    if unit not in by_unit:
      unit = "hunk"
    by_unit[unit].append(r)

  total = len(rows)
  if heldout_n <= 0 or heldout_n >= total:
    raise SystemExit(f"heldout_n={heldout_n} out of range for n={total}")

  rng = random.Random(seed)
  assignment: dict[str, str] = {}
  heldout_left = heldout_n
  units = sorted(by_unit.keys())
  for i, unit in enumerate(units):
    bucket = list(by_unit[unit])
    rng.shuffle(bucket)
    if i == len(units) - 1:
      n_hold = min(heldout_left, len(bucket))
    else:
      n_hold = round(heldout_n * len(bucket) / total)
      n_hold = min(n_hold, len(bucket), heldout_left)
      # 残りユニットで埋められるよう上限
      remaining_capacity = sum(len(by_unit[u]) for u in units[i + 1 :])
      n_hold = min(n_hold, heldout_left)
      if heldout_left - n_hold > remaining_capacity:
        n_hold = heldout_left - remaining_capacity
    for r in bucket[:n_hold]:
      assignment[str(r["id"])] = "heldout"
    for r in bucket[n_hold:]:
      assignment[str(r["id"])] = "train"
    heldout_left -= n_hold

  if sum(1 for v in assignment.values() if v == "heldout") != heldout_n:
    # 端数調整: train から補う / heldout を戻す
    hold_ids = [i for i, s in assignment.items() if s == "heldout"]
    train_ids = [i for i, s in assignment.items() if s == "train"]
    rng.shuffle(hold_ids)
    rng.shuffle(train_ids)
    while len(hold_ids) < heldout_n and train_ids:
      tid = train_ids.pop()
      assignment[tid] = "heldout"
      hold_ids.append(tid)
    while len(hold_ids) > heldout_n:
      hid = hold_ids.pop()
      assignment[hid] = "train"
  return assignment


def to_chat(rec: dict) -> dict:
  meta = {
    "id": rec["id"],
    "project_id": rec.get("project_id"),
    "unit": rec.get("unit"),
    "corpus_source": rec.get("corpus_source"),
    "has_paragraph_break": rec.get("has_paragraph_break"),
  }
  return {
    "messages": [
      {"role": "user", "content": f"{INSTRUCTION}\n\n{rec['source_text']}"},
      {"role": "assistant", "content": rec["edited_text"]},
    ],
    "meta": meta,
  }


def write_chat_dirs(
  rows: list[dict],
  assignment: dict[str, str],
  *,
  section_out: Path,
  hunk_out: Path,
  all_out: Path,
) -> dict:
  buckets = {
    "section": {"train": [], "heldout": []},
    "hunk": {"train": [], "heldout": []},
  }
  for rec in rows:
    rid = str(rec["id"])
    split = assignment[rid]
    unit = "section" if rec.get("unit") == "section" else "hunk"
    buckets[unit][split].append(to_chat(rec))

  def dump(out_dir: Path, train: list, heldout: list) -> dict:
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "heldout.jsonl", heldout)
    return {"train": len(train), "heldout": len(heldout)}

  sec = dump(
    section_out,
    buckets["section"]["train"],
    buckets["section"]["heldout"],
  )
  hn = dump(
    hunk_out,
    buckets["hunk"]["train"],
    buckets["hunk"]["heldout"],
  )
  all_counts = dump(
    all_out,
    buckets["section"]["train"] + buckets["hunk"]["train"],
    buckets["section"]["heldout"] + buckets["hunk"]["heldout"],
  )
  return {"section": sec, "hunk_nopara": hn, "all": all_counts}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--canonical", default="data/revision_corpus/canonical.jsonl")
  parser.add_argument("--pairsplit-dir", default="data/pairsplit")
  parser.add_argument("--heldout-n", type=int, default=260)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--section-out", default="data/edit_sft_section")
  parser.add_argument("--hunk-nopara-out", default="data/edit_sft_hunk_nopara")
  parser.add_argument("--all-out", default="data/edit_sft_all")
  parser.add_argument(
    "--also-pref",
    action="store_true",
    help="作り直した canonical で make pref-keep-data 相当を実行",
  )
  args = parser.parse_args()

  canon_path = Path(args.canonical)
  if not canon_path.is_file():
    raise SystemExit(f"missing {canon_path}")

  rows = load_jsonl(canon_path)
  if len(rows) < args.heldout_n + 10:
    raise SystemExit(f"too few rows: {len(rows)}")

  pairsplit = Path(args.pairsplit_dir)
  pairsplit.mkdir(parents=True, exist_ok=True)

  # 旧書籍単位の id 一覧を退避
  book = {"train": [], "heldout": []}
  for r in rows:
    sp = str((r.get("meta") or {}).get("split") or "train")
    if sp not in book:
      sp = "train"
    book[sp].append(str(r["id"]))
  (pairsplit / "booksplit_ids.json").write_text(
    json.dumps(book, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )

  assignment = stratify_assign(rows, heldout_n=args.heldout_n, seed=args.seed)
  by_id = {str(r["id"]): r for r in rows}
  assign_rows = [
    {
      "id": rid,
      "split": split,
      "unit": by_id[rid].get("unit"),
      "project_id": by_id[rid].get("project_id"),
    }
    for rid, split in sorted(assignment.items())
  ]
  write_jsonl(pairsplit / "assignment.jsonl", assign_rows)

  # canonical / keep の meta.split を更新
  updated: list[dict] = []
  keep_section: list[dict] = []
  keep_hunk: list[dict] = []
  for r in rows:
    rid = str(r["id"])
    meta = dict(r.get("meta") or {})
    meta["split"] = assignment[rid]
    meta["split_kind"] = "pair_stratified"
    meta["split_seed"] = args.seed
    nr = {**r, "meta": meta}
    updated.append(nr)
    if r.get("unit") == "section":
      keep_section.append(nr)
    else:
      keep_hunk.append(nr)

  write_jsonl(canon_path, updated)
  corpus = canon_path.parent
  write_jsonl(corpus / "keep_section.jsonl", keep_section)
  write_jsonl(corpus / "keep_hunk_nopara.jsonl", keep_hunk)

  chat_counts = write_chat_dirs(
    updated,
    assignment,
    section_out=Path(args.section_out),
    hunk_out=Path(args.hunk_nopara_out),
    all_out=Path(args.all_out),
  )

  unit_hold = Counter()
  unit_train = Counter()
  for rid, split in assignment.items():
    u = str(by_id[rid].get("unit") or "?")
    if split == "heldout":
      unit_hold[u] += 1
    else:
      unit_train[u] += 1

  report = {
    "built_at": _now(),
    "seed": args.seed,
    "heldout_n": args.heldout_n,
    "n_total": len(rows),
    "n_train": sum(1 for s in assignment.values() if s == "train"),
    "n_heldout": sum(1 for s in assignment.values() if s == "heldout"),
    "heldout_by_unit": dict(unit_hold),
    "train_by_unit": dict(unit_train),
    "instruction": INSTRUCTION,
    "chat_counts": chat_counts,
    "note": "pair-stratified split; book-level heldout retired",
  }
  (pairsplit / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  for out_dir in (Path(args.section_out), Path(args.hunk_nopara_out), Path(args.all_out)):
    (out_dir / "stats.json").write_text(
      json.dumps({**report, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2)
      + "\n",
      encoding="utf-8",
    )

  print(json.dumps(report, ensure_ascii=False, indent=2))

  if args.also_pref:
    for units, ds, split_dir, report_path in (
      (
        "hunk",
        "data/pref_keep/dataset_hunk.jsonl",
        "data/pref_keep_split_hunk",
        "data/pref_keep/build_report_hunk.json",
      ),
      (
        "section",
        "data/pref_keep/dataset_section.jsonl",
        "data/pref_keep_split_section",
        "data/pref_keep/build_report_section.json",
      ),
    ):
      cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_pref_from_keep.py"),
        "--input",
        str(canon_path),
        "--out-dataset",
        ds,
        "--out-split-dir",
        split_dir,
        "--report",
        report_path,
        "--units",
        units,
      ]
      print("run:", " ".join(cmd), flush=True)
      subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
  main()
