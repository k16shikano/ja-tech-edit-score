#!/usr/bin/env python3
"""machine_revisions から「人間編集 ≻ 機械推敲案」の pref 行を作る。

入力:
  - data/machine_revisions.jsonl（status=rejected は除外）
  - data/revision_pairs.jsonl（draft_id で突き合わせ）

出力 pref 行:
  source_text = 下書き
  candidate_a = 人間編集（revised）
  candidate_b = 機械推敲案
  swap 増強あり
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from steering_utils import load_jsonl, write_jsonl


def merge_pref(paths: list[Path], out: Path) -> tuple[int, int]:
  """source/candidate/label で重複排除してマージする。"""
  merged: list[dict] = []
  seen: set[tuple] = set()

  def key(rec: dict) -> tuple:
    return (
      rec.get("source_text", ""),
      rec.get("candidate_a", ""),
      rec.get("candidate_b", ""),
      int(rec.get("label", -1)),
    )

  for path in paths:
    if not path.is_file() or path.stat().st_size == 0:
      continue
    for line in path.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      rec = json.loads(line)
      k = key(rec)
      if k in seen:
        continue
      seen.add(k)
      merged.append(rec)

  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for rec in merged:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")
  tagged = sum(
    1
    for r in merged
    if "machine_neg" in (r.get("meta", {}).get("labels") or [])
  )
  return len(merged), tagged


def index_revision_pairs(rows: list[dict]) -> dict[str, dict]:
  out: dict[str, dict] = {}
  for row in rows:
    row_id = str(row.get("id") or "")
    if row_id:
      out[row_id] = row
  return out


def emit_pref_rows(
  *,
  draft_id: str,
  project_id: str,
  draft: str,
  revised: str,
  machine_text: str,
  generator: str,
  created_at: str,
) -> list[dict]:
  if revised.strip() == machine_text.strip():
    return []

  base_id = draft_id
  labels = ["machine_neg", f"gen:{generator}"]
  meta_common = {
    "project_id": project_id,
    "source_reference": f"machine_revision:{draft_id}",
    "labels": labels,
    "created_at": created_at,
    "base_id": base_id,
    "draft_id": draft_id,
    "generator": generator,
  }
  row_id = f"{draft_id}__machine_neg"
  return [
    {
      "id": row_id,
      "source_text": draft,
      "candidate_a": revised,
      "candidate_b": machine_text,
      "label": 1,
      "meta": {**meta_common, "pair_order": "chosen_first"},
    },
    {
      "id": row_id + "-swap",
      "source_text": draft,
      "candidate_a": machine_text,
      "candidate_b": revised,
      "label": 0,
      "meta": {**meta_common, "pair_order": "rejected_first"},
    },
  ]


def build_machine_neg_rows(
  revisions: list[dict],
  pairs_by_id: dict[str, dict],
) -> tuple[list[dict], dict[str, int]]:
  stats = {
    "input": len(revisions),
    "rejected_status": 0,
    "missing_pair": 0,
    "identical_to_human": 0,
    "empty_text": 0,
    "emitted_items": 0,
  }
  rows: list[dict] = []
  created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

  for rev in revisions:
    if rev.get("status") == "rejected":
      stats["rejected_status"] += 1
      continue

    draft_id = str(rev.get("draft_id") or "")
    machine_text = str(rev.get("text") or "").strip()
    if not draft_id or not machine_text:
      stats["empty_text"] += 1
      continue

    pair = pairs_by_id.get(draft_id)
    if not pair:
      stats["missing_pair"] += 1
      continue

    draft = str(pair.get("draft") or "")
    revised = str(pair.get("revised") or "")
    if not draft or not revised:
      stats["empty_text"] += 1
      continue

    generator = str(rev.get("generator") or "unknown")
    item_rows = emit_pref_rows(
      draft_id=draft_id,
      project_id=str(rev.get("project_id") or pair.get("project_id") or ""),
      draft=draft,
      revised=revised,
      machine_text=machine_text,
      generator=generator,
      created_at=created_at,
    )
    if not item_rows:
      stats["identical_to_human"] += 1
      continue

    rows.extend(item_rows)
    stats["emitted_items"] += 1

  return rows, stats


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--revisions",
    default="data/machine_revisions.jsonl",
    help="machine revisions JSONL",
  )
  parser.add_argument(
    "--pairs",
    default="data/revision_pairs.jsonl",
    help="revision pairs JSONL",
  )
  parser.add_argument(
    "--out",
    default="data/pref_dataset_machine_neg.jsonl",
    help="output pref JSONL",
  )
  parser.add_argument(
    "--merge-into",
    default="",
    help="既存 pref_dataset.jsonl。指定時は別ファイルへマージ結果を書く（上書きしない）",
  )
  parser.add_argument(
    "--merged-out",
    default="data/pref_dataset_merged_machine_neg.jsonl",
    help="--merge-into 指定時のマージ出力先",
  )
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  revisions_path = (root / args.revisions).resolve()
  pairs_path = (root / args.pairs).resolve()
  out_path = (root / args.out).resolve()

  if not revisions_path.is_file():
    raise SystemExit(f"missing revisions: {revisions_path}")
  if not pairs_path.is_file():
    raise SystemExit(f"missing pairs: {pairs_path}")

  revisions = load_jsonl(revisions_path)
  pairs_by_id = index_revision_pairs(load_jsonl(pairs_path))
  rows, stats = build_machine_neg_rows(revisions, pairs_by_id)

  write_jsonl(out_path, rows)
  print(f"wrote {len(rows)} pref rows -> {out_path}")
  print(
    "stats: "
    f"input={stats['input']} "
    f"items={stats['emitted_items']} "
    f"skip_rejected={stats['rejected_status']} "
    f"skip_missing_pair={stats['missing_pair']} "
    f"skip_identical={stats['identical_to_human']} "
    f"skip_empty={stats['empty_text']}"
  )

  if args.merge_into:
    merge_base = (root / args.merge_into).resolve()
    if not merge_base.is_file():
      raise SystemExit(f"missing merge base: {merge_base}")
    merged_out = (root / args.merged_out).resolve()
    n, tagged = merge_pref([merge_base, out_path], merged_out)
    print(f"merged pref_dataset: {n} rows -> {merged_out} (machine_neg-tagged: {tagged})")


if __name__ == "__main__":
  main()
