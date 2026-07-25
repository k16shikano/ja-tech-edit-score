#!/usr/bin/env python3
"""構成負例（人手編集）から pref 学習行を作る。

入力:
  - bases MD（崩れた構成の下書き）
  - edited MD（文移動中心の人手編集）

出力 pref 行（各例ごと）:
  source = base
  chosen = edited
  rejected ∈ {base, deg-join(edited), deg-split(edited), deg-reverse(edited)}
  （同一文面はスキップ）

swap 増強あり。既存 pref_dataset へマージして split し直すこともできる。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hard_eval_v2 import DEGRADATIONS, deg_join, deg_split, deg_reverse  # noqa: F401

SECTION_RE = re.compile(r"^## (cn-\d+)\b.*$", re.M)


def parse_sections(path: Path) -> dict[str, str]:
  text = path.read_text(encoding="utf-8")
  matches = list(SECTION_RE.finditer(text))
  if not matches:
    raise SystemExit(f"no cn-* sections in {path}")
  out: dict[str, str] = {}
  for i, m in enumerate(matches):
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    body = text[start:end].strip()
    body = re.sub(r"\n---\s*$", "", body).strip()
    # 箇条書き・番号リストが残っていないか検査
    for line in body.splitlines():
      if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
        raise SystemExit(
          f"{path}: {m.group(1)} still has list marker: {line!r}"
        )
    out[m.group(1)] = body + "\n"
  return out


def emit_pref_rows(
  *,
  item_id: str,
  base: str,
  edited: str,
  created_at: str,
) -> list[dict]:
  rejecteds: list[tuple[str, str]] = [("base", base)]
  seen = {edited.strip(), base.strip()}
  for deg_id, fn in DEGRADATIONS:
    deg_text = fn(edited)
    if deg_text.strip() in seen:
      continue
    seen.add(deg_text.strip())
    rejecteds.append((deg_id, deg_text))

  rows: list[dict] = []
  for rej_tag, rejected in rejecteds:
    base_id = f"{item_id}-{rej_tag}"
    labels = ["composition_neg", f"rej:{rej_tag}"]
    meta_common = {
      "project_id": "composition-neg",
      "source_reference": f"hard_eval/composition_neg:{item_id}",
      "labels": labels,
      "created_at": created_at,
      "base_id": base_id,
    }
    rows.append(
      {
        "id": base_id,
        "source_text": base,
        "candidate_a": edited,
        "candidate_b": rejected,
        "label": 1,
        "meta": {**meta_common, "pair_order": "chosen_first"},
      }
    )
    rows.append(
      {
        "id": base_id + "-swap",
        "source_text": base,
        "candidate_a": rejected,
        "candidate_b": edited,
        "label": 0,
        "meta": {**meta_common, "pair_order": "rejected_first"},
      }
    )
  return rows


def merge_pref(paths: list[Path], out: Path) -> tuple[int, int]:
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
    if "composition_neg" in (r.get("meta", {}).get("labels") or [])
  )
  return len(merged), tagged


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--bases",
    default="data/hard_eval/composition_neg_bases.md",
  )
  parser.add_argument(
    "--edited",
    default="data/hard_eval/composition_neg_edited.md",
  )
  parser.add_argument(
    "--out",
    default="data/pref_dataset_composition_neg.jsonl",
  )
  parser.add_argument(
    "--merge-into",
    default="",
    help="既存 pref_dataset.jsonl。指定時はマージして上書きし、split も更新",
  )
  parser.add_argument(
    "--split-dir",
    default="data/pref_split",
  )
  args = parser.parse_args()

  bases = parse_sections(Path(args.bases))
  edited = parse_sections(Path(args.edited))
  if set(bases) != set(edited):
    raise SystemExit(
      f"id mismatch: bases={sorted(bases)} edited={sorted(edited)}"
    )

  created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
  rows: list[dict] = []
  for item_id in sorted(bases):
    rows.extend(
      emit_pref_rows(
        item_id=item_id,
        base=bases[item_id],
        edited=edited[item_id],
        created_at=created_at,
      )
    )

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as f:
    for rec in rows:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")
  print(f"wrote {len(rows)} pref rows -> {out}")
  print(f"items: {len(bases)}  pairs/item≈{len(rows) // len(bases)}")

  if args.merge_into:
    merge_path = Path(args.merge_into)
    if not merge_path.is_file():
      raise SystemExit(f"missing merge base: {merge_path}")
    # 退避
    backup = merge_path.with_suffix(merge_path.suffix + ".pre_composition_neg")
    if not backup.exists():
      backup.write_bytes(merge_path.read_bytes())
      print(f"backup -> {backup}")
    n, tagged = merge_pref([merge_path, out], merge_path)
    print(f"merged pref_dataset: {n} rows (composition_neg-tagged: {tagged})")

    from subprocess import check_call

    root = Path(__file__).resolve().parents[1]
    py = root / ".venv" / "bin" / "python3"
    python = str(py if py.is_file() else "python3")
    check_call(
      [
        python,
        str(root / "scripts" / "split_pref_dataset.py"),
        "--input",
        str(merge_path),
        "--out-dir",
        args.split_dir,
        "--group-by",
        "base_id",
        "--force-train-label",
        "composition_neg",
      ]
    )


if __name__ == "__main__":
  main()
