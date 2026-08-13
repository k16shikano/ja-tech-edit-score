#!/usr/bin/env python3
"""レビュー keep を学習用に書き出す。

規則:
- 8310（節）・8312（節追加）・8311（hunk キュー）の keep をすべて採用
- 推敲前・推敲後のどちらかに空行（\\n\\n）があれば「節」側
- どちらにも空行がなければ「空行なし hunk」側
- 8310 と 8312 は同質。8311 のうち空行ありも節側へ合流

出力（chat SFT 形式）:
  data/edit_sft_section/{train,heldout}.jsonl
  data/edit_sft_hunk_nopara/{train,heldout}.jsonl

正本（examples 互換）:
  data/revision_corpus/keep_section.jsonl
  data/revision_corpus/keep_hunk_nopara.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from export_edit_sft import INSTRUCTION, INSTRUCTION_LEGACY, INSTRUCTION_V1

QUEUES = (
  {
    "name": "8310_section",
    "dir": "data/edit_sft",
    "default_unit": "section",
  },
  {
    "name": "8312_section_extra",
    "dir": "data/edit_sft_section_extra_review",
    "default_unit": "section",
  },
  {
    "name": "8311_hunk_queue",
    "dir": "data/edit_sft_hunk_review",
    "default_unit": "hunk",
  },
)


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def has_blank_line(text: str) -> bool:
  return "\n\n" in (text or "")


def load_state(path: Path) -> dict:
  if not path.is_file():
    return {}
  st = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(st, dict) and "items" in st and isinstance(st["items"], dict):
    return st["items"]
  return st if isinstance(st, dict) else {}


def parse_chat_row(obj: dict) -> tuple[str, str, dict]:
  draft = revised = ""
  for msg in obj.get("messages") or []:
    if msg.get("role") == "user":
      content = str(msg.get("content") or "")
      for instr in (INSTRUCTION, INSTRUCTION_V1, INSTRUCTION_LEGACY):
        prefix = instr + "\n\n"
        if content.startswith(prefix):
          draft = content[len(prefix) :]
          break
      else:
        if "\n\n" in content:
          draft = content.split("\n\n", 1)[-1]
        else:
          draft = content
    elif msg.get("role") == "assistant":
      revised = str(msg.get("content") or "")
  meta = dict(obj.get("meta") or {})
  return draft, revised, meta


def iter_keep_rows(queue: dict) -> list[dict]:
  root = Path(queue["dir"])
  state = load_state(root / "review_state.json")
  out: list[dict] = []
  for split in ("train", "heldout"):
    path = root / f"{split}.jsonl"
    if not path.is_file():
      continue
    for line in path.open(encoding="utf-8"):
      if not line.strip():
        continue
      obj = json.loads(line)
      draft, revised, meta = parse_chat_row(obj)
      rid = str(meta.get("id") or "")
      if not rid:
        continue
      dec = state.get(rid)
      if not isinstance(dec, dict) or dec.get("status") != "keep":
        continue
      if "draft" in dec and dec["draft"] is not None:
        draft = str(dec["draft"])
      if "revised" in dec and dec["revised"] is not None:
        revised = str(dec["revised"])
      draft = draft.strip()
      revised = revised.strip()
      if not draft or not revised or draft == revised:
        continue
      out.append(
        {
          "id": rid,
          "project_id": str(meta.get("project_id") or "unknown"),
          "draft": draft,
          "revised": revised,
          "split": split,
          "queue": queue["name"],
          "default_unit": queue["default_unit"],
          "source_meta": meta,
          "review_note": str(dec.get("note") or ""),
        }
      )
  return out


def to_chat(row: dict, *, bucket: str, unit: str) -> dict:
  meta = {
    "id": row["id"],
    "project_id": row["project_id"],
    "review_status": "keep",
    "review_note": row["review_note"],
    "bucket": bucket,
    "unit": unit,
    "has_blank_line": has_blank_line(row["draft"]) or has_blank_line(row["revised"]),
    "review_queue": row["queue"],
    "source": row["source_meta"].get("source") or row["queue"],
  }
  for k in (
    "char_similarity",
    "bigram_jaccard",
    "length_ratio",
    "corpus_source",
    "compare_kind",
  ):
    if k in row["source_meta"]:
      meta[k] = row["source_meta"][k]
  return {
    "messages": [
      {"role": "user", "content": f"{INSTRUCTION}\n\n{row['draft']}"},
      {"role": "assistant", "content": row["revised"]},
    ],
    "meta": meta,
  }


def to_examples(row: dict, *, bucket: str, unit: str) -> dict:
  rid = row["id"]
  if rid.startswith("section:"):
    rid = rid[len("section:") :]
  elif rid.startswith("hunk:"):
    rid = rid[len("hunk:") :]
  return {
    "id": f"keep-{bucket}:{rid}",
    "project_id": row["project_id"],
    "source_text": row["draft"],
    "edited_text": row["revised"],
    "source_reference": str(
      row["source_meta"].get("section_key")
      or row["source_meta"].get("source_reference")
      or row["source_meta"].get("path")
      or ""
    ),
    "rationale": row["review_note"],
    "labels": [bucket, "human_reviewed", unit],
    "author": "human",
    "review_result": "accepted",
    "created_at": _now(),
    "unit": unit,
    "compare_kind": "human_reviewed",
    "quality": "keep",
    "has_paragraph_break": has_blank_line(row["draft"]) or has_blank_line(row["revised"]),
    "corpus_source": row["queue"],
    "meta": {
      "split": row["split"],
      "review_queue": row["queue"],
      "bucket": bucket,
      "original_id": row["id"],
    },
  }


def write_jsonl(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_chat_split(out_dir: Path, chats: list[dict]) -> dict:
  out_dir.mkdir(parents=True, exist_ok=True)
  by_split = {"train": [], "heldout": []}
  for chat in chats:
    split = "heldout" if chat["meta"].get("_split") == "heldout" else "train"
    # _split is internal; strip before write
    meta = {k: v for k, v in chat["meta"].items() if k != "_split"}
    row = {"messages": chat["messages"], "meta": meta}
    by_split[split].append(row)
  for split, rows in by_split.items():
    write_jsonl(out_dir / f"{split}.jsonl", rows)
  return {k: len(v) for k, v in by_split.items()}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--section-out", default="data/edit_sft_section")
  parser.add_argument("--hunk-nopara-out", default="data/edit_sft_hunk_nopara")
  parser.add_argument(
    "--all-out",
    default="data/edit_sft_all",
    help="節+空行なしhunk の合流（Qwen SFT 本線）",
  )
  parser.add_argument("--corpus-dir", default="data/revision_corpus")
  args = parser.parse_args()

  collected: list[dict] = []
  for q in QUEUES:
    rows = iter_keep_rows(q)
    collected.extend(rows)

  # 同一本文は先勝ち（8310 → 8312 → 8311）
  seen: set[tuple[str, str, str]] = set()
  unique: list[dict] = []
  dup = 0
  for row in collected:
    key = (row["project_id"], row["draft"], row["revised"])
    if key in seen:
      dup += 1
      continue
    seen.add(key)
    unique.append(row)

  section_rows: list[dict] = []
  hunk_rows: list[dict] = []
  from_queue = Counter()
  for row in unique:
    blank = has_blank_line(row["draft"]) or has_blank_line(row["revised"])
    bucket = "section" if blank else "hunk_nopara"
    unit = "section" if blank else "hunk"
    row = {**row, "bucket": bucket, "unit": unit}
    from_queue[(row["queue"], bucket)] += 1
    if blank:
      section_rows.append(row)
    else:
      hunk_rows.append(row)

  def pack(rows: list[dict], bucket: str, unit: str) -> tuple[list[dict], list[dict]]:
    chats: list[dict] = []
    examples: list[dict] = []
    for row in rows:
      chat = to_chat(row, bucket=bucket, unit=unit)
      chat["meta"]["_split"] = row["split"]
      chats.append(chat)
      examples.append(to_examples(row, bucket=bucket, unit=unit))
    return chats, examples

  sec_chats, sec_ex = pack(section_rows, "section", "section")
  hn_chats, hn_ex = pack(hunk_rows, "hunk_nopara", "hunk")

  sec_split = write_chat_split(Path(args.section_out), sec_chats)
  hn_split = write_chat_split(Path(args.hunk_nopara_out), hn_chats)
  # SFT 本線: 二系統を管理したまま合流して学習する
  all_chats = sec_chats + hn_chats
  all_split = write_chat_split(Path(args.all_out), all_chats)

  corpus = Path(args.corpus_dir)
  write_jsonl(corpus / "keep_section.jsonl", sec_ex)
  write_jsonl(corpus / "keep_hunk_nopara.jsonl", hn_ex)

  stats = {
    "built_at": _now(),
    "rule": "blank_line_in_draft_or_revised -> section; else hunk_nopara",
    "n_keep_input": len(collected),
    "n_after_dedupe": len(unique),
    "n_dup_dropped": dup,
    "section": {
      "n": len(section_rows),
      **sec_split,
      "out_dir": args.section_out,
      "corpus": str(corpus / "keep_section.jsonl"),
    },
    "hunk_nopara": {
      "n": len(hunk_rows),
      **hn_split,
      "out_dir": args.hunk_nopara_out,
      "corpus": str(corpus / "keep_hunk_nopara.jsonl"),
    },
    "all": {
      "n": len(section_rows) + len(hunk_rows),
      **all_split,
      "out_dir": args.all_out,
      "note": "Qwen SFT 本線（節+空行なしhunk）",
    },
    "from_queue_bucket": {f"{q}/{b}": n for (q, b), n in sorted(from_queue.items())},
    "projects_section": dict(Counter(r["project_id"] for r in section_rows)),
    "projects_hunk_nopara": dict(Counter(r["project_id"] for r in hunk_rows)),
  }
  for out_dir in (
    Path(args.section_out),
    Path(args.hunk_nopara_out),
    Path(args.all_out),
    corpus,
  ):
    out_dir.mkdir(parents=True, exist_ok=True)
  for p in (
    Path(args.section_out) / "stats.json",
    Path(args.hunk_nopara_out) / "stats.json",
    Path(args.all_out) / "stats.json",
    corpus / "keep_export_stats.json",
  ):
    p.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
