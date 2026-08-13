#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path


DEFAULT_PROMPT = (
  "次の日本語テキストを、意味保持を最優先に技術文書として推敲してください。"
  "\n修正前にない含意を追加せず、段落内の主題連鎖を意識してください。"
)


def make_input_text(source_text: str, source_reference: str) -> str:
  text = source_text
  if source_reference:
    text += f"\n\n[参照]\n{source_reference}"
  return text


def make_rejected(mode: str, source_text: str, edited_text: str) -> str:
  if mode == "source":
    return source_text
  # very light "worse" candidate for preference training fallback
  # (keeps meaning mostly but weakens style)
  weakened = edited_text.replace("。", "。 ").replace("，", "、")
  if weakened == edited_text:
    return source_text
  return weakened


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--db", required=True, help="sqlite path")
  parser.add_argument("--out", required=True, help="output jsonl")
  parser.add_argument("--accepted-only", action="store_true")
  parser.add_argument(
    "--rejected-mode",
    default="source",
    choices=["source", "weakened"],
    help="how to produce rejected samples",
  )
  args = parser.parse_args()

  conn = sqlite3.connect(args.db)
  conn.row_factory = sqlite3.Row
  cur = conn.cursor()

  query = "SELECT * FROM examples"
  if args.accepted_only:
    query += " WHERE review_result='accepted'"
  query += " ORDER BY created_at ASC"
  rows = cur.execute(query).fetchall()

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  emitted = 0
  seen = set()
  with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
      if not row["source_text"] or not row["edited_text"]:
        continue
      key = (row["source_text"], row["edited_text"])
      if key in seen:
        continue
      seen.add(key)

      rejected = make_rejected(
        args.rejected_mode,
        row["source_text"],
        row["edited_text"],
      )
      if rejected == row["edited_text"]:
        continue

      rec = {
        "id": row["id"],
        "prompt": DEFAULT_PROMPT,
        "input": make_input_text(row["source_text"], row["source_reference"] or ""),
        "chosen": row["edited_text"],
        "rejected": rejected,
        "meta": {
          "project_id": row["project_id"],
          "source_reference": row["source_reference"] or "",
          "labels": json.loads(row["labels"] or "[]"),
          "created_at": row["created_at"],
          "rejected_mode": args.rejected_mode,
        },
      }
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")
      emitted += 1

  conn.close()
  print(f"exported: {emitted}")


if __name__ == "__main__":
  main()

