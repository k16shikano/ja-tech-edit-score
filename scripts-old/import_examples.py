#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS examples (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  source_text TEXT NOT NULL,
  edited_text TEXT NOT NULL,
  source_reference TEXT,
  rationale TEXT,
  labels TEXT,
  author TEXT NOT NULL,
  review_result TEXT,
  created_at TEXT NOT NULL
);
"""


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, help="input jsonl")
  parser.add_argument("--db", required=True, help="sqlite path")
  args = parser.parse_args()

  in_path = Path(args.input)
  db_path = Path(args.db)
  db_path.parent.mkdir(parents=True, exist_ok=True)

  conn = sqlite3.connect(db_path)
  cur = conn.cursor()
  cur.execute(DDL)

  inserted = 0
  with in_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      cur.execute(
        """
        INSERT OR REPLACE INTO examples
        (id, project_id, source_text, edited_text, source_reference, rationale, labels, author, review_result, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          row["id"],
          row["project_id"],
          row["source_text"],
          row["edited_text"],
          row.get("source_reference"),
          row.get("rationale"),
          json.dumps(row.get("labels", []), ensure_ascii=False),
          row["author"],
          row.get("review_result"),
          row["created_at"],
        ),
      )
      inserted += 1

  conn.commit()
  conn.close()
  print(f"imported: {inserted}")


if __name__ == "__main__":
  main()

