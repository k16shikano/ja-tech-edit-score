#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, help="input DPO jsonl")
  parser.add_argument("--out", required=True, help="output preference-classification jsonl")
  parser.add_argument("--project-id", default="", help="optional project id filter")
  parser.add_argument(
    "--augment-swap",
    action="store_true",
    help="emit swapped candidate order examples as well",
  )
  args = parser.parse_args()

  in_path = Path(args.input)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  emitted = 0
  with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
    for line in src:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      meta = rec.get("meta", {})
      if args.project_id and meta.get("project_id") != args.project_id:
        continue

      base = {
        "id": rec["id"],
        "source_text": rec["input"],
        "candidate_a": rec["chosen"],
        "candidate_b": rec["rejected"],
        "label": 1,
        "meta": {
          "project_id": meta.get("project_id", ""),
          "source_reference": meta.get("source_reference", ""),
          "labels": meta.get("labels", []),
          "created_at": meta.get("created_at", ""),
          "base_id": rec["id"],
          "pair_order": "chosen_first",
        },
      }
      dst.write(json.dumps(base, ensure_ascii=False) + "\n")
      emitted += 1

      if args.augment_swap:
        swapped = {
          "id": rec["id"] + "-swap",
          "source_text": rec["input"],
          "candidate_a": rec["rejected"],
          "candidate_b": rec["chosen"],
          "label": 0,
          "meta": {
            "project_id": meta.get("project_id", ""),
            "source_reference": meta.get("source_reference", ""),
            "labels": meta.get("labels", []),
            "created_at": meta.get("created_at", ""),
            "base_id": rec["id"],
            "pair_order": "rejected_first",
          },
        }
        dst.write(json.dumps(swapped, ensure_ascii=False) + "\n")
        emitted += 1

  print(f"exported: {emitted}")


if __name__ == "__main__":
  main()
