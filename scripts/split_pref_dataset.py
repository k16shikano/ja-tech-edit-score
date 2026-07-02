#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def bucket_for_key(key: str) -> float:
  digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
  value = int(digest[:12], 16)
  return value / float(16**12 - 1)


def choose_split(score: float, train_ratio: float, valid_ratio: float) -> str:
  if score < train_ratio:
    return "train"
  if score < train_ratio + valid_ratio:
    return "valid"
  return "test"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, help="preference jsonl")
  parser.add_argument("--out-dir", required=True, help="output dir")
  parser.add_argument("--train-ratio", type=float, default=0.8)
  parser.add_argument("--valid-ratio", type=float, default=0.1)
  parser.add_argument(
    "--group-by",
    choices=["base_id", "source_reference", "source_text"],
    default="base_id",
    help="group key to keep related examples in the same split",
  )
  args = parser.parse_args()

  if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.train_ratio + args.valid_ratio >= 1:
    raise SystemExit("ratios must satisfy 0 < train,valid and train+valid < 1")

  in_path = Path(args.input)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  out_files = {
    "train": (out_dir / "train.jsonl").open("w", encoding="utf-8"),
    "valid": (out_dir / "valid.jsonl").open("w", encoding="utf-8"),
    "test": (out_dir / "test.jsonl").open("w", encoding="utf-8"),
  }
  counts = {"train": 0, "valid": 0, "test": 0}

  try:
    with in_path.open("r", encoding="utf-8") as src:
      for line in src:
        line = line.strip()
        if not line:
          continue
        rec = json.loads(line)
        meta = rec.get("meta", {})
        if args.group_by == "base_id":
          key = meta.get("base_id", rec["id"])
        elif args.group_by == "source_reference":
          key = meta.get("source_reference", rec["id"])
        else:
          key = rec.get("source_text", rec["id"])
        score = bucket_for_key(key)
        split = choose_split(score, args.train_ratio, args.valid_ratio)
        out_files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split] += 1
  finally:
    for f in out_files.values():
      f.close()

  for split in ("train", "valid", "test"):
    print(f"{split}: {counts[split]}")


if __name__ == "__main__":
  main()
