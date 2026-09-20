#!/usr/bin/env python3
"""B 生成実験のブラインド比較ペアと対応表を作る。"""
from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from b_generation_common import (
  eligible_sources,
  generation_id,
  load_manifest,
  load_records,
  manifest_sha256,
  sha256_text,
  write_json,
)

COMPARISONS = ("xy:x", "xy:y", "xy:composer", "xy:xy-base")
SEEDS = (0, 1, 2)
BLIND_SEED = 20260907


def load_jsonl(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def load_predictions_dir(pred_dir: Path) -> dict[str, dict]:
  by_id: dict[str, dict] = {}
  if not pred_dir.is_dir():
    return by_id
  for path in sorted(pred_dir.glob("*.jsonl")):
    for row in load_jsonl(path):
      gid = str(row.get("generation_id") or "")
      if not gid:
        raise ValueError(f"missing generation_id in {path}")
      if gid in by_id:
        raise ValueError(f"duplicate generation_id {gid!r}")
      by_id[gid] = row
  return by_id


def composer_generation(source_id: str, composer_text: str) -> dict:
  return {
    "generation_id": generation_id(condition="composer", source_id=source_id),
    "source_id": source_id,
    "condition": "composer",
    "train_seed": None,
    "checkpoint": "stored",
    "text": composer_text,
    "text_sha256": sha256_text(composer_text),
    "prompt_tokens": 0,
    "generated_tokens": 0,
    "finish_reason": "stored",
    "truncated": False,
    "error": None,
  }



def resolve_generation(
  by_id: dict[str, dict],
  *,
  condition: str,
  source_id: str,
  train_seed: int | None,
) -> dict:
  gid = generation_id(condition=condition, source_id=source_id, train_seed=train_seed)
  row = by_id.get(gid)
  if row is None:
    raise SystemExit(f"missing generation {gid}")
  return row


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=Path("data/b_generation/manifest.json"))
  parser.add_argument("--predictions", type=Path, default=Path("outputs/b_generation/predictions"))
  parser.add_argument("--comparisons", default=",".join(COMPARISONS))
  parser.add_argument("--blind-seed", type=int, default=BLIND_SEED)
  parser.add_argument("--out", type=Path, default=Path("data/b_generation/blind"))
  args = parser.parse_args()

  code_root = Path(__file__).resolve().parents[1]
  manifest_path = code_root / args.manifest
  manifest = load_manifest(manifest_path)
  records = load_records(manifest_path.parent / manifest["records_file"])
  mhash = manifest_sha256(manifest)
  holdout = eligible_sources(manifest, split="holdout")
  if not holdout:
    raise SystemExit("no holdout sources")

  pred_dir = code_root / args.predictions
  by_id = load_predictions_dir(pred_dir)
  for src in holdout:
    sid = src["source_id"]
    rec = records[sid]
    cg = composer_generation(sid, rec["composer"])
    by_id[cg["generation_id"]] = cg

  comparisons = [c.strip() for c in args.comparisons.split(",") if c.strip()]
  for c in comparisons:
    if c not in COMPARISONS:
      raise SystemExit(f"unsupported comparison {c!r}")

  rng = random.Random(args.blind_seed)
  pairs: list[dict] = []
  keys: list[dict] = []
  missing: list[str] = []

  for seed in SEEDS:
    for src in holdout:
      sid = src["source_id"]
      rec = records[sid]
      draft = rec["draft"]
      for comparison in comparisons:
        _, right_c = comparison.split(":", 1)
        try:
          xy_gen = resolve_generation(by_id, condition="xy", source_id=sid, train_seed=seed)
          if right_c == "x":
            other_gen = resolve_generation(by_id, condition="x", source_id=sid, train_seed=seed)
          elif right_c == "y":
            other_gen = resolve_generation(by_id, condition="y", source_id=sid, train_seed=seed)
          elif right_c == "composer":
            other_gen = by_id[generation_id(condition="composer", source_id=sid)]
          elif right_c == "xy-base":
            other_gen = resolve_generation(by_id, condition="xy-base", source_id=sid, train_seed=None)
          else:
            raise ValueError(right_c)
        except SystemExit as exc:
          missing.append(str(exc))
          continue

        swap = bool(rng.getrandbits(1))
        if not swap:
          left_gen, right_gen = xy_gen, other_gen
        else:
          left_gen, right_gen = other_gen, xy_gen
        left_text = left_gen["text"]
        right_text = right_gen["text"]

        pair_id = secrets.token_hex(16)
        pairs.append(
          {
            "schema_version": "b-blind-pair-v1",
            "pair_id": pair_id,
            "order": len(pairs),
            "context": {"draft": draft},
            "left": {"text": left_text, "empty": not left_text},
            "right": {"text": right_text, "empty": not right_text},
          }
        )
        keys.append(
          {
            "schema_version": "b-blind-key-v1",
            "pair_id": pair_id,
            "source_id": sid,
            "group_id": src["group_id"],
            "split": "holdout",
            "train_seed": seed,
            "comparison": comparison,
            "left": {
              "condition": left_gen["condition"],
              "generation_id": left_gen["generation_id"],
            },
            "right": {
              "condition": right_gen["condition"],
              "generation_id": right_gen["generation_id"],
            },
            "manifest_sha256": mhash,
            "left_finish_reason": left_gen.get("finish_reason"),
            "right_finish_reason": right_gen.get("finish_reason"),
          }
        )

  if missing:
    raise SystemExit("missing predictions:\n" + "\n".join(missing[:20]))

  out_dir = code_root / args.out
  out_dir.mkdir(parents=True, exist_ok=True)
  pairs_path = out_dir / "pairs.jsonl"
  keys_path = out_dir / "keys.jsonl"
  gens_path = out_dir / "generations.jsonl"

  with pairs_path.open("w", encoding="utf-8") as f:
    for row in pairs:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  with keys_path.open("w", encoding="utf-8") as f:
    for row in keys:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  with gens_path.open("w", encoding="utf-8") as f:
    for row in sorted(by_id.values(), key=lambda r: r["generation_id"]):
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  blind_config = {
    "blind_seed": args.blind_seed,
    "random_module": "random",
    "comparisons": comparisons,
    "train_seeds": list(SEEDS),
    "n_holdout": len(holdout),
    "n_pairs": len(pairs),
    "expected_pairs": len(holdout) * len(SEEDS) * len(comparisons),
    "manifest_sha256": mhash,
  }
  write_json(out_dir / "blind_config.json", blind_config)
  print(json.dumps(blind_config, ensure_ascii=False))


if __name__ == "__main__":
  main()
