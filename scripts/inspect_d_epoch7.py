#!/usr/bin/env python3
"""D epoch7 判定器の復元情報を lock ファイルへ保存する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from b_generation_common import write_json
from pref_d_cross_encoder import CrossEncoderConfig
from pref_d_interval_utils import IntervalLossConfig


def file_sha256(path: Path) -> str | None:
  if not path.is_file():
    return None
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def git_info(code_root: Path) -> dict:
  try:
    commit = subprocess.check_output(
      ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
      ["git", "-C", str(code_root), "status", "--porcelain"], text=True
    ).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return {"commit": None, "dirty": None, "status": "git_unavailable"}
  return {
    "commit": commit,
    "dirty": bool(status),
    "status_porcelain": status or None,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("outputs/pref-d-modernbert-cv/fold0/best"),
  )
  parser.add_argument("--out", type=Path, default=Path("outputs/d_epoch7/d_epoch7.lock.json"))
  args = parser.parse_args()

  ckpt = args.checkpoint if args.checkpoint.is_absolute() else args.code_root / args.checkpoint
  if not (ckpt / "model.safetensors").is_file():
    raise SystemExit(f"missing checkpoint weights: {ckpt / 'model.safetensors'}")

  meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
  train_log = ckpt / "train_log.jsonl"
  train_history = []
  if train_log.is_file():
    with train_log.open(encoding="utf-8") as f:
      for line in f:
        if line.strip():
          train_history.append(json.loads(line))

  target_files = [
    Path(__file__).relative_to(args.code_root),
    Path("scripts/pref_d_cross_encoder.py"),
    Path("scripts/pref_d_interval_utils.py"),
    Path("scripts/train_pref_d_interval.py"),
  ]
  implementation = []
  for rel in target_files:
    path = args.code_root / rel
    implementation.append(
      {
        "path": str(rel),
        "sha256": file_sha256(path),
        "exists": path.is_file(),
      }
    )

  cfg = CrossEncoderConfig(
    base_model=str(meta.get("base_model", "sbintuitions/modernbert-ja-310m")),
    max_length=int(meta.get("max_length", 1024)),
  )
  loss_cfg = IntervalLossConfig()

  lock = {
    "schema_version": "d-epoch7-lock-v1",
    "git": git_info(args.code_root),
    "checkpoint": {
      "path": str(ckpt),
      "model_sha256": file_sha256(ckpt / "model.safetensors"),
      "config_sha256": file_sha256(ckpt / "config.json"),
      "meta": meta,
      "restorable": True,
      "best_epoch": meta.get("best_epoch"),
      "train_log_entries": len(train_history),
    },
    "implementation_files": implementation,
    "input": {
      "format": "cross_encoder_pair",
      "draft_field": "draft",
      "candidate_field": "y",
      "tokenizer_pair_order": ["draft", "candidate"],
      "truncation": "longest_first",
      "max_length": cfg.max_length,
    },
    "score": {
      "definition": "f(x,y)=g(x,y)-g(x,x)",
      "direction": "higher_is_better",
      "function": "pref_d_cross_encoder.forward_f_delta",
    },
    "loss": {
      "class": "IntervalLossConfig",
      "margin_a": loss_cfg.margin_a,
      "margin_c_low": loss_cfg.margin_c_low,
      "margin_c_high": loss_cfg.margin_c_high,
      "zone_weight": loss_cfg.zone_weight,
      "weight_a": loss_cfg.weight_a,
      "weight_b": loss_cfg.weight_b,
      "weight_c": loss_cfg.weight_c,
      "weight_d": loss_cfg.weight_d,
      "position_mapping": {"a": 0, "b": 1, "c": 2, "d": 3},
    },
    "selection": {
      "metric": "valid_primary_a_sign_acc",
      "note": "best/ holds weights from epoch with highest valid_primary_a_sign_acc",
    },
    "inference": {
      "dtype": "float32",
      "eval_mode": True,
    },
  }

  out = args.out if args.out.is_absolute() else args.code_root / args.out
  write_json(out, lock)
  print(json.dumps({"wrote": str(out), "best_epoch": meta.get("best_epoch")}, ensure_ascii=False))


if __name__ == "__main__":
  main()
