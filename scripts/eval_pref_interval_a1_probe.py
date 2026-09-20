#!/usr/bin/env python3
"""pref-interval-a1-probe の区間検証 (H1 + 区間所属)。"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from a1_probe_interval_utils import char_edit_ratio, load_interval_rows
from train_pref_interval_a1_probe import delta_xy, eval_zone_accuracy
from train_pref_sentseq import (
  PreparedSentSeqData,
  SentSeqTrainConfig,
  encode_unique_sentences,
  load_sentseq_model_from_artifact,
  prepare_sentence_data,
  resolve_device,
  unique_preference_pairs,
)
from pref_static_utils import load_jsonl


def pearson(xs: list[float], ys: list[float]) -> float | None:
  if len(xs) < 2:
    return None
  x = np.array(xs, dtype=np.float64)
  y = np.array(ys, dtype=np.float64)
  if float(x.std()) == 0.0 or float(y.std()) == 0.0:
    return None
  return float(np.corrcoef(x, y)[0, 1])


def build_prepared(
  texts: list[str],
  cfg: SentSeqTrainConfig,
  *,
  device: torch.device,
) -> PreparedSentSeqData:
  pseudo = [
    {
      "source_text": t,
      "candidate_a": t,
      "candidate_b": t,
      "label": 1,
      "meta": {"pair_order": "chosen_first"},
    }
    for t in sorted(set(texts))
  ]
  sent_to_embedding = encode_unique_sentences(pseudo, cfg, device=device, show_progress_bar=False)
  return prepare_sentence_data(
    sorted(set(texts)),
    sent_to_embedding=sent_to_embedding,
    max_sents=cfg.max_sents,
    device=device,
  )


def h1_within_position(
  rows: list[dict],
  delta_y: list[float],
) -> dict[str, dict]:
  by_pos: dict[str, list[int]] = defaultdict(list)
  for i, row in enumerate(rows):
    by_pos[row["position"]].append(i)

  out: dict[str, dict] = {}
  for pos, idxs in sorted(by_pos.items()):
    dys = [delta_y[i] for i in idxs]
    len_diff = [float(len(rows[i]["y"]) - len(rows[i]["draft"])) for i in idxs]
    edit_ratio = [char_edit_ratio(rows[i]["draft"], rows[i]["y"]) for i in idxs]
    out[pos] = {
      "n": len(idxs),
      "pearson_delta_y_len_diff": pearson(dys, len_diff),
      "pearson_delta_y_edit_ratio": pearson(dys, edit_ratio),
    }
    if pos == "a":
      shorter = [i for i in idxs if len(rows[i]["y"]) < len(rows[i]["draft"])]
      longer = [i for i in idxs if len(rows[i]["y"]) >= len(rows[i]["draft"])]
      out[pos]["frac_delta_lt_0_if_shorter"] = (
        sum(delta_y[i] < 0 for i in shorter) / len(shorter) if shorter else None
      )
      out[pos]["frac_delta_lt_0_if_longer"] = (
        sum(delta_y[i] < 0 for i in longer) / len(longer) if longer else None
      )
  return out


@torch.no_grad()
def run_eval(
  model,
  rows: list[dict],
  prepared,
  *,
  device: torch.device,
  batch_size: int = 32,
  eps_zero: float = 0.25,
) -> dict:
  model.eval()
  delta_y_all: list[float] = []
  delta_g_all: list[float] = []

  for start in range(0, len(rows), batch_size):
    batch = rows[start : start + batch_size]
    drafts = [r["draft"] for r in batch]
    ys = [r["y"] for r in batch]
    golds = [r["gold"] for r in batch]
    dy = delta_xy(model, drafts, ys, prepared, device=device)
    dg = delta_xy(model, drafts, golds, prepared, device=device)
    delta_y_all.extend(dy.float().cpu().tolist())
    delta_g_all.extend(dg.float().cpu().tolist())

  zone = eval_zone_accuracy(
    model, rows, prepared, device=device, batch_size=batch_size, eps_zero=eps_zero
  )
  h1 = h1_within_position(rows, delta_y_all)

  # f(x,g) アンカー: 中央値
  anchor_median = float(np.median(delta_g_all)) if delta_g_all else math.nan

  return {
    "n": len(rows),
    "zone": zone,
    "anchor_delta_g_median": anchor_median,
    "h1_by_position": h1,
    "rows": [
      {
        "pair_id": row["pair_id"],
        "position": row["position"],
        "delta_y": delta_y_all[i],
        "delta_g": delta_g_all[i],
      }
      for i, row in enumerate(rows)
    ],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="outputs/pref-interval-a1-probe")
  parser.add_argument("--samples-dir", default="outputs/a1-probe")
  parser.add_argument("--judgments", default="outputs/a1-probe/position_judgments.jsonl")
  parser.add_argument("--valid-probe-file", default="data/a1_probe_interval/valid_probe.jsonl")
  parser.add_argument("--hunk-valid-file", default="data/pref_keep_split_hunk/valid.jsonl")
  parser.add_argument("--out", default="outputs/pref-interval-a1-probe/eval_report.json")
  parser.add_argument("--eps-zero", type=float, default=0.25)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  args = parser.parse_args()

  device = resolve_device(args.device)
  model_dir = Path(args.model)
  artifact = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
  model = load_sentseq_model_from_artifact(artifact, device=device)
  config = artifact["config"]

  cfg = SentSeqTrainConfig(
    sentence_model_name=config["sentence_model_name"],
    truncate_dim=config.get("truncate_dim"),
    text_prefix=config.get("text_prefix", "文章: "),
    max_seq_length=int(config.get("max_seq_length") or 256),
    d_model=int(config["d_model"]),
    num_layers=int(config["num_layers"]),
    max_sents=int(config["max_sents"]),
  )

  valid_path = Path(args.valid_probe_file)
  if valid_path.is_file():
    valid_rows = load_jsonl(valid_path)
  else:
    valid_rows = load_interval_rows(Path(args.samples_dir), Path(args.judgments))
    # fallback: use all 600 if split file missing
    print(f"warning: {valid_path} missing, evaluating all {len(valid_rows)} rows", flush=True)

  hunk_valid = unique_preference_pairs(load_jsonl(Path(args.hunk_valid_file)))
  hunk_probe_rows = [
    {
      "pair_id": f"hunk::{r['id']}",
      "position": "anchor_g",
      "draft": r["source_text"],
      "y": r["candidate_a"],
      "gold": r["candidate_a"],
    }
    for r in hunk_valid
  ]

  texts: list[str] = []
  for row in valid_rows + hunk_probe_rows:
    texts.extend([row["draft"], row["y"], row["gold"]])
  prepared = build_prepared(texts, cfg, device=device)

  report = {
    "probe_valid": run_eval(
      model,
      valid_rows,
      prepared,
      device=device,
      batch_size=args.batch_size,
      eps_zero=args.eps_zero,
    ),
  }

  # A1 hunk valid: f(x,g) アンカーのみ
  hunk_dg: list[float] = []
  for start in range(0, len(hunk_probe_rows), args.batch_size):
    batch = hunk_probe_rows[start : start + args.batch_size]
    drafts = [r["draft"] for r in batch]
    golds = [r["gold"] for r in batch]
    dg = delta_xy(model, drafts, golds, prepared, device=device)
    hunk_dg.extend(dg.float().cpu().tolist())
  report["hunk_valid_anchor"] = {
    "n": len(hunk_dg),
    "median_delta_g": float(np.median(hunk_dg)) if hunk_dg else math.nan,
    "frac_delta_g_near_1": (
      sum(abs(x - 1.0) <= args.eps_zero for x in hunk_dg) / len(hunk_dg) if hunk_dg else 0.0
    ),
  }

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({k: v for k, v in report["probe_valid"]["zone"].items()}, ensure_ascii=False), flush=True)
  print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
  main()
