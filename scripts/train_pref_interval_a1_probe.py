#!/usr/bin/env python3
"""下書き x と候補 y から f(x,y)=s(x,y)-s(x,x) を区間損失で学習する。

教師:
- A1 probe 600 対の位置ラベル (a/eq/b/c/d)
- A1 hunk の下書き対人間の推敲 (f(x,g)=1 のアンカー)
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from a1_probe_interval_utils import (
  collect_interval_texts,
  load_interval_rows,
  split_interval_rows,
  write_jsonl,
)
from pref_static_utils import load_jsonl
from train_pref_sentseq import (
  SentSeqEncoder,
  SentSeqRewardModel,
  SentSeqTrainConfig,
  encode_texts_to_doc_vectors,
  encode_unique_sentences,
  prepare_sentence_data,
  resolve_device,
  save_sentseq_model,
  unique_preference_pairs,
)

KIND = "pref-interval-a1-probe"


@dataclass
class IntervalLossConfig:
  margin_a: float = 0.0
  margin_c_low: float = 0.05
  margin_c_high: float = 0.05
  anchor_weight: float = 1.0
  zone_weight: float = 1.0


def score_source_candidate(
  model: SentSeqRewardModel,
  sources: list[str],
  candidates: list[str],
  prepared,
  *,
  device: torch.device,
) -> torch.Tensor:
  len_s = torch.tensor([float(len(t)) for t in sources], dtype=torch.float32, device=device)
  len_c = torch.tensor([float(len(t)) for t in candidates], dtype=torch.float32, device=device)
  v_src = encode_texts_to_doc_vectors(model, sources, prepared, device=device)
  v_c = encode_texts_to_doc_vectors(model, candidates, prepared, device=device)
  return model.score_from_doc_vectors_fast(
    v_src, v_c, len_source=len_s, len_candidate=len_c
  )


def delta_xy(
  model: SentSeqRewardModel,
  sources: list[str],
  candidates: list[str],
  prepared,
  *,
  device: torch.device,
) -> torch.Tensor:
  s_xy = score_source_candidate(model, sources, candidates, prepared, device=device)
  s_xx = score_source_candidate(model, sources, sources, prepared, device=device)
  return s_xy - s_xx


def interval_zone_loss(
  delta_y: torch.Tensor,
  delta_g: torch.Tensor,
  positions: list[str],
  *,
  cfg: IntervalLossConfig,
) -> torch.Tensor:
  losses: list[torch.Tensor] = []
  for i, pos in enumerate(positions):
    dy = delta_y[i]
    dg = delta_g[i]
    losses.append(cfg.anchor_weight * (dg - 1.0) ** 2)
    if pos == "a":
      losses.append(cfg.zone_weight * F.relu(dy + cfg.margin_a) ** 2)
    elif pos in ("b", "eq"):
      losses.append(cfg.zone_weight * dy ** 2)
    elif pos == "c":
      losses.append(cfg.zone_weight * F.relu(cfg.margin_c_low - dy) ** 2)
      losses.append(cfg.zone_weight * F.relu(dy - dg + cfg.margin_c_high) ** 2)
    elif pos == "d":
      losses.append(cfg.zone_weight * (dy - dg) ** 2)
    else:
      raise ValueError(f"unknown position {pos!r}")
  if not losses:
    return torch.tensor(0.0, device=delta_y.device)
  return torch.stack(losses).mean()


def hunk_anchor_loss(
  delta_g: torch.Tensor,
  *,
  cfg: IntervalLossConfig,
) -> torch.Tensor:
  return cfg.anchor_weight * ((delta_g - 1.0) ** 2).mean()


@torch.no_grad()
def eval_zone_accuracy(
  model: SentSeqRewardModel,
  rows: list[dict],
  prepared,
  *,
  device: torch.device,
  eps_zero: float = 0.25,
  batch_size: int = 32,
) -> dict[str, float]:
  model.eval()
  ok = 0
  total = 0
  by_pos: dict[str, list[bool]] = {p: [] for p in ("a", "eq", "b", "c", "d")}

  for start in range(0, len(rows), batch_size):
    batch = rows[start : start + batch_size]
    drafts = [r["draft"] for r in batch]
    ys = [r["y"] for r in batch]
    golds = [r["gold"] for r in batch]
    dy = delta_xy(model, drafts, ys, prepared, device=device)
    dg = delta_xy(model, drafts, golds, prepared, device=device)
    for i, row in enumerate(batch):
      pos = row["position"]
      dy_i = float(dy[i].item())
      dg_i = float(dg[i].item())
      if pos == "a":
        hit = dy_i < 0
      elif pos in ("b", "eq"):
        hit = abs(dy_i) <= eps_zero
      elif pos == "c":
        hit = dy_i > 0 and dy_i < dg_i
      elif pos == "d":
        hit = abs(dy_i - dg_i) <= eps_zero
      else:
        hit = False
      by_pos[pos].append(hit)
      ok += int(hit)
      total += 1

  out: dict[str, float] = {
    "zone_accuracy": ok / total if total else 0.0,
    "n": float(total),
  }
  for pos, hits in by_pos.items():
    if hits:
      out[f"zone_acc_{pos}"] = sum(hits) / len(hits)
      out[f"n_{pos}"] = float(len(hits))
  return out


def train_interval_model(
  probe_train: list[dict],
  probe_valid: list[dict],
  hunk_rows: list[dict],
  cfg: SentSeqTrainConfig,
  loss_cfg: IntervalLossConfig,
  *,
  device: torch.device,
  output_dir: Path,
) -> dict:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  hunk_rows = unique_preference_pairs(hunk_rows)
  all_texts = collect_interval_texts(probe_train + probe_valid)
  for row in hunk_rows:
    for key in ("source_text", "candidate_a", "candidate_b"):
      t = str(row[key])
      if t not in all_texts:
        all_texts.append(t)

  pseudo_rows = [
    {
      "source_text": t,
      "candidate_a": t,
      "candidate_b": t,
      "label": 1,
      "meta": {"pair_order": "chosen_first"},
    }
    for t in sorted(set(all_texts))
  ]
  sent_to_embedding = encode_unique_sentences(pseudo_rows, cfg, device=device)

  embed_dim = next(iter(sent_to_embedding.values())).shape[0]
  prepared = prepare_sentence_data(
    all_texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=cfg.max_sents,
    device=device,
  )

  feature_dim = cfg.d_model * 4 + 5
  model = SentSeqRewardModel(
    SentSeqEncoder(
      embed_dim=embed_dim,
      d_model=cfg.d_model,
      nhead=cfg.nhead,
      num_layers=cfg.num_layers,
      dim_feedforward=cfg.dim_feedforward,
      dropout=cfg.dropout,
      max_sents=cfg.max_sents,
    ),
    feature_dim=feature_dim,
  ).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

  n_probe = len(probe_train)
  n_hunk = len(hunk_rows)
  best_zone = -1.0
  best_epoch = -1
  best_state = None

  print(
    f"train: probe={n_probe} hunk_anchor={n_hunk} valid_probe={len(probe_valid)} "
    f"device={device.type} epochs={cfg.epochs}",
    flush=True,
  )

  for epoch in range(cfg.epochs):
    model.train()
    probe_order = list(range(n_probe))
    random.shuffle(probe_order)
    hunk_order = list(range(n_hunk))
    random.shuffle(hunk_order)

    epoch_losses: list[float] = []
    # probe batches
    for start in range(0, n_probe, cfg.batch_size):
      idxs = probe_order[start : start + cfg.batch_size]
      batch = [probe_train[i] for i in idxs]
      drafts = [r["draft"] for r in batch]
      ys = [r["y"] for r in batch]
      golds = [r["gold"] for r in batch]
      positions = [r["position"] for r in batch]
      dy = delta_xy(model, drafts, ys, prepared, device=device)
      dg = delta_xy(model, drafts, golds, prepared, device=device)
      loss = interval_zone_loss(dy, dg, positions, cfg=loss_cfg)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_losses.append(float(loss.item()))

    # hunk anchor batches
    for start in range(0, n_hunk, cfg.batch_size):
      idxs = hunk_order[start : start + cfg.batch_size]
      batch = [hunk_rows[i] for i in idxs]
      drafts = [r["source_text"] for r in batch]
      golds = [r["candidate_a"] for r in batch]
      dg = delta_xy(model, drafts, golds, prepared, device=device)
      loss = hunk_anchor_loss(dg, cfg=loss_cfg)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_losses.append(float(loss.item()))

    valid_metrics = eval_zone_accuracy(
      model, probe_valid, prepared, device=device, batch_size=cfg.batch_size
    )
    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    print(
      f"epoch {epoch + 1}/{cfg.epochs} loss={mean_loss:.4f} "
      f"valid_zone={valid_metrics['zone_accuracy']:.4f}",
      flush=True,
    )
    if valid_metrics["zone_accuracy"] > best_zone:
      best_zone = valid_metrics["zone_accuracy"]
      best_epoch = epoch + 1
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()

  save_sentseq_model(model, cfg, embed_dim=embed_dim, output_dir=output_dir)
  meta = {
    "kind": KIND,
    "best_epoch": best_epoch,
    "best_valid_zone_accuracy": best_zone,
    "loss_config": {
      "margin_a": loss_cfg.margin_a,
      "margin_c_low": loss_cfg.margin_c_low,
      "margin_c_high": loss_cfg.margin_c_high,
      "anchor_weight": loss_cfg.anchor_weight,
      "zone_weight": loss_cfg.zone_weight,
    },
    "train_probe": n_probe,
    "train_hunk_anchor": n_hunk,
    "valid_probe": len(probe_valid),
  }
  (output_dir / "train_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"best epoch: {best_epoch}/{cfg.epochs} valid_zone={best_zone:.4f}", flush=True)
  return meta


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--samples-dir", default="outputs/a1-probe")
  parser.add_argument("--judgments", default="outputs/a1-probe/position_judgments.jsonl")
  parser.add_argument("--hunk-train-file", default="data/pref_keep_split_hunk/train.jsonl")
  parser.add_argument("--eval-pair-ids", default="", help="optional jsonl of pair_id for eval split")
  parser.add_argument("--eval-fraction", type=float, default=0.2)
  parser.add_argument("--out-data-dir", default="data/a1_probe_interval")
  parser.add_argument("--output-dir", default="outputs/pref-interval-a1-probe")
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--num-layers", type=int, default=2)
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
  args = parser.parse_args()

  device = resolve_device(args.device)
  samples_dir = Path(args.samples_dir)
  judgments_path = Path(args.judgments)
  out_data = Path(args.out_data_dir)
  output_dir = Path(args.output_dir)

  all_rows = load_interval_rows(samples_dir, judgments_path)
  if len(all_rows) != 600:
    raise SystemExit(f"expected 600 probe rows, got {len(all_rows)}")

  eval_ids = None
  if args.eval_pair_ids:
    from a1_probe_interval_utils import load_eval_pair_ids

    eval_ids = load_eval_pair_ids(Path(args.eval_pair_ids))

  train_rows, valid_rows, split_stats = split_interval_rows(
    all_rows,
    eval_pair_ids=eval_ids,
    eval_fraction=args.eval_fraction,
    seed=args.seed,
  )
  write_jsonl(out_data / "train_probe.jsonl", train_rows)
  write_jsonl(out_data / "valid_probe.jsonl", valid_rows)
  (out_data / "split_stats.json").write_text(
    json.dumps(split_stats, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )

  hunk_rows = load_jsonl(Path(args.hunk_train_file))
  cfg = SentSeqTrainConfig(
    sentence_model_name=args.model,
    truncate_dim=args.truncate_dim or None,
    text_prefix=args.text_prefix,
    max_seq_length=args.max_seq_length,
    encode_batch_size=args.encode_batch_size,
    d_model=args.d_model,
    num_layers=args.num_layers,
    max_sents=args.max_sents,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    seed=args.seed,
  )
  loss_cfg = IntervalLossConfig()
  train_interval_model(
    train_rows,
    valid_rows,
    hunk_rows,
    cfg,
    loss_cfg,
    device=device,
    output_dir=output_dir,
  )


if __name__ == "__main__":
  main()
