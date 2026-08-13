#!/usr/bin/env python3
"""生成文人手選好（tie 含む）で section sentseq 評価器を学習する実験入口。

既存 outputs/pref-sentseq-keep* は上書きしない。
legacy keep anchor と生成教師は別 loader で読み、valid へ anchor を混ぜない。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from generated_pref_utils import (
  augment_generated_side_swap,
  build_sentseq_model,
  evaluate_generated_preferences,
  fit_tie_threshold,
  legacy_row_to_training,
  load_generated_pref_rows,
  load_legacy_anchor_rows,
  preference_loss,
  score_from_doc_vectors_with_length_toggle,
  score_pair_deltas,
)
from pref_static_utils import load_jsonl, normalize_truncate_dim
from train_pref_sentseq import (
  SentSeqTrainConfig,
  build_model_config,
  collect_unique_texts,
  encode_unique_sentences,
  prepare_sentence_data,
  resolve_device,
  save_sentseq_model,
)


def build_training_pool(
  generated_rows: list[dict],
  anchor_rows: list[dict],
  *,
  side_swap_augment: bool,
) -> list[dict]:
  pool = list(generated_rows)
  if side_swap_augment:
    pool = augment_generated_side_swap(pool)
  for row in anchor_rows:
    pool.append(legacy_row_to_training(row))
  return pool


def sample_batch_indices(
  pool: list[dict],
  generated_indices: list[int],
  anchor_indices: list[int],
  *,
  batch_size: int,
  anchor_batch_fraction: float,
  rng: random.Random,
) -> list[int]:
  if not anchor_indices or anchor_batch_fraction <= 0:
    return rng.sample(range(len(pool)), k=min(batch_size, len(pool)))
  n_anchor = min(
    len(anchor_indices),
    max(0, int(round(batch_size * anchor_batch_fraction))),
  )
  n_generated = min(len(generated_indices), batch_size - n_anchor)
  if n_generated <= 0:
    n_generated = min(batch_size, len(generated_indices))
    n_anchor = min(batch_size - n_generated, len(anchor_indices))
  chosen = (
    rng.sample(anchor_indices, k=n_anchor)
    + rng.sample(generated_indices, k=n_generated)
  )
  if len(chosen) < batch_size:
    rest = [i for i in range(len(pool)) if i not in chosen]
    if rest:
      chosen.extend(rng.sample(rest, k=min(batch_size - len(chosen), len(rest))))
  rng.shuffle(chosen)
  return chosen


def train_generated_pref_sentseq(
  train_rows: list[dict],
  valid_rows: list[dict],
  cfg: SentSeqTrainConfig,
  *,
  device: torch.device,
  use_length_features: bool,
  anchor_rows: list[dict],
  anchor_batch_fraction: float,
  side_swap_augment: bool,
  log_prefix: str = "",
) -> tuple[torch.nn.Module, dict, dict]:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  generated_train = [r for r in train_rows if r.get("schema") != "legacy_anchor_v1"]
  pool = build_training_pool(
    generated_train,
    anchor_rows,
    side_swap_augment=side_swap_augment,
  )
  generated_indices = [i for i, r in enumerate(pool) if r.get("schema") != "legacy_anchor_v1"]
  anchor_indices = [i for i, r in enumerate(pool) if r.get("schema") == "legacy_anchor_v1"]

  sent_to_embedding = encode_unique_sentences(
    pool + valid_rows,
    cfg,
    device=device,
    show_progress_bar=bool(log_prefix),
  )
  embed_dim = next(iter(sent_to_embedding.values())).shape[0]
  all_texts = collect_unique_texts(pool + valid_rows)
  prepared = prepare_sentence_data(
    all_texts,
    sent_to_embedding=sent_to_embedding,
    max_sents=cfg.max_sents,
    device=device,
  )

  model = build_sentseq_model(
    embed_dim=embed_dim,
    d_model=cfg.d_model,
    use_length_features=use_length_features,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
    max_sents=cfg.max_sents,
  ).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
  )

  rng = random.Random(cfg.seed)
  best_valid: dict[str, float] = {}
  best_state = None
  best_epoch = -1
  best_tie_threshold = 0.0

  print(
    f"{log_prefix}train pool={len(pool)} generated={len(generated_train)} "
    f"anchor={len(anchor_rows)} valid={len(valid_rows)} "
    f"length_features={use_length_features}",
    flush=True,
  )

  for epoch in range(cfg.epochs):
    model.train()
    # 1 epoch = floor(len(pool)/batch_size) 更新
    n_steps = max(1, len(pool) // max(cfg.batch_size, 1))
    epoch_loss = 0.0
    for _step in range(n_steps):
      batch_idx = sample_batch_indices(
        pool,
        generated_indices,
        anchor_indices,
        batch_size=cfg.batch_size,
        anchor_batch_fraction=anchor_batch_fraction,
        rng=rng,
      )
      batch = [pool[i] for i in batch_idx]
      sources = [r["source_text"] for r in batch]
      cand_a = [r["candidate_a"] for r in batch]
      cand_b = [r["candidate_b"] for r in batch]
      from train_pref_sentseq import encode_texts_to_doc_vectors

      v_src = encode_texts_to_doc_vectors(model, sources, prepared, device=device)
      v_a = encode_texts_to_doc_vectors(model, cand_a, prepared, device=device)
      v_b = encode_texts_to_doc_vectors(model, cand_b, prepared, device=device)
      len_s = torch.tensor([float(len(t)) for t in sources], dtype=torch.float32, device=device)
      len_a = torch.tensor([float(len(t)) for t in cand_a], dtype=torch.float32, device=device)
      len_b = torch.tensor([float(len(t)) for t in cand_b], dtype=torch.float32, device=device)
      s_a = score_from_doc_vectors_with_length_toggle(
        model, v_src, v_a, len_source=len_s, len_candidate=len_a, use_length_features=use_length_features
      )
      s_b = score_from_doc_vectors_with_length_toggle(
        model, v_src, v_b, len_source=len_s, len_candidate=len_b, use_length_features=use_length_features
      )
      delta = s_a - s_b
      losses = [preference_loss(delta[i : i + 1], str(batch[i]["preference"])) for i in range(len(batch))]
      loss = torch.stack(losses).mean()

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      epoch_loss += float(loss.item())

    train_deltas = score_pair_deltas(
      model,
      generated_train,
      prepared,
      device=device,
      batch_size=cfg.batch_size,
      use_length_features=use_length_features,
    )
    train_prefs = [str(r["preference"]).lower() for r in generated_train]
    tie_threshold = fit_tie_threshold(train_deltas, train_prefs)
    valid_deltas = score_pair_deltas(
      model,
      valid_rows,
      prepared,
      device=device,
      batch_size=cfg.batch_size,
      use_length_features=use_length_features,
    )
    valid_prefs = [str(r["preference"]).lower() for r in valid_rows]
    valid_metrics = evaluate_generated_preferences(
      valid_deltas,
      valid_prefs,
      tie_threshold=tie_threshold,
    )
    print(
      f"{log_prefix}epoch {epoch + 1}/{cfg.epochs} "
      f"loss={epoch_loss / n_steps:.4f} "
      f"valid_three_way_macro_recall={valid_metrics['three_way_macro_recall']:.4f} "
      f"valid_three_way_acc={valid_metrics['three_way_accuracy']:.4f} "
      f"tie_n={valid_metrics['tie_n']}",
      flush=True,
    )
    score = float(valid_metrics["three_way_macro_recall"])
    if score >= best_valid.get("valid_three_way_macro_recall", -1.0):
      best_valid = {f"valid_{k}": v for k, v in valid_metrics.items() if k != "confusion"}
      best_valid["valid_confusion"] = valid_metrics["confusion"]
      best_epoch = epoch + 1
      best_tie_threshold = tie_threshold
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()

  train_metrics = {
    "best_epoch": float(best_epoch),
    "train_generated_rows": len(generated_train),
    "train_pool_rows": len(pool),
    "anchor_rows": len(anchor_rows),
    "anchor_batch_fraction": anchor_batch_fraction,
    "tie_threshold": best_tie_threshold,
    "length_features": use_length_features,
  }
  return model, train_metrics, best_valid


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-file", required=True)
  parser.add_argument("--valid-file", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--anchor-file", default="data/pref_keep_split_section/train.jsonl")
  parser.add_argument("--anchor-batch-fraction", type=float, default=0.25)
  parser.add_argument("--no-side-swap-augment", action="store_true")
  parser.add_argument("--length-features", action="store_true", help="文字数特徴を使う（既定 off）")
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--truncate-dim", type=int, default=0)
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--encode-batch-size", type=int, default=64)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--num-layers", type=int, default=2)
  parser.add_argument("--max-sents", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  parser.add_argument("--limit-train", type=int, default=0)
  parser.add_argument("--limit-valid", type=int, default=0)
  args = parser.parse_args()

  device = resolve_device(args.device)
  train_rows, train_stats = load_generated_pref_rows(args.train_file, unit="section")
  valid_rows, valid_stats = load_generated_pref_rows(args.valid_file, unit="section")
  if train_stats["excluded"] or valid_stats["excluded"]:
    print(
      f"excluded non-section rows: train={train_stats['excluded']} "
      f"valid={valid_stats['excluded']}",
      flush=True,
    )
  if not train_rows or not valid_rows:
    raise SystemExit("section rows are empty after filtering")

  if args.limit_train > 0:
    train_rows = train_rows[: args.limit_train]
  if args.limit_valid > 0:
    valid_rows = valid_rows[: args.limit_valid]

  anchor_rows: list[dict] = []
  anchor_path = Path(args.anchor_file)
  if anchor_path.is_file() and args.anchor_batch_fraction > 0:
    anchor_rows = load_legacy_anchor_rows(anchor_path, unit="section")

  cfg = SentSeqTrainConfig(
    sentence_model_name=args.model,
    truncate_dim=normalize_truncate_dim(args.truncate_dim),
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

  model, train_metrics, valid_metrics = train_generated_pref_sentseq(
    train_rows,
    valid_rows,
    cfg,
    device=device,
    use_length_features=args.length_features,
    anchor_rows=anchor_rows,
    anchor_batch_fraction=args.anchor_batch_fraction,
    side_swap_augment=not args.no_side_swap_augment,
  )

  embed_dim = model.encoder.embed_dim
  feature_dim = model.head.linear.in_features
  output_dir = Path(args.output_dir)
  save_sentseq_model(model, cfg, embed_dim=embed_dim, output_dir=output_dir)
  # save_sentseq_model の config は length 5 固定なので、実験設定を上書きする
  artifact_path = output_dir / "model.pt"
  artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
  config = build_model_config(cfg, embed_dim=embed_dim, feature_dim=feature_dim)
  config["length_features"] = args.length_features
  config["experiment"] = "generated_pref_sentseq_v1"
  artifact["config"] = config
  torch.save(artifact, artifact_path)
  (output_dir / "model_config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )

  metrics = {
    "train_file": args.train_file,
    "valid_file": args.valid_file,
    "anchor_file": str(anchor_path) if anchor_path.is_file() else None,
    "train_rows": len(train_rows),
    "valid_rows": len(valid_rows),
    **train_metrics,
    **valid_metrics,
  }
  (output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"saved: {output_dir}")
  print(
    f"valid_three_way_macro_recall: "
    f"{valid_metrics.get('valid_three_way_macro_recall', 0.0):.4f}"
  )


if __name__ == "__main__":
  main()
