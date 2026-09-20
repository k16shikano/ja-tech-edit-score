#!/usr/bin/env python3
"""Phase 3 タスク 3-2/3-3: 水準関門 G2（累積リンク）。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import (
  POSITION_RANK,
  ead_out,
  ead_reports,
  ead_work,
  fmt_pct,
  fmt_rate,
  load_jsonl,
  md_table,
  repo_root,
  rps_ordinal,
  surface_features,
  write_json,
  write_jsonl,
)
from ead.den_score import assistant_token_ids, load_density_model, template_text
from section_middle_utils import build_revision_prompt

try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from sklearn.metrics import precision_recall_fscore_support
  from sklearn.model_selection import GroupKFold
  from sklearn.preprocessing import StandardScaler
except ImportError as exc:
  raise SystemExit(f"torch/sklearn required: {exc}") from exc


HIDDEN_MLP = 256
N_THRESHOLDS = 3
NORM_CHOICES = ("s_sum", "s_mean", "s_resid")
INPUT_MODES = ("legacy", "r7", "no_s_den", "s_den_only", "s_den_surface")


def decile_buckets(deltas: list[float]) -> list[int]:
  if not deltas:
    return []
  arr = np.asarray(deltas, dtype=np.float64)
  edges = np.quantile(arr, np.linspace(0, 1, 11))
  edges = np.unique(edges)
  if len(edges) <= 2:
    return [0] * len(deltas)
  return list(np.digitize(arr, edges[1:-1], right=True))


class OrdinalGate(nn.Module):
  def __init__(self, input_dim: int, hidden: int = HIDDEN_MLP) -> None:
    super().__init__()
    if hidden <= 0:
      self.backbone = nn.Linear(input_dim, 1, bias=False)
    else:
      self.backbone = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, 1, bias=False),
      )
    self.b = nn.Parameter(torch.zeros(N_THRESHOLDS))

  def thresholds(self) -> torch.Tensor:
    t1 = self.b[0]
    t2 = t1 + F.softplus(self.b[1])
    t3 = t2 + F.softplus(self.b[2])
    return torch.stack([t1, t2, t3])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.backbone(x).reshape(-1)

  def category_probs(self, g: torch.Tensor) -> torch.Tensor:
    g = g.reshape(-1)
    thetas = self.thresholds()
    cdf = torch.stack([torch.sigmoid(t - g) for t in thetas], dim=-1)
    p0 = cdf[:, 0]
    p1 = cdf[:, 1] - cdf[:, 0]
    p2 = cdf[:, 2] - cdf[:, 1]
    p3 = 1.0 - cdf[:, 2]
    pk = torch.stack([p0, p1, p2, p3], dim=-1)
    pk = torch.clamp(pk, 1e-6, 1.0)
    return pk / pk.sum(dim=-1, keepdim=True)


def ordinal_nll(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  idx = torch.arange(probs.size(0), device=probs.device)
  logp = torch.log(probs[idx, y])
  return -logp.mean()


def reliability_bins(probs_ge_c: np.ndarray, y_ge_c: np.ndarray, *, n_bins: int = 5) -> list[dict[str, float]]:
  edges = np.linspace(0.0, 1.0, n_bins + 1)
  out: list[dict[str, float]] = []
  for b in range(n_bins):
    lo, hi = edges[b], edges[b + 1]
    if b == n_bins - 1:
      mask = (probs_ge_c >= lo) & (probs_ge_c <= hi)
    else:
      mask = (probs_ge_c >= lo) & (probs_ge_c < hi)
    if mask.sum() == 0:
      continue
    out.append(
      {
        "bin_lo": float(lo),
        "bin_hi": float(hi),
        "n": int(mask.sum()),
        "pred_mean": float(probs_ge_c[mask].mean()),
        "obs_rate": float(y_ge_c[mask].mean()),
      }
    )
  return out


def theta_relative_report(g: np.ndarray, model: OrdinalGate) -> dict[str, float]:
  with torch.no_grad():
    thetas = model.thresholds().cpu().numpy()
  g_mean = float(np.mean(g))
  g_med = float(np.median(g))
  t2 = float(thetas[1])
  pct = float(np.mean(g <= t2) * 100.0)
  return {
    "mean_g": g_mean,
    "median_g": g_med,
    "theta2_minus_mean_g": t2 - g_mean,
    "theta2_minus_median_g": t2 - g_med,
    "theta2_percentile_of_g": pct,
    "max_abs_theta_minus_mean_g": float(np.max(np.abs(thetas - g_mean))),
  }


def d_row_to_prompt(row: dict) -> tuple[str, str, str]:
  draft = str(row["draft"])
  cand = str(row["y"])
  user = build_revision_prompt(draft)
  return user, draft, cand


def extract_hidden_and_logps(
  model,
  tokenizer,
  device,
  *,
  user_content: str,
  draft: str,
  candidate: str,
  max_seq_len: int,
) -> dict[str, Any] | None:
  ans_ids = assistant_token_ids(tokenizer, user_content, candidate)
  if not ans_ids:
    return None
  prompt_text = template_text(
    tokenizer,
    [{"role": "user", "content": user_content}],
    add_generation_prompt=True,
  )
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  seq = prompt_ids + ans_ids
  if len(seq) > max_seq_len:
    overflow = len(seq) - max_seq_len
    if overflow >= len(ans_ids):
      return None
    prompt_ids = prompt_ids[-(max_seq_len - len(ans_ids)) :]
    seq = prompt_ids + ans_ids

  input_ids = torch.tensor([seq], device=device)
  n_ans = len(ans_ids)
  start = input_ids.size(1) - n_ans

  with torch.no_grad():
    with model.disable_adapter():
      out_base = model(input_ids, output_hidden_states=True)
      logp_base = _token_logprob_sum(out_base.logits, input_ids, start, n_ans)
    out_ad = model(input_ids, output_hidden_states=True)
    logp_ad = _token_logprob_sum(out_ad.logits, input_ids, start, n_ans)
    hidden = out_ad.hidden_states[-1][0, start:, :].mean(dim=0)

  if logp_base is None or logp_ad is None:
    return None
  s_sum = float(logp_ad - logp_base)
  return {
    "hidden": hidden.detach().cpu().float().numpy(),
    "s_sum": s_sum,
    "s_mean": s_sum / n_ans,
    "n_tokens": n_ans,
    "n_chars_draft": len(draft),
    "n_chars_cand": len(candidate),
    "delta_chars": len(candidate) - len(draft),
  }


def _token_logprob_sum(logits, input_ids, start: int, n_ans: int) -> float | None:
  if n_ans <= 0:
    return None
  target = input_ids[:, start:]
  pred = logits[:, start - 1 : -1, :]
  lp = F.log_softmax(pred, dim=-1)
  tok_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
  return float(tok_lp.sum().item())


def fit_resid_from_features(rows: list[dict]) -> tuple[float, float, float]:
  xs: list[list[float]] = []
  ys: list[float] = []
  for row in rows:
    xs.append([abs(row["n_chars_cand"]), float(row["delta_chars"])])
    ys.append(float(row["s_sum"]))
  if len(xs) < 3:
    return 0.0, 0.0, 0.0
  x_arr = np.asarray(xs, dtype=np.float64)
  y_arr = np.asarray(ys, dtype=np.float64)
  x1 = np.column_stack([np.ones(len(xs)), x_arr])
  coef, _, _, _ = np.linalg.lstsq(x1, y_arr, rcond=None)
  return float(coef[0]), float(coef[1]), float(coef[2])


def apply_resid(row: dict, intercept: float, c_abs: float, c_delta: float) -> float:
  pred = intercept + c_abs * abs(row["n_chars_cand"]) + c_delta * float(row["delta_chars"])
  return float(row["s_sum"]) - pred


def extract_d_features(
  rows: list[dict],
  model,
  tokenizer,
  device,
  *,
  max_seq_len: int,
) -> list[dict]:
  out: list[dict] = []
  for row in rows:
    user, draft, cand = d_row_to_prompt(row)
    feat = extract_hidden_and_logps(
      model,
      tokenizer,
      device,
      user_content=user,
      draft=draft,
      candidate=cand,
      max_seq_len=max_seq_len,
    )
    if feat is None:
      continue
    out.append(
      {
        "row_id": row.get("row_id"),
        "item_id": row.get("item_id"),
        "position": row.get("position"),
        "label": POSITION_RANK[str(row["position"])],
        "split": row.get("_split"),
        **feat,
      }
    )
  return out


def attach_surface_features(feature_rows: list[dict], raw_rows: list[dict]) -> None:
  raw_by_id = {r["row_id"]: r for r in raw_rows}
  for row in feature_rows:
    raw = raw_by_id.get(row.get("row_id"))
    if raw is None:
      row["edit_distance_norm"] = 0.0
      row["punct_density_diff"] = 0.0
      continue
    feats = surface_features(str(raw["draft"]), str(raw["y"]))
    row["edit_distance_norm"] = feats["edit_distance_norm"]
    row["punct_density_diff"] = feats["punct_density_diff"]


def build_feature_vector(row: dict, *, norm: str, input_mode: str) -> np.ndarray | None:
  parts: list[np.ndarray] = []
  if input_mode in ("legacy", "r7", "no_s_den"):
    hidden = row.get("hidden")
    if hidden is None:
      return None
    parts.append(hidden)
  if input_mode in ("legacy", "r7", "s_den_only", "s_den_surface"):
    if norm == "s_sum":
      scalar = float(row["s_sum"])
    elif norm == "s_mean":
      scalar = float(row["s_mean"])
    else:
      scalar = float(row["s_resid"])
    parts.append(np.array([scalar], dtype=np.float32))
  if input_mode in ("r7", "no_s_den", "s_den_surface"):
    parts.append(
      np.array(
        [float(row.get("edit_distance_norm", 0.0)), float(row.get("punct_density_diff", 0.0))],
        dtype=np.float32,
      )
    )
  if not parts:
    return None
  return np.concatenate(parts)


def feature_matrix(
  rows: list[dict],
  *,
  norm: str,
  input_mode: str = "legacy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
  xs: list[np.ndarray] = []
  ys: list[int] = []
  groups: list[str] = []
  deltas: list[float] = []
  for row in rows:
    vec = build_feature_vector(row, norm=norm, input_mode=input_mode)
    if vec is None:
      continue
    xs.append(vec)
    ys.append(int(row["label"]))
    groups.append(str(row["item_id"]))
    deltas.append(float(row["delta_chars"]))
  if not xs:
    raise SystemExit(f"no feature rows for input_mode={input_mode}")
  return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(groups), deltas


def train_one(
  x_train: np.ndarray,
  y_train: np.ndarray,
  *,
  seed: int,
  epochs: int,
  lr: float,
  device: str,
  mlp_hidden: int = HIDDEN_MLP,
  scaler: StandardScaler | None = None,
) -> tuple[OrdinalGate, StandardScaler, list[dict[str, float]]]:
  torch.manual_seed(seed)
  if scaler is None:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
  else:
    x_train = scaler.transform(x_train)
  model = OrdinalGate(x_train.shape[1], hidden=mlp_hidden).to(device)
  opt = torch.optim.Adam(
    [
      {"params": model.backbone.parameters(), "weight_decay": 0.01},
      {"params": [model.b], "weight_decay": 0.0},
    ],
    lr=lr,
  )
  x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
  y_t = torch.tensor(y_train, dtype=torch.long, device=device)
  logs: list[dict[str, float]] = []
  for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    g = model(x_t)
    probs = model.category_probs(g)
    loss = ordinal_nll(probs, y_t)
    loss.backward()
    opt.step()
    with torch.no_grad():
      rel = theta_relative_report(g.detach().cpu().numpy(), model)
    logs.append({"epoch": epoch, "loss": float(loss.item()), **rel})
  return model, scaler, logs


def predict_rows(
  model: OrdinalGate,
  x: np.ndarray,
  *,
  device: str,
  scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
  model.eval()
  xs = scaler.transform(x)
  with torch.no_grad():
    x_t = torch.tensor(xs, dtype=torch.float32, device=device)
    g_t = model(x_t)
    probs = model.category_probs(g_t).cpu().numpy()
    g = g_t.cpu().numpy()
  return g, probs


def eval_predictions(
  y_true: np.ndarray,
  probs: np.ndarray,
  g: np.ndarray,
  deltas: list[float],
  *,
  floor_recall_a: float,
  floor_rps: float,
) -> dict[str, Any]:
  pred = probs.argmax(axis=1)
  prec, rec, _, _ = precision_recall_fscore_support(
    y_true == 0, pred == 0, labels=[True], average="binary", zero_division=0
  )
  p_ge_c = probs[:, 2] + probs[:, 3]
  y_ge_c = (y_true >= 2).astype(np.float64)
  deciles = decile_buckets(deltas)
  strata: dict[str, Any] = {}
  for d in sorted(set(deciles)):
    mask = np.array(deciles) == d
    if mask.sum() == 0:
      continue
    strata[str(d)] = {
      "n": int(mask.sum()),
      "rps": rps_ordinal(y_true[mask], probs[mask]),
      "label_a_recall": float(
        precision_recall_fscore_support(
          y_true[mask] == 0,
          pred[mask] == 0,
          labels=[True],
          average="binary",
          zero_division=0,
        )[1]
      ),
      "reliability": reliability_bins(p_ge_c[mask], y_ge_c[mask]),
    }
  return {
    "n": int(len(y_true)),
    "rps": rps_ordinal(y_true, probs),
    "accuracy_4": float(np.mean(pred == y_true)),
    "label_a_precision": float(prec),
    "label_a_recall": float(rec),
    "floor_rps": floor_rps,
    "floor_label_a_recall": floor_recall_a,
    "rps_beats_floor": bool(rps_ordinal(y_true, probs) < floor_rps),
    "label_a_recall_beats_floor": bool(rec > floor_recall_a),
    "reliability_overall": reliability_bins(p_ge_c, y_ge_c),
    "strata_delta_decile": strata,
    "p_better": p_ge_c.tolist(),
    "pred_class": pred.tolist(),
    "g": g.tolist(),
  }


def run_cv_and_valid(
  train_rows: list[dict],
  valid_rows: list[dict],
  *,
  norm: str,
  input_mode: str = "legacy",
  adapter_subdir: str = "ead-gate-level",
  mlp_hidden: int = HIDDEN_MLP,
  seeds: list[int],
  epochs: int,
  lr: float,
  device: str,
  floor_recall_a: float,
  floor_rps: float,
) -> dict[str, Any]:
  resid_i, c_abs, c_delta = fit_resid_from_features(train_rows)
  for row in train_rows + valid_rows:
    row["s_resid"] = apply_resid(row, resid_i, c_abs, c_delta)

  x_train, y_train, groups, _ = feature_matrix(train_rows, norm=norm, input_mode=input_mode)
  x_valid, y_valid, _, deltas_valid = feature_matrix(valid_rows, norm=norm, input_mode=input_mode)

  seed_metrics: list[dict[str, Any]] = []
  all_valid_preds: list[dict[str, Any]] = []

  gkf = GroupKFold(n_splits=5)
  for seed in seeds:
    cv_preds_probs = np.zeros((len(train_rows), 4), dtype=np.float64)
    cv_mask = np.zeros(len(train_rows), dtype=bool)
    fold_logs: list[dict[str, Any]] = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x_train, y_train, groups)):
      model, scaler, logs = train_one(
        x_train[tr_idx],
        y_train[tr_idx],
        seed=seed + fold,
        epochs=epochs,
        lr=lr,
        device=device,
        mlp_hidden=mlp_hidden,
      )
      _, probs_va = predict_rows(model, x_train[va_idx], device=device, scaler=scaler)
      cv_preds_probs[va_idx] = probs_va
      cv_mask[va_idx] = True
      fold_logs.append({"fold": fold, "last_epoch": logs[-1] if logs else {}})

    oof = eval_predictions(
      y_train[cv_mask],
      cv_preds_probs[cv_mask],
      np.zeros(int(cv_mask.sum())),
      [train_rows[i]["delta_chars"] for i, ok in enumerate(cv_mask.tolist()) if ok],
      floor_recall_a=floor_recall_a,
      floor_rps=floor_rps,
    )

    final_model, final_scaler, final_logs = train_one(
      x_train, y_train, seed=seed, epochs=epochs, lr=lr, device=device, mlp_hidden=mlp_hidden
    )
    g_valid, probs_valid = predict_rows(final_model, x_valid, device=device, scaler=final_scaler)
    valid_metrics = eval_predictions(
      y_valid,
      probs_valid,
      g_valid,
      deltas_valid,
      floor_recall_a=floor_recall_a,
      floor_rps=floor_rps,
    )
    valid_metrics["theta_relative"] = theta_relative_report(g_valid, final_model)
    valid_metrics["training_last_epoch"] = final_logs[-1] if final_logs else {}
    valid_metrics["cv_oof"] = oof
    valid_metrics["seed"] = seed
    seed_metrics.append(valid_metrics)

    ckpt_dir = ead_out() / "adapters" / adapter_subdir / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "state_dict": final_model.state_dict(),
        "input_dim": x_train.shape[1],
        "norm": norm,
        "input_mode": input_mode,
        "mlp_hidden": mlp_hidden,
        "resid": {"intercept": resid_i, "coef_abs_y": c_abs, "coef_delta": c_delta},
        "scaler_mean": final_scaler.mean_.tolist(),
        "scaler_scale": final_scaler.scale_.tolist(),
        "seed": seed,
      },
      ckpt_dir / "model.pt",
    )

    for row, prob, g_val, pred in zip(valid_rows, probs_valid, g_valid, probs_valid.argmax(axis=1)):
      all_valid_preds.append(
        {
          "row_id": row.get("row_id"),
          "item_id": row.get("item_id"),
          "position": row.get("position"),
          "label": row.get("label"),
          "pred_class": int(pred),
          "p_better": float(prob[2] + prob[3]),
          "g": float(g_val),
          "seed": seed,
          "norm": norm,
          "probs": [float(x) for x in prob],
        }
      )

  def avg(key: str) -> float:
    vals = [m[key] for m in seed_metrics if m.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")

  return {
    "norm": norm,
    "input_mode": input_mode,
    "mlp_hidden": mlp_hidden,
    "seeds": seeds,
    "seed_metrics": seed_metrics,
    "aggregate_valid": {
      "rps_mean": avg("rps"),
      "label_a_recall_mean": avg("label_a_recall"),
      "accuracy_4_mean": avg("accuracy_4"),
    },
    "predictions_valid": all_valid_preds,
  }


def learning_curve(
  train_rows: list[dict],
  valid_rows: list[dict],
  *,
  norm: str,
  input_mode: str = "legacy",
  seed: int,
  train_device: str,
  sizes: list[int],
) -> list[dict[str, Any]]:
  draft_ids = sorted({str(r["item_id"]) for r in train_rows})
  out: list[dict[str, Any]] = []
  floor = load_floor_metrics()
  for n_drafts in sizes:
    use_ids = set(draft_ids[:n_drafts])
    subset = [r for r in train_rows if str(r["item_id"]) in use_ids]
    if len({r["item_id"] for r in subset}) < n_drafts:
      continue
    res = run_cv_and_valid(
      subset,
      valid_rows,
      norm=norm,
      input_mode=input_mode,
      seeds=[seed],
      epochs=40,
      lr=1e-3,
      device=train_device,
      floor_recall_a=floor["label_a_recall"],
      floor_rps=floor["rps"],
    )
    out.append(
      {
        "n_drafts": n_drafts,
        "n_rows": len(subset),
        "valid_rps": res["seed_metrics"][0]["rps"],
        "valid_label_a_recall": res["seed_metrics"][0]["label_a_recall"],
      }
    )
  return out


def load_floor_metrics() -> dict[str, float]:
  path = ead_reports() / "ead-floor.json"
  obj = json.loads(path.read_text(encoding="utf-8"))
  valid = obj["valid"]
  out = {"rps": float(valid["rps"]), "label_a_recall": float(valid["label_a_recall"])}
  sanity = obj.get("rps_sanity_valid")
  if sanity:
    out["rps_sanity_uniform"] = float(sanity["uniform"])
    out["rps_sanity_marginal"] = float(sanity["marginal_train"])
  return out


def cache_features(
  rows: list[dict],
  *,
  adapter_dir: Path,
  fit_train_jsonl: Path,
  base_model: str,
  device: str,
  max_seq_len: int,
  trust_remote_code: bool,
  cache_path: Path,
) -> list[dict]:
  if cache_path.is_file() and rows:
    obj = np.load(cache_path, allow_pickle=True)
    meta = json.loads(str(obj["meta"]))
    if meta.get("n_rows") == len(rows):
      rebuilt: list[dict] = []
      ok = True
      for i, row in enumerate(rows):
        key = str(i)
        if key not in obj:
          ok = False
          break
        feat = obj[key].item()
        rebuilt.append({**row, **feat, "_split": row.get("_split"), "split": row.get("_split")})
      if ok and len(rebuilt) == len(rows):
        return rebuilt

  if (adapter_dir / "adapter").is_dir():
    adapter_dir = adapter_dir / "adapter"
  model, tokenizer, dev = load_density_model(adapter_dir, base_model, device, trust_remote_code=trust_remote_code)
  features = extract_d_features(rows, model, tokenizer, dev, max_seq_len=max_seq_len)
  for src, dst in zip(rows, features):
    dst["_split"] = src.get("_split")
  save_obj: dict[str, Any] = {"meta": json.dumps({"n_rows": len(rows)})}
  for i, feat in enumerate(features):
    save_obj[str(i)] = np.array(
      {
        k: v
        for k, v in feat.items()
        if k
        in {
          "row_id",
          "item_id",
          "position",
          "label",
          "split",
          "s_sum",
          "s_mean",
          "n_tokens",
          "n_chars_draft",
          "n_chars_cand",
          "delta_chars",
          "hidden",
        }
      },
      dtype=object,
    )
  cache_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(cache_path, **save_obj)
  return features


def build_report(summary: dict[str, Any], floor: dict[str, float], *, report_stem: str = "ead-gate-level") -> str:
  agg = summary["aggregate_valid"]
  sm0 = summary["seed_metrics"][0]
  input_mode = summary.get("input_mode", "legacy")
  lines = [
    f"# {report_stem}",
    "",
    f"入力: `{input_mode}`。正規化: `{summary['norm']}`。床は `ead-floor`（D valid 表層特徴）: RPS {floor['rps']:.3f}, ラベル a 再現率 {floor['label_a_recall']:.3f}。",
    "",
    "RPS 健全性（D valid）: 一様 "
    + fmt_pct(floor.get("rps_sanity_uniform"))
    + "、周辺分布 "
    + fmt_pct(floor.get("rps_sanity_marginal"))
    + "。",
    "",
    "## Valid 集計（3 seed 平均）",
    "",
    md_table(
      ["指標", "値", "床", "床超え"],
      [
        ["RPS", fmt_pct(agg["rps_mean"]), fmt_pct(floor["rps"]), "✓" if agg["rps_mean"] < floor["rps"] else "✗"],
        [
          "ラベル a 再現率",
          fmt_pct(agg["label_a_recall_mean"]),
          fmt_pct(floor["label_a_recall"]),
          "✓" if agg["label_a_recall_mean"] > floor["label_a_recall"] else "✗",
        ],
        ["4値精度", fmt_pct(agg["accuracy_4_mean"]), "—", "—"],
      ],
    ),
    "",
    "## θ の相対位置（seed 0, valid g 分布に対する θ₂）",
    "",
    md_table(
      ["量", "値"],
      [[k, fmt_pct(v) if "percentile" in k else f"{v:.4f}"] for k, v in sm0.get("theta_relative", {}).items()],
    ),
    "",
    "## Reliability P(Y≥c)（seed 0, 全体）",
    "",
    md_table(
      ["bin_lo", "bin_hi", "n", "pred_mean", "obs_rate"],
      [
        [b["bin_lo"], b["bin_hi"], b["n"], fmt_pct(b["pred_mean"]), fmt_pct(b["obs_rate"])]
        for b in sm0.get("reliability_overall", [])
      ],
    ),
    "",
    "## Acceptance (P3 G2)",
    "",
    f"- RPS が床を下回る: {'✓' if agg['rps_mean'] < floor['rps'] else '✗'}",
    f"- ラベル a 再現率が床を上回る: {'✓' if agg['label_a_recall_mean'] > floor['label_a_recall'] else '✗'}",
  ]
  if summary.get("learning_curve"):
    lines.extend(
      [
        "",
        "## 学習曲線（draft 数）",
        "",
        md_table(
          ["drafts", "rows", "valid RPS", "valid ラベル a 再現率"],
          [
            [lc["n_drafts"], lc["n_rows"], fmt_pct(lc["valid_rps"]), fmt_pct(lc["valid_label_a_recall"])]
            for lc in summary["learning_curve"]
          ],
        ),
      ]
    )
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--adapter-dir", type=Path, default=None)
  parser.add_argument("--fit-train-jsonl", type=Path, default=None)
  parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
  parser.add_argument("--device", default="cuda", help="LLM feature extraction device")
  parser.add_argument("--train-device", default="cpu", help="G2 MLP training device")
  parser.add_argument("--max-seq-length", type=int, default=4096)
  parser.add_argument("--trust-remote-code", action="store_true")
  parser.add_argument("--norm", choices=NORM_CHOICES, default="s_sum")
  parser.add_argument("--input-mode", choices=INPUT_MODES, default="legacy")
  parser.add_argument("--report-stem", default="ead-gate-level")
  parser.add_argument("--adapter-subdir", default=None)
  parser.add_argument("--seeds", default="0,1,2")
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--learning-curve", action="store_true")
  parser.add_argument("--skip-learning-curve", action="store_true")
  parser.add_argument("--refresh-features", action="store_true")
  parser.add_argument("--mlp-hidden", type=int, default=HIDDEN_MLP)
  args = parser.parse_args()

  root = args.root.resolve()
  extract_device = args.device
  if extract_device == "cuda" and not torch.cuda.is_available():
    extract_device = "cpu"
  train_device = args.train_device
  if train_device == "cuda" and not torch.cuda.is_available():
    train_device = "cpu"

  adapter_dir = args.adapter_dir or (ead_out() / "adapters" / "ead-den-a-excl")
  fit_path = args.fit_train_jsonl or (ead_work() / "ead-den-a-excl-train.jsonl")
  if not fit_path.is_absolute():
    fit_path = root / fit_path

  train_raw = load_jsonl(root / "data/d/train.jsonl")
  valid_raw = load_jsonl(root / "data/d/valid.jsonl")
  for row in train_raw:
    row["_split"] = "train"
  for row in valid_raw:
    row["_split"] = "valid"
  all_rows = train_raw + valid_raw

  cache_path = ead_work() / "gate_level_features.npz"
  if args.refresh_features and cache_path.is_file():
    cache_path.unlink()
  features = cache_features(
    all_rows,
    adapter_dir=adapter_dir,
    fit_train_jsonl=fit_path,
    base_model=args.base_model,
    device=extract_device,
    max_seq_len=args.max_seq_length,
    trust_remote_code=args.trust_remote_code,
    cache_path=cache_path,
  )

  train_rows = [r for r in features if r.get("split") == "train" or r.get("_split") == "train"]
  valid_rows = [r for r in features if r.get("split") == "valid" or r.get("_split") == "valid"]
  if not train_rows or not valid_rows:
    raise SystemExit(f"empty feature split train={len(train_rows)} valid={len(valid_rows)}")

  if args.input_mode in ("r7", "no_s_den", "s_den_surface"):
    attach_surface_features(train_rows + valid_rows, all_rows)

  floor = load_floor_metrics()
  seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
  adapter_subdir = args.adapter_subdir or args.report_stem
  report_stem = args.report_stem
  norms_to_run = NORM_CHOICES if args.input_mode == "legacy" else (args.norm,)

  norm_results: dict[str, dict[str, Any]] = {}
  for norm in norms_to_run:
    norm_results[norm] = run_cv_and_valid(
      train_rows,
      valid_rows,
      norm=norm,
      input_mode=args.input_mode,
      adapter_subdir=adapter_subdir,
      mlp_hidden=args.mlp_hidden,
      seeds=seeds,
      epochs=args.epochs,
      lr=args.lr,
      device=train_device,
      floor_recall_a=floor["label_a_recall"],
      floor_rps=floor["rps"],
    )

  selected = args.norm
  summary = norm_results[selected]
  need_curve = not args.skip_learning_curve and (
    args.learning_curve
    or (
      args.input_mode == "legacy"
      and not (
        summary["aggregate_valid"]["rps_mean"] < floor["rps"]
        and summary["aggregate_valid"]["label_a_recall_mean"] > floor["label_a_recall"]
      )
    )
  )
  if need_curve:
    summary["learning_curve"] = learning_curve(
      train_rows,
      valid_rows,
      norm=selected,
      input_mode=args.input_mode,
      seed=seeds[0],
      train_device=train_device,
      sizes=[40, 80, 120, 160],
    )

  reports = ead_reports()
  json_path = reports / f"{report_stem}.json"
  md_path = reports / f"{report_stem}.md"
  pred_path = reports / f"{report_stem}-predictions.jsonl"
  write_json(
    json_path,
    {
      "floor": floor,
      "input_mode": args.input_mode,
      "norms": norm_results,
      "selected_norm": selected,
    },
  )
  write_jsonl(pred_path, summary["predictions_valid"])
  md_path.write_text(build_report(summary, floor, report_stem=report_stem), encoding="utf-8")
  print(
    json.dumps(
      {
        "wrote": str(md_path),
        "input_mode": args.input_mode,
        "selected_norm": selected,
        "valid_rps_mean": summary["aggregate_valid"]["rps_mean"],
        "valid_label_a_recall_mean": summary["aggregate_valid"]["label_a_recall_mean"],
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
