#!/usr/bin/env python3
"""層活性に対する leave-one-project-out 線形プローブ（フェーズ A）。

各層で 2 つの読み取りを評価し、cross-project 精度の層プロファイルを出す。

- single: 「このベクトルは下書き側か推敲後側か」（mean-difference 射影）
- pair  : 差分ベクトル h(推敲後) - h(下書き) の符号当て（内容成分が打ち消える）

原稿本文は不要（activations.npz の活性と project_id のみ）。
4096 次元のフル LR は実測で数時間規模になるため使わない
（PCA128+LR でも mean-diff 射影を超えないことを確認済み）。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def normalized(v: np.ndarray) -> np.ndarray:
  n = float(np.linalg.norm(v))
  if n < 1e-12:
    return np.zeros_like(v)
  return v / n


def single_text_lopo(
  draft: np.ndarray,
  revised: np.ndarray,
  layer: int,
  idx_by_project: dict[str, list[int]],
) -> dict[str, float]:
  """単独ベクトル分類（mean-diff 方向＋中点しきい値）の LOPO。"""
  d = draft[:, layer, :]
  r = revised[:, layer, :]
  correct = total = 0
  fold_accs = []
  for held, ev_list in idx_by_project.items():
    ev = np.asarray(ev_list)
    tr = np.ones(len(d), dtype=bool)
    tr[ev] = False
    v = normalized(r[tr].mean(axis=0) - d[tr].mean(axis=0))
    thr = 0.5 * (float((d[tr] @ v).mean()) + float((r[tr] @ v).mean()))
    preds_d = (d[ev] @ v) < thr
    preds_r = (r[ev] @ v) >= thr
    c = int(preds_d.sum()) + int(preds_r.sum())
    fold_accs.append(c / (2 * len(ev)))
    correct += c
    total += 2 * len(ev)
  return {"micro": correct / total, "macro": float(np.mean(fold_accs))}


def pair_diff_lopo(
  draft: np.ndarray,
  revised: np.ndarray,
  layer: int,
  idx_by_project: dict[str, list[int]],
) -> dict[str, float]:
  """差分ベクトルの符号当て（mean-diff 方向への射影 > 0）の LOPO。"""
  diff = revised[:, layer, :] - draft[:, layer, :]
  correct = total = 0
  fold_accs = []
  for held, ev_list in idx_by_project.items():
    ev = np.asarray(ev_list)
    tr = np.ones(len(diff), dtype=bool)
    tr[ev] = False
    v = normalized(diff[tr].mean(axis=0))
    c = int(((diff[ev] @ v) > 0).sum())
    fold_accs.append(c / len(ev))
    correct += c
    total += len(ev)
  return {"micro": correct / total, "macro": float(np.mean(fold_accs))}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--activations", default="", help="activations.npz from extract script")
  parser.add_argument(
    "--model",
    default="",
    help="HF model id（--activations 省略時、activations-dir/<slug>[--variant]/activations.npz を探す）",
  )
  parser.add_argument(
    "--variant",
    default="",
    help="抽出時の prompt-mode（reading / norms）。空なら素の <slug> ディレクトリ",
  )
  parser.add_argument(
    "--activations-dir",
    default="outputs/steering",
    help="steering 出力の親ディレクトリ",
  )
  parser.add_argument("--report", default="", help="JSON report path")
  parser.add_argument("--markdown", default="", help="Markdown summary path")
  args = parser.parse_args()

  from steering_utils import model_slug

  if args.activations:
    act_path = Path(args.activations)
  elif args.model:
    name = model_slug(args.model)
    if args.variant:
      name = f"{name}--{args.variant}"
    act_path = Path(args.activations_dir) / name / "activations.npz"
  else:
    raise SystemExit("--activations または --model が必要です")
  if not act_path.is_file():
    raise SystemExit(f"missing activations: {act_path}")

  print(f"loading {act_path}", flush=True)
  data = np.load(act_path, allow_pickle=True)
  draft = np.asarray(data["draft"], dtype=np.float32)
  revised = np.asarray(data["revised"], dtype=np.float32)
  project_ids = np.asarray(data["project_id"], dtype=object)
  print(
    f"loaded pairs={draft.shape[0]} layers={draft.shape[1]} dim={draft.shape[2]}",
    flush=True,
  )
  if draft.shape != revised.shape:
    raise SystemExit(f"shape mismatch: draft={draft.shape} revised={revised.shape}")
  if len(project_ids) != draft.shape[0]:
    raise SystemExit("project_id length mismatch")

  n_pairs, n_layers, _dim = draft.shape
  idx_by_project: dict[str, list[int]] = defaultdict(list)
  for i, pid in enumerate(project_ids):
    idx_by_project[str(pid)].append(i)
  projects = sorted(idx_by_project)
  if len(projects) < 2:
    raise SystemExit("need at least 2 projects for leave-one-project-out")

  layer_rows = []
  for layer in range(n_layers):
    single = single_text_lopo(draft, revised, layer, idx_by_project)
    pair = pair_diff_lopo(draft, revised, layer, idx_by_project)
    layer_rows.append(
      {
        "layer": layer,
        "label": "embed" if layer == 0 else f"block_{layer - 1}",
        "single_micro": single["micro"],
        "single_macro": single["macro"],
        "pair_micro": pair["micro"],
        "pair_macro": pair["macro"],
      }
    )
    print(
      f"layer {layer:02d} ({layer_rows[-1]['label']}): "
      f"single={single['micro']:.4f} pair={pair['micro']:.4f}",
      flush=True,
    )

  best_single = max(layer_rows, key=lambda r: r["single_micro"])
  best_pair = max(layer_rows, key=lambda r: r["pair_micro"])
  report = {
    "activations": str(act_path),
    "probe_method": "mean_difference_projection (single) / diff-sign (pair)",
    "n_pairs": int(n_pairs),
    "n_layers_including_embed": int(n_layers),
    "n_projects": len(projects),
    "projects": projects,
    "best_single": {k: best_single[k] for k in ("layer", "label", "single_micro", "single_macro")},
    "best_pair": {k: best_pair[k] for k in ("layer", "label", "pair_micro", "pair_macro")},
    "layers": layer_rows,
    "stop_heuristic": {
      "pref_static_ballpark": 0.95,
      "note": "pair の最良 micro がここから大きく劣るならフェーズ B/C は縮小を検討",
    },
  }

  out_dir = act_path.parent
  report_path = Path(args.report) if args.report else out_dir / "probe_report.json"
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  md_path = Path(args.markdown) if args.markdown else out_dir / "probe_report.md"
  lines = [
    "# Activation reading probe (leave-one-project-out)",
    "",
    f"- activations: `{act_path}`",
    "- single: mean-difference projection / pair: diff-sign",
    f"- pairs: {n_pairs}",
    f"- projects: {len(projects)}",
    f"- best single: layer {best_single['layer']} ({best_single['label']}) "
    f"micro={best_single['single_micro']:.4f}",
    f"- best pair: layer {best_pair['layer']} ({best_pair['label']}) "
    f"micro={best_pair['pair_micro']:.4f}",
    "",
    "| layer | label | single micro | single macro | pair micro | pair macro |",
    "|------:|-------|-------------:|-------------:|-----------:|-----------:|",
  ]
  for r in layer_rows:
    lines.append(
      f"| {r['layer']} | {r['label']} | {r['single_micro']:.4f} | {r['single_macro']:.4f} "
      f"| {r['pair_micro']:.4f} | {r['pair_macro']:.4f} |"
    )
  lines.append("")
  md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  print(f"wrote {report_path}")
  print(f"wrote {md_path}")
  print(
    f"best single layer {best_single['layer']}: micro={best_single['single_micro']:.4f} / "
    f"best pair layer {best_pair['layer']}: micro={best_pair['pair_micro']:.4f}"
  )


if __name__ == "__main__":
  main()
