#!/usr/bin/env python3
"""収束判定: 追加推敲が実質改善しないなら『推敲済み』とみなす。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import load

from pref_bt_runtime import load_bt_model, score_candidates_bt
from pref_static_utils import load_sentence_model_from_artifact, static_pair_preference_inference


def read_text(value: str | None, file_path: str | None, *, required: bool = True) -> str:
  if value:
    return value
  if file_path:
    return Path(file_path).read_text(encoding="utf-8")
  if required:
    raise SystemExit("text or file is required")
  return ""


def pair_convergence(
  model_dir: Path,
  current: str,
  revised: str,
  *,
  epsilon: float,
) -> dict:
  """pref-static で P(revised > current) ≈ 0.5 なら収束。"""
  artifact = load(model_dir / "model.joblib")
  sentence_model = load_sentence_model_from_artifact(artifact, device="cpu")
  # source スロットは「比較の基準文脈」。現版を置く。
  score_revised, score_current, detail = static_pair_preference_inference(
    artifact["classifier"],
    sentence_model,
    current,
    revised,
    current,
    normalize_embeddings=bool(artifact["normalize_embeddings"]),
    symmetric=True,
    text_prefix=str(artifact.get("text_prefix", "")),
  )
  # score_revised ≈ P(revised が良い), score_current ≈ P(current が良い)
  # 収束: どちらかが支配的でない（差が小さい）
  margin = score_revised - score_current
  converged = abs(margin) <= epsilon
  return {
    "mode": "pair",
    "converged": converged,
    "epsilon": epsilon,
    "score_revised": score_revised,
    "score_current": score_current,
    "margin": margin,
    "detail": detail,
  }


def bt_convergence(
  model_dir: Path,
  source: str,
  current: str,
  revised: str,
  *,
  min_improvement: float,
) -> dict:
  """BT: s(source, revised) - s(source, current) が閾値未満なら収束。"""
  loaded = load_bt_model(model_dir)
  scores = score_candidates_bt(loaded, source, [current, revised], batch_size=4)
  score_current, score_revised = scores
  improvement = score_revised - score_current
  converged = improvement < min_improvement
  return {
    "mode": "bt",
    "converged": converged,
    "min_improvement": min_improvement,
    "score_current": score_current,
    "score_revised": score_revised,
    "improvement": improvement,
    "source_used": source,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--mode",
    choices=["pair", "bt"],
    default="pair",
    help="pair=pref-static の対称比較 / bt=報酬マージン",
  )
  parser.add_argument("--static-model", help="pref-static model dir (mode=pair)")
  parser.add_argument("--bt-model", help="pref-bt model dir (mode=bt)")
  parser.add_argument("--current-text")
  parser.add_argument("--current-file")
  parser.add_argument("--revised-text")
  parser.add_argument("--revised-file")
  parser.add_argument(
    "--source-text",
    help="BT 用の固定下書き。省略時は current を使う",
  )
  parser.add_argument("--source-file")
  parser.add_argument(
    "--epsilon",
    type=float,
    default=0.08,
    help="pair モード: |P(rev)-P(cur)| がこれ以下なら収束",
  )
  parser.add_argument(
    "--min-improvement",
    type=float,
    default=0.5,
    help="bt モード: 改善幅がこれ未満なら収束",
  )
  parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
  args = parser.parse_args()

  current = read_text(args.current_text, args.current_file)
  revised = read_text(args.revised_text, args.revised_file)

  if args.mode == "pair":
    model_dir = Path(args.static_model or "")
    if not model_dir.is_dir():
      raise SystemExit("--static-model is required for mode=pair")
    result = pair_convergence(model_dir, current, revised, epsilon=args.epsilon)
  else:
    model_dir = Path(args.bt_model or "")
    if not model_dir.is_dir():
      raise SystemExit("--bt-model is required for mode=bt")
    source = read_text(args.source_text, args.source_file, required=False) or current
    result = bt_convergence(
      model_dir,
      source,
      current,
      revised,
      min_improvement=args.min_improvement,
    )

  if args.format == "json":
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return

  if args.format == "markdown":
    state = "converged" if result["converged"] else "continue"
    print(f"**Convergence**: `{state}` (mode={result['mode']})")
    if result["mode"] == "pair":
      print(
        f"- P(revised): `{result['score_revised']:.4f}` / "
        f"P(current): `{result['score_current']:.4f}` / "
        f"margin: `{result['margin']:.4f}` (ε={result['epsilon']})"
      )
    else:
      print(
        f"- s(revised): `{result['score_revised']:.4f}` / "
        f"s(current): `{result['score_current']:.4f}` / "
        f"improvement: `{result['improvement']:.4f}` "
        f"(min={result['min_improvement']})"
      )
    return

  print(f"converged: {str(result['converged']).lower()}")
  print(f"mode: {result['mode']}")
  if result["mode"] == "pair":
    print(f"score_revised: {result['score_revised']:.6f}")
    print(f"score_current: {result['score_current']:.6f}")
    print(f"margin: {result['margin']:.6f}")
    print(f"epsilon: {result['epsilon']}")
  else:
    print(f"score_revised: {result['score_revised']:.6f}")
    print(f"score_current: {result['score_current']:.6f}")
    print(f"improvement: {result['improvement']:.6f}")
    print(f"min_improvement: {result['min_improvement']}")


if __name__ == "__main__":
  main()
