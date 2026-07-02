#!/usr/bin/env python3
import argparse
from pathlib import Path

from joblib import load
from sentence_transformers import SentenceTransformer

from pref_static_utils import static_pair_preference_inference


def read_text(value: str | None, file_path: str | None) -> str:
  if value:
    return value
  if file_path:
    return Path(file_path).read_text(encoding="utf-8")
  raise SystemExit("text or file input is required")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, help="trained static preference model directory")
  parser.add_argument(
    "--pref-symmetric-pair",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="rewrite_with_pref_rerank と同じ対称ペア評価。1回だけにするには --no-pref-symmetric-pair",
  )
  parser.add_argument("--source-text", help="source text")
  parser.add_argument("--source-file", help="source text file")
  parser.add_argument("--candidate-a", help="candidate A text")
  parser.add_argument("--candidate-a-file", help="candidate A file")
  parser.add_argument("--candidate-b", help="candidate B text")
  parser.add_argument("--candidate-b-file", help="candidate B file")
  args = parser.parse_args()

  source_text = read_text(args.source_text, args.source_file)
  candidate_a = read_text(args.candidate_a, args.candidate_a_file)
  candidate_b = read_text(args.candidate_b, args.candidate_b_file)

  artifact = load(Path(args.model) / "model.joblib")
  sentence_model = SentenceTransformer(
    artifact["sentence_model_name"],
    device="cpu",
    truncate_dim=artifact["truncate_dim"],
  )

  score_a, score_b, pref_detail = static_pair_preference_inference(
    artifact["classifier"],
    sentence_model,
    source_text,
    candidate_a,
    candidate_b,
    normalize_embeddings=artifact["normalize_embeddings"],
    symmetric=args.pref_symmetric_pair,
  )

  winner = "B" if score_b > score_a else "A"
  print(f"winner: {winner}")
  print(f"score_a: {score_a:.6f}")
  print(f"score_b: {score_b:.6f}")
  if pref_detail:
    pa = pref_detail["prob_a_order_AB"]
    pb = pref_detail["prob_b_order_AB"]
    pbf = pref_detail["prob_first_is_B_order_BAfirst"]
    pba = pref_detail["prob_second_is_A_order_BAfirst"]
    print(
      "[pref-detail] ordered (入力,A,B): "
      f"P(A側スロット良)= {pa:.6f}, P(B側スロット良)= {pb:.6f} | "
      f"ordered (入力,B,A): P(先頭=B良)= {pbf:.6f}, P(末尾=A良)= {pba:.6f}"
    )


if __name__ == "__main__":
  main()
