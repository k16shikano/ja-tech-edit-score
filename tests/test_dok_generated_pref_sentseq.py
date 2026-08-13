from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dok_generated_pref_script_is_cuda_and_folds() -> None:
  text = (ROOT / "scripts" / "dok_generated_pref_sentseq.sh").read_text(encoding="utf-8")
  assert 'DEVICE="${DEVICE:-cuda}"' in text
  assert 'FOLDS="${FOLDS:-0 1 2 3 4}"' in text
  assert "train_generated_pref_sentseq.py" in text
  assert "--device" in text


def test_generated_pref_dockerfile_copies_folds() -> None:
  text = (ROOT / "Dockerfile.generated-pref-sentseq").read_text(encoding="utf-8")
  assert "dok_generated_pref_sentseq.sh" in text
  assert "folds_section/fold_4/train.jsonl" in text
  assert "pref_keep_split_section/train.jsonl" in text
  assert "ENV DEVICE=cuda" in text


def test_dockerignore_allows_generated_pref_paths() -> None:
  text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
  assert "!scripts/train_generated_pref_sentseq.py" in text
  assert "!scripts/generated_pref_utils.py" in text
  assert "!scripts/dok_generated_pref_sentseq.sh" in text
  assert "!data/generated_pref_experiment/folds_section/**" in text
