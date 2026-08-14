from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_make_generated_pref_sentseq_cv_respects_fold() -> None:
  result = subprocess.run(
    ["make", "-n", "generated-pref-sentseq-cv", "FOLD=3"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "folds_section/fold_3/train.jsonl" in out
  assert "folds_section/fold_3/valid.jsonl" in out
  assert "generated-pref-sentseq-fold3" in out
  assert "folds_section/fold_0/train.jsonl" not in out


def test_make_section_middle_sentseq_uses_triples_not_keep() -> None:
  result = subprocess.run(
    ["make", "-n", "section-middle-sentseq"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "data/section_middle/pref_train.jsonl" in out
  assert "data/section_middle/pref_valid.jsonl" in out
  assert "outputs/pref-sentseq-section-triples" in out
  assert "pref-sentseq-keep" not in out
  assert "train_pref_sentseq.py" in out
