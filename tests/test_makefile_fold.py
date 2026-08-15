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


def test_make_section_middle_nce_uses_triples_not_keep() -> None:
  result = subprocess.run(
    ["make", "-n", "section-middle-nce"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "data/section_middle/pref_train.jsonl" in out
  assert "data/section_middle/pref_valid.jsonl" in out
  assert "outputs/pref-nce-section" in out
  assert "train_pref_nce.py" in out
  assert "pref-sentseq-keep" not in out
  assert "--tau" in out


def test_make_section_middle_detect_uses_triples_not_keep() -> None:
  result = subprocess.run(
    ["make", "-n", "section-middle-detect"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "data/section_middle/pref_train.jsonl" in out
  assert "data/section_middle/pref_valid.jsonl" in out
  assert "outputs/pref-detect-section" in out
  assert "train_pref_detect.py" in out
  assert "pref-sentseq-keep" not in out
  assert "pref-nce-section" not in out


def test_make_section_middle_detect_cd_uses_triples_not_keep() -> None:
  result = subprocess.run(
    ["make", "-n", "section-middle-detect-cd"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "data/section_middle/pref_train.jsonl" in out
  assert "data/section_middle/pref_valid.jsonl" in out
  assert "outputs/pref-detect-cd-section" in out
  assert "train_pref_detect.py" in out
  assert "--composer-over-draft" in out
  assert "pref-sentseq-keep" not in out
  assert "--composer-over-draft" in out
  assert "outputs/pref-detect-section" not in out
  assert "pref-sentseq-keep" not in out


def test_make_revise_uses_section_triples_primary() -> None:
  result = subprocess.run(
    ["make", "-n", "revise", "FILE=dummy.md"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  out = result.stdout
  assert "outputs/pref-sentseq-section-triples" in out
  assert "revise_loop.py" in out


def test_score_server_primary_is_section_triples() -> None:
  text = (ROOT / "scripts" / "score_server.py").read_text(encoding="utf-8")
  assert 'DEFAULT_PRIMARY = ROOT / "outputs" / "pref-sentseq-section-triples"' in text
  assert 'os.environ.get("MIN_MARGIN", "3.4")' in text
  dockerfile = (ROOT / "Dockerfile.serve").read_text(encoding="utf-8")
  assert "PRIMARY_MODEL=/app/outputs/pref-sentseq-section-triples" in dockerfile
  assert "outputs/pref-sentseq-section-triples/model.pt" in dockerfile
  assert "MIN_MARGIN=3.4" in dockerfile
  dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
  assert "!outputs/pref-sentseq-section-triples" in dockerignore
  assert "!outputs/pref-sentseq-section-triples/**" in dockerignore

