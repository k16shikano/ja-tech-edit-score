from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dok_nce_script_is_cuda() -> None:
  text = (ROOT / "scripts" / "dok_section_middle_nce.sh").read_text(encoding="utf-8")
  assert 'DEVICE="${DEVICE:-cuda}"' in text
  assert "train_pref_nce.py" in text
  assert "--tau" in text
  assert "--device" in text
  assert "pref-nce-section" in text


def test_nce_dockerfile_copies_required_scripts() -> None:
  text = (ROOT / "Dockerfile.section-middle-nce").read_text(encoding="utf-8")
  for path in (
    "train_pref_nce.py",
    "train_pref_sentseq.py",
    "setwise_triple_utils.py",
    "dok_section_middle_nce.sh",
    "data/section_middle/pref_train.jsonl",
    "data/section_middle/pref_valid.jsonl",
  ):
    assert path in text
  assert "ENV DEVICE=cuda" in text
  assert "ENV TAU=0.07" in text


def test_dockerignore_allows_nce_paths() -> None:
  text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
  for path in (
    "!scripts/train_pref_nce.py",
    "!scripts/pref_nce_runtime.py",
    "!scripts/dok_section_middle_nce.sh",
    "!scripts/train_pref_sentseq.py",
    "!scripts/setwise_triple_utils.py",
    "!data/section_middle/**",
  ):
    assert path in text


def test_nce_build_scripts_are_executable() -> None:
  for rel in (
    "scripts/build_push_section_middle_nce_image.sh",
    "scripts/dok_section_middle_nce.sh",
  ):
    assert (ROOT / rel).stat().st_mode & 0o111
