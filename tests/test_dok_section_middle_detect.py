from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dok_detect_script_is_cuda() -> None:
  text = (ROOT / "scripts" / "dok_section_middle_detect.sh").read_text(encoding="utf-8")
  assert 'DEVICE="${DEVICE:-cuda}"' in text
  assert "train_pref_detect.py" in text
  assert "--device" in text
  assert "pref-detect-section" in text
  assert "--composer-over-draft" in text
  assert "gen_vs_draft" not in text


def test_detect_dockerfile_copies_required_scripts() -> None:
  text = (ROOT / "Dockerfile.section-middle-detect").read_text(encoding="utf-8")
  for path in (
    "train_pref_detect.py",
    "train_pref_sentseq.py",
    "setwise_triple_utils.py",
    "dok_section_middle_detect.sh",
    "data/section_middle/pref_train.jsonl",
    "data/section_middle/pref_valid.jsonl",
  ):
    assert path in text
  assert "ENV DEVICE=cuda" in text


def test_dockerignore_allows_detect_paths() -> None:
  text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
  for path in (
    "!scripts/train_pref_detect.py",
    "!scripts/dok_section_middle_detect.sh",
    "!scripts/train_pref_sentseq.py",
    "!scripts/setwise_triple_utils.py",
    "!data/section_middle/**",
  ):
    assert path in text


def test_detect_cd_dockerfile_sets_composer_over_draft() -> None:
  text = (ROOT / "Dockerfile.section-middle-detect-cd").read_text(encoding="utf-8")
  assert "ENV COMPOSER_OVER_DRAFT=1" in text
  assert "pref-detect-cd-section" in text
  assert "dok_section_middle_detect.sh" in text


def test_detect_build_scripts_are_executable() -> None:
  for rel in (
    "scripts/build_push_section_middle_detect_image.sh",
    "scripts/build_push_section_middle_detect_cd_image.sh",
    "scripts/dok_section_middle_detect.sh",
  ):
    assert (ROOT / rel).stat().st_mode & 0o111
