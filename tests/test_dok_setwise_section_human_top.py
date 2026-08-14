from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dok_setwise_human_top_script_is_cuda() -> None:
  text = (ROOT / "scripts" / "dok_setwise_section_human_top.sh").read_text(encoding="utf-8")
  assert 'DEVICE="${DEVICE:-cuda}"' in text
  assert "train_pref_setwise.py" in text
  assert "--loss-mode human_top" in text
  assert "--device" in text


def test_setwise_human_top_dockerfile_copies_required_scripts() -> None:
  text = (ROOT / "Dockerfile.setwise-section-human-top").read_text(encoding="utf-8")
  for path in (
    "train_pref_setwise.py",
    "setwise_model.py",
    "setwise_triple_utils.py",
    "dok_setwise_section_human_top.sh",
    "data/section_middle/pref_train.jsonl",
    "data/section_middle/pref_valid.jsonl",
  ):
    assert path in text
  assert "ENV DEVICE=cuda" in text


def test_dockerignore_allows_setwise_human_top_paths() -> None:
  text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
  for path in (
    "!scripts/train_pref_setwise.py",
    "!scripts/setwise_model.py",
    "!scripts/setwise_triple_utils.py",
    "!scripts/dok_setwise_section_human_top.sh",
    "!data/section_middle/**",
  ):
    assert path in text


def test_setwise_human_top_build_scripts_are_executable() -> None:
  for rel in (
    "scripts/build_push_setwise_section_human_top_image.sh",
    "scripts/dok_setwise_section_human_top.sh",
  ):
    assert (ROOT / rel).stat().st_mode & 0o111
