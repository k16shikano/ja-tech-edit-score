from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dok_gpm_script_is_cuda() -> None:
  text = (ROOT / "scripts" / "dok_pref_gpm.sh").read_text(encoding="utf-8")
  assert 'DEVICE="${DEVICE:-cuda}"' in text
  assert "train_pref_gpm.py" in text
  assert "--device" in text
  assert "pref-gpm-a1b" in text


def test_gpm_dockerfile_copies_required_scripts() -> None:
  text = (ROOT / "Dockerfile.pref-gpm").read_text(encoding="utf-8")
  for path in (
    "train_pref_gpm.py",
    "pref_gpm_runtime.py",
    "train_pref_sentseq.py",
    "sentseq_utils.py",
    "pref_static_utils.py",
    "pref_pair_utils.py",
    "setwise_triple_utils.py",
    "pref_sentseq_runtime.py",
    "dok_pref_gpm.sh",
    "data/pref_keep_split_hunk/train.jsonl",
    "data/pref_keep_split_hunk/valid.jsonl",
    "data/section_middle/pref_train.jsonl",
    "data/section_middle/pref_valid.jsonl",
  ):
    assert path in text
  assert "ENV DEVICE=cuda" in text
  assert 'ENTRYPOINT ["/bin/bash", "/app/dok_pref_gpm.sh"]' in text


def test_dockerignore_allows_gpm_paths() -> None:
  text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
  for path in (
    "!Dockerfile.pref-gpm",
    "!scripts/train_pref_gpm.py",
    "!scripts/pref_gpm_runtime.py",
    "!scripts/dok_pref_gpm.sh",
    "!scripts/build_push_pref_gpm_image.sh",
    "!scripts/train_pref_sentseq.py",
    "!scripts/pref_pair_utils.py",
    "!scripts/setwise_triple_utils.py",
    "!scripts/pref_sentseq_runtime.py",
    "!data/pref_keep_split_hunk/**",
    "!data/section_middle/**",
  ):
    assert path in text


def test_gpm_build_scripts_are_executable() -> None:
  for rel in (
    "scripts/build_push_pref_gpm_image.sh",
    "scripts/dok_pref_gpm.sh",
  ):
    assert (ROOT / rel).stat().st_mode & 0o111


def test_gpm_build_push_requires_registry() -> None:
  text = (ROOT / "scripts" / "build_push_pref_gpm_image.sh").read_text(encoding="utf-8")
  assert 'test -n "${REGISTRY:-}"' in text
  assert "REGISTRY 未設定" in text
