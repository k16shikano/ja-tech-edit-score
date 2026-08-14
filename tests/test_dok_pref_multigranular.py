from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dok_multigranular_script_is_cuda() -> None:
  text = (ROOT / "scripts" / "dok_pref_multigranular.sh").read_text(encoding="utf-8")
  assert 'DEVICE="${DEVICE:-cuda}"' in text
  assert "run_pref_multigranular_stages.py" in text
  assert "--stages" in text
  assert "--device" in text


def test_multigranular_dockerfile_copies_required_scripts() -> None:
  text = (ROOT / "Dockerfile.pref-multigranular").read_text(encoding="utf-8")
  for path in (
    "train_pref_multigranular.py",
    "run_pref_multigranular_stages.py",
    "eval_pref_multigranular.py",
    "setwise_model.py",
    "pref_pair_utils.py",
    "dok_pref_multigranular.sh",
    "data/section_middle/pref_train.jsonl",
    "data/pref_keep_split_hunk/train.jsonl",
    "outputs/pref-bt-keep/",
  ):
    assert path in text
  assert "ENV DEVICE=cuda" in text


def test_dockerignore_allows_multigranular_paths() -> None:
  text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
  for path in (
    "!scripts/train_pref_multigranular.py",
    "!scripts/run_pref_multigranular_stages.py",
    "!scripts/eval_pref_multigranular.py",
    "!scripts/pref_multigranular_runtime.py",
    "!scripts/pref_pair_utils.py",
    "!scripts/dok_pref_multigranular.sh",
    "!data/pref_keep_split_hunk/**",
    "!data/section_middle/**",
  ):
    assert path in text


def test_multigranular_build_scripts_are_executable() -> None:
  for rel in (
    "scripts/build_push_pref_multigranular_image.sh",
    "scripts/dok_pref_multigranular.sh",
  ):
    assert (ROOT / rel).stat().st_mode & 0o111
