from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dok_a1_probe_script_three_modes() -> None:
  text = (ROOT / "scripts" / "dok_a1_probe_gen.sh").read_text(encoding="utf-8")
  assert 'DEVICE="${DEVICE:-cuda}"' in text
  assert "generate_edit_sft.py" in text
  assert "--mode" in text
  assert "run_one base" in text
  assert "run_one base_norms" in text
  assert "run_one adapter" in text
  assert "japanese-tech-writing.md" in text
  assert "tech-writing-norms.md" not in text
  assert "--num-samples 1" in text
  assert 'LIMIT="${LIMIT:-0}"' in text
  assert 'IDS_FILE="${IDS_FILE:-}"' in text
  assert "edit_sft_hunk_nopara/train.jsonl" in text


def test_a1_probe_dockerfile_copies_required_paths() -> None:
  text = (ROOT / "Dockerfile.a1-probe").read_text(encoding="utf-8")
  for path in (
    "generate_edit_sft.py",
    "generation_integrity.py",
    "dok_a1_probe_gen.sh",
    "data/edit_sft_hunk_nopara/train.jsonl",
    "data/a1_probe/japanese-tech-writing.md",
  ):
    assert path in text
  assert "ids.jsonl" not in text
  assert "tech-writing-norms.md" not in text
  assert "ENV DEVICE=cuda" in text
  assert "ENV LIMIT=0" in text
  assert 'ENTRYPOINT ["/bin/bash", "/app/dok_a1_probe_gen.sh"]' in text


def test_a1_probe_dockerignore_allows_paths() -> None:
  text = (ROOT / ".dockerignore.a1-probe").read_text(encoding="utf-8")
  for path in (
    "!scripts/generate_edit_sft.py",
    "!scripts/dok_a1_probe_gen.sh",
    "!data/edit_sft_hunk_nopara/train.jsonl",
    "!data/a1_probe/japanese-tech-writing.md",
    "!outputs/edit-sft/_dok_adapter/**",
  ):
    assert path in text
  assert "ids.jsonl" not in text
  assert "tech-writing-norms.md" not in text


def test_a1_probe_build_scripts_are_executable() -> None:
  for rel in (
    "scripts/build_push_a1_probe_image.sh",
    "scripts/dok_a1_probe_gen.sh",
  ):
    assert (ROOT / rel).stat().st_mode & 0o111


def test_a1_probe_build_push_requires_registry() -> None:
  text = (ROOT / "scripts" / "build_push_a1_probe_image.sh").read_text(
    encoding="utf-8"
  )
  assert 'test -n "${REGISTRY:-}"' in text
  assert "Dockerfile.a1-probe" in text
  assert "a1-probe-gen:latest" in text
  assert "ids.jsonl" not in text
