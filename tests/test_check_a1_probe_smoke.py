from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from check_a1_probe_smoke import check_dir, dump_texts, drop_ids, leak_ids, looks_like_prompt_leak


def _write_mode(path: Path, mode: str, *, gen: str, item_id: str = "id-1") -> None:
  row = {
    "id": item_id,
    "mode": mode,
    "draft": "下書き。",
    "generated": gen,
  }
  path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_looks_like_prompt_leak() -> None:
  assert looks_like_prompt_leak("あなたは日本語技術文書の編集者である。\n本文")
  assert not looks_like_prompt_leak("下書きを直した本文。")


def test_check_dir_ok(tmp_path: Path) -> None:
  for mode in ("base", "base_norms", "adapter"):
    _write_mode(tmp_path / f"{mode}_samples.jsonl", mode, gen=f"{mode}の推敲。")
  assert check_dir(tmp_path) == []
  assert dump_texts(tmp_path, n=1).count("== ") == 1


def test_check_dir_id_mismatch(tmp_path: Path) -> None:
  _write_mode(tmp_path / "base_samples.jsonl", "base", gen="a。", item_id="id-1")
  _write_mode(
    tmp_path / "base_norms_samples.jsonl", "base_norms", gen="b。", item_id="id-2"
  )
  _write_mode(tmp_path / "adapter_samples.jsonl", "adapter", gen="c。", item_id="id-1")
  errors = check_dir(tmp_path)
  assert any("id order differs" in e for e in errors)


def test_check_dir_prompt_leak(tmp_path: Path) -> None:
  _write_mode(tmp_path / "base_samples.jsonl", "base", gen="よい本文。")
  _write_mode(
    tmp_path / "base_norms_samples.jsonl",
    "base_norms",
    gen="【文章規範】\n中身",
  )
  _write_mode(tmp_path / "adapter_samples.jsonl", "adapter", gen="よい本文。")
  errors = check_dir(tmp_path)
  assert any("looks like the prompt" in e for e in errors)


def test_drop_leak_ids(tmp_path: Path) -> None:
  for mode in ("base", "base_norms", "adapter"):
    rows = [
      {
        "id": "keep",
        "mode": mode,
        "draft": "下書き。",
        "generated": f"{mode}の推敲。",
      },
      {
        "id": "leak",
        "mode": mode,
        "draft": "下書き。",
        "generated": "【文章規範】\n中身" if mode == "base" else f"{mode}の推敲。",
      },
    ]
    (tmp_path / f"{mode}_samples.jsonl").write_text(
      "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
      encoding="utf-8",
    )
  assert leak_ids(tmp_path) == ["leak"]
  dropped = drop_ids(tmp_path, {"leak"})
  assert len(dropped) == 3
  assert check_dir(tmp_path) == []
  assert leak_ids(tmp_path) == []
