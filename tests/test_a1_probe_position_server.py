from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))


def _write_samples(tmp: Path) -> None:
  rows = {
    "keep": {
      "id": "keep",
      "draft": "下書き本文。",
      "gold": "人間の推敲。",
    },
    "same": {
      "id": "same",
      "draft": "  無編集の下書き。  ",
      "gold": "人間の推敲。",
    },
  }
  gens = {
    "keep": {"base": "生成A。", "base_norms": "生成B。", "adapter": "人間の推敲。"},
    "same": {
      "base": "無編集の下書き。",
      "base_norms": "生成だけ違う。",
      "adapter": "  無編集の下書き。  ",
    },
  }
  for mode in ("base", "base_norms", "adapter"):
    path = tmp / f"{mode}_samples.jsonl"
    with path.open("w", encoding="utf-8") as f:
      for rid, meta in rows.items():
        row = dict(meta)
        row["mode"] = mode
        row["generated"] = gens[rid][mode]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_noedit_is_machine_b_and_human_labels_acd(tmp_path: Path) -> None:
  from fastapi.testclient import TestClient

  from a1_probe_position_server import create_app

  samples = tmp_path / "samples"
  samples.mkdir()
  _write_samples(samples)
  ids = tmp_path / "ids.jsonl"
  ids.write_text('{"id":"keep"}\n{"id":"same"}\n', encoding="utf-8")
  jud = tmp_path / "jud.jsonl"

  client = TestClient(
    create_app(
      samples_dir=samples,
      judgments_path=jud,
      ids_path=ids,
      seed=0,
    )
  )
  st = client.get("/api/status").json()
  assert st["total"] == 6
  assert st["auto_b"] == 2
  assert st["auto_d"] == 1
  assert st["remaining"] == 3
  nxt = client.get("/api/next").json()
  assert "mode" not in nxt
  assert "gold" not in nxt
  assert nxt.get("item_id") in {"keep", "same"}
  assert nxt["pair_id"].startswith("keep::") or nxt["pair_id"].startswith("same::")
  assert nxt["draft"].strip() != nxt["generated"].strip()
  assert nxt["generated"].strip() != "人間の推敲。"
  bad = client.post("/api/judge", json={"pair_id": nxt["pair_id"], "position": "b"})
  assert bad.status_code == 400
  ok = client.post("/api/judge", json={"pair_id": nxt["pair_id"], "position": "c"})
  assert ok.json()["ok"] is True
  nxt2 = client.get("/api/next").json()
  if not nxt2.get("done"):
    eq = client.post("/api/judge", json={"pair_id": nxt2["pair_id"], "position": "eq"})
    assert eq.status_code == 200
    assert eq.json()["ok"] is True
  rows = [
    json.loads(line)
    for line in jud.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  auto_b = [r for r in rows if r["source"] == "machine_noedit"]
  auto_d = [r for r in rows if r["source"] == "machine_gold"]
  assert {r["position"] for r in auto_b} == {"b"}
  assert all(r["item_id"] == "same" for r in auto_b)
  assert {r["position"] for r in auto_d} == {"d"}
  assert all(r["item_id"] == "keep" and r["mode"] == "adapter" for r in auto_d)


def test_trailing_line_spaces_are_machine_eq(tmp_path: Path) -> None:
  from fastapi.testclient import TestClient

  from a1_probe_position_server import create_app

  samples = tmp_path / "samples"
  samples.mkdir()
  draft = "は`exp`型の式です．\n先程の例。"
  gen = "は`exp`型の式です．  \n先程の例。  \n"
  for mode in ("base", "base_norms", "adapter"):
    row = {
      "id": "trail",
      "mode": mode,
      "draft": draft,
      "gold": "人間の推敲。",
      "generated": gen if mode == "base" else "別の生成。",
    }
    with (samples / f"{mode}_samples.jsonl").open("w", encoding="utf-8") as f:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  ids = tmp_path / "ids.jsonl"
  ids.write_text('{"id":"trail"}\n', encoding="utf-8")
  jud = tmp_path / "jud.jsonl"
  client = TestClient(
    create_app(
      samples_dir=samples,
      judgments_path=jud,
      ids_path=ids,
      seed=0,
    )
  )
  st = client.get("/api/status").json()
  assert st["auto_b"] == 0
  assert st["auto_d"] == 0
  assert st["remaining"] == 2
  nxt = client.get("/api/next").json()
  assert nxt["item_id"] == "trail"
  assert nxt["generated"] != gen

  rows = [
    json.loads(line)
    for line in jud.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  auto_eq = [
    r
    for r in rows
    if r["position"] == "eq" and r["source"] == "machine_final_newline_eq"
  ]
  assert len(auto_eq) == 1
  assert auto_eq[0]["pair_id"].endswith("::base")


def test_loopback_host_is_refused(tmp_path: Path, monkeypatch) -> None:
  from a1_probe_position_server import main

  monkeypatch.setattr(
    sys,
    "argv",
    [
      "a1_probe_position_server.py",
      "--dir",
      str(tmp_path),
      "--host",
      "127.0.0.1",
    ],
  )
  try:
    main()
  except SystemExit as exc:
    assert "refusing to bind" in str(exc)
  else:
    raise AssertionError("expected SystemExit")


def test_trailing_newline_removal_is_machine_eq(tmp_path: Path) -> None:
  from fastapi.testclient import TestClient

  from a1_probe_position_server import create_app

  samples = tmp_path / "samples"
  samples.mkdir()

  # draft は末尾改行あり、generated は末尾改行なし。中身は同一。
  draft = "これは末尾改行あり。\n"
  gen_no_nl = "これは末尾改行あり。"

  for mode in ("base", "base_norms", "adapter"):
    row = {
      "id": "trailnl",
      "mode": mode,
      "draft": draft,
      "gold": "人間の推敲。",
      "generated": gen_no_nl if mode == "base" else "別の生成。",
    }
    (samples / f"{mode}_samples.jsonl").write_text(
      json.dumps(row, ensure_ascii=False) + "\n",
      encoding="utf-8",
    )

  ids = tmp_path / "ids.jsonl"
  ids.write_text("{\"id\":\"trailnl\"}\n", encoding="utf-8")
  jud = tmp_path / "jud.jsonl"

  client = TestClient(
    create_app(
      samples_dir=samples,
      judgments_path=jud,
      ids_path=ids,
      seed=0,
    )
  )

  st = client.get("/api/status").json()
  assert st["total"] == 3
  assert st["auto_b"] == 0
  assert st["auto_d"] == 0
  assert st["remaining"] == 2

  rows = [
    json.loads(line)
    for line in jud.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  auto_eq = [r for r in rows if r["position"] == "eq" and r["source"] == "machine_final_newline_eq"]
  assert len(auto_eq) == 1
  assert auto_eq[0]["pair_id"].endswith("::base")


def test_punctuation_variant_to_target_is_machine_degraded(tmp_path: Path) -> None:
  from fastapi.testclient import TestClient

  from a1_probe_position_server import create_app

  samples = tmp_path / "samples"
  samples.mkdir()

  # draft の句読点（，/.．. 系）を generated 側が（、/。）へ置換しただけ。
  draft = "a，b．c．"
  generated = "a、b。c。"

  for mode in ("base", "base_norms", "adapter"):
    row = {
      "id": "punc",
      "mode": mode,
      "draft": draft,
      "gold": "人間の推敲。",
      "generated": generated if mode == "base" else "別の生成。",
    }
    (samples / f"{mode}_samples.jsonl").write_text(
      json.dumps(row, ensure_ascii=False) + "\n",
      encoding="utf-8",
    )

  ids = tmp_path / "ids.jsonl"
  ids.write_text("{\"id\":\"punc\"}\n", encoding="utf-8")
  jud = tmp_path / "jud.jsonl"

  client = TestClient(
    create_app(
      samples_dir=samples,
      judgments_path=jud,
      ids_path=ids,
      seed=0,
    )
  )

  st = client.get("/api/status").json()
  assert st["total"] == 3
  assert st["auto_b"] == 0
  assert st["auto_d"] == 0
  assert st["remaining"] == 2

  rows = [
    json.loads(line)
    for line in jud.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  degraded = [
    r
    for r in rows
    if r["position"] == "a" and r["source"] == "machine_degraded_punc"
  ]
  assert len(degraded) == 1
  assert degraded[0]["pair_id"].endswith("::base")


def test_punctuation_variant_to_target_is_machine_degraded_even_with_other_edits(
  tmp_path: Path,
) -> None:
  from fastapi.testclient import TestClient

  from a1_probe_position_server import create_app

  samples = tmp_path / "samples"
  samples.mkdir()

  # 句読点（，/.．/. → 、/。）の置換が入っているが、間に非句読点の編集が混ざって
  # 全文の長さや位置は一致しない。
  draft = "a，b．c．"
  generated = "a、bX。cY。"

  for mode in ("base", "base_norms", "adapter"):
    row = {
      "id": "punc2",
      "mode": mode,
      "draft": draft,
      "gold": "人間の推敲。",
      "generated": generated if mode == "base" else "別の生成。",
    }
    (samples / f"{mode}_samples.jsonl").write_text(
      json.dumps(row, ensure_ascii=False) + "\n",
      encoding="utf-8",
    )

  ids = tmp_path / "ids.jsonl"
  ids.write_text("{\"id\":\"punc2\"}\n", encoding="utf-8")
  jud = tmp_path / "jud.jsonl"

  client = TestClient(
    create_app(
      samples_dir=samples,
      judgments_path=jud,
      ids_path=ids,
      seed=0,
    )
  )

  st = client.get("/api/status").json()
  assert st["total"] == 3
  assert st["auto_b"] == 0
  assert st["auto_d"] == 0
  assert st["remaining"] == 2

  rows = [
    json.loads(line)
    for line in jud.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  degraded = [
    r
    for r in rows
    if r["position"] == "a" and r["source"] == "machine_degraded_punc"
  ]
  assert len(degraded) == 1
  assert degraded[0]["pair_id"].endswith("::base")
