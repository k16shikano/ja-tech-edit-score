#!/usr/bin/env python3
"""A1 三群の生成に、人間の推敲への近さの位置を付けるローカル Web。

原稿本文を画面に出すため、公開インターネットには出さない。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from text_diff_html import draft_to_candidate_diff_html

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "web" / "a1_probe_position.html"
DEFAULT_DIR = ROOT / "outputs" / "a1-probe"
DEFAULT_IDS = ROOT / "data" / "a1_probe" / "ids.jsonl"
DEFAULT_JUDGMENTS = DEFAULT_DIR / "position_judgments.jsonl"
MODES = ("base", "base_norms", "adapter")
HUMAN_POSITIONS = ("a", "eq", "c", "d")
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class JudgeBody(BaseModel):
  pair_id: str
  position: str = Field(description="a | eq | c | d")
  comment: str = ""


def load_jsonl(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def append_jsonl(path: Path, row: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def body_text(s: str) -> str:
  """
  本文一致（b）判定用の正規化。

  - 改行は LF に揃える
  - 行末の空白（スペース/タブ）は無視する（差分強調の都合）
  - ただし「文末改行を消しただけ」は一致扱いにしないため、行末改行の有無は保持する
  """
  s = s.replace("\r\n", "\n").replace("\r", "\n")
  lines = s.split("\n")  # trailing '\n' の場合、末尾に '' が残る
  while lines and lines[0] == "":
    lines.pop(0)
  lines = [ln.rstrip(" \t") for ln in lines]
  if lines:
    lines[0] = lines[0].lstrip(" \t")
  # trailing の ''（=末尾改行）を落とさない
  return "\n".join(lines)


def body_text_trim_trailing_newlines(s: str) -> str:
  """文末改行だけを落として比較する正規化。eq 判定に使う。"""
  s = s.replace("\r\n", "\n").replace("\r", "\n")
  lines = s.split("\n")
  while lines and lines[0] == "":
    lines.pop(0)
  lines = [ln.rstrip(" \t") for ln in lines]
  if lines:
    lines[0] = lines[0].lstrip(" \t")
  while lines and lines[-1] == "":
    lines.pop()
  return "\n".join(lines)


def body_equal(left: str, right: str) -> bool:
  """b（無編集）のための「本文一致」。文末改行の有無も区別する。"""
  return body_text(left) == body_text(right)


def only_trailing_newline_removed(left: str, right: str) -> bool:
  """
  文末の改行だけが違う（改行数の削除/追加）場合だけ True。
  """
  if body_equal(left, right):
    return False
  return body_text_trim_trailing_newlines(left) == body_text_trim_trailing_newlines(right)


_COMMA_VARIANTS = {"，", ","}
_PERIOD_VARIANTS = {"．", "."}
_COMMA_TARGET = {"、"}
_PERIOD_TARGET = {"。"}


def _map_punc_groups(s: str) -> str:
  """
  句読点グループ化（b/eq/d の比較から切り離した上で）:
  - comma group: {， , 、}
  - period group: {． . 。}
  """
  s = s.replace("\r\n", "\n").replace("\r", "\n")
  out_chars: list[str] = []
  for ch in s:
    if ch in _COMMA_VARIANTS or ch in _COMMA_TARGET:
      out_chars.append("X")
    elif ch in _PERIOD_VARIANTS or ch in _PERIOD_TARGET:
      out_chars.append("Y")
    else:
      out_chars.append(ch)
  return "".join(out_chars)


def punctuation_variant_to_target_is_degraded(draft: str, generated: str) -> bool:
  """
  draft の句読点（，/.．/.）を generated が（、/。）に「必要ない置換」している場合だけ劣化（a）に倒す。

  また「句読点以外の編集が混ざって全文長や位置がズレる」ケースでも、
  下書き側に target（、/。）が無いのに生成側に target が出ている場合は
  variant→target の置換が入ったものとして劣化(a)に倒す。
  """
  d0 = body_text_trim_trailing_newlines(draft)
  g0 = body_text_trim_trailing_newlines(generated)

  # 文末改行だけの修正は eq 扱いなので、この関数は先に only_trailing_newline_removed を通ったものだけにする。
  if d0 == g0:
    return False

  d_has_variant = any((ch in _COMMA_VARIANTS or ch in _PERIOD_VARIANTS) for ch in d0)
  g_has_target = any((ch in _COMMA_TARGET or ch in _PERIOD_TARGET) for ch in g0)
  if not (d_has_variant and g_has_target):
    return False

  # 全文比較ではなく「句読点だけの並び」で揃える。
  def extract_tokens(s: str) -> list[str]:
    # tokens:
    # - vc: variant comma（， / ,）
    # - vp: variant period（． / .）
    # - tc: target comma（、）
    # - tp: target period（。）
    out: list[str] = []
    for ch in s:
      if ch in _COMMA_VARIANTS:
        out.append("vc")
      elif ch in _PERIOD_VARIANTS:
        out.append("vp")
      elif ch in _COMMA_TARGET:
        out.append("tc")
      elif ch in _PERIOD_TARGET:
        out.append("tp")
    return out

  dtoks = extract_tokens(d0)
  gtoks = extract_tokens(g0)

  # variant→target への「置換」らしさを、個数で最低限担保する。
  # - 下書き側: target が無い（tc/tp が 0）
  # - 生成側: target が十分に出ている（target の個数 >= draft の variant の個数）
  vc_d = dtoks.count("vc")
  vp_d = dtoks.count("vp")
  tc_d = dtoks.count("tc")
  tp_d = dtoks.count("tp")
  tc_g = gtoks.count("tc")
  tp_g = gtoks.count("tp")

  # comma: 下書きに target（、）が無いのに、生成側に target（、）が variant（，）相当の個数以上ある
  if vc_d > 0 and tc_d == 0 and tc_g >= vc_d:
    return True
  # period: 下書きに target（。）が無いのに、生成側に target（。）が variant（．）相当の個数以上ある
  if vp_d > 0 and tp_d == 0 and tp_g >= vp_d:
    return True

  return False


def pair_id_of(item_id: str, mode: str) -> str:
  return f"{item_id}::{mode}"


def load_ids(path: Path) -> list[str]:
  ids: list[str] = []
  seen: set[str] = set()
  for row in load_jsonl(path):
    rid = str(row.get("id") or "").strip()
    if not rid or rid in seen:
      continue
    seen.add(rid)
    ids.append(rid)
  return ids


def load_samples(samples_dir: Path) -> dict[tuple[str, str], dict]:
  by: dict[tuple[str, str], dict] = {}
  for mode in MODES:
    path = samples_dir / f"{mode}_samples.jsonl"
    if not path.is_file():
      raise FileNotFoundError(f"missing {path}")
    for row in load_jsonl(path):
      rid = str(row.get("id") or "").strip()
      if not rid:
        continue
      by[(rid, mode)] = row
  return by


def judged_pair_ids(path: Path) -> set[str]:
  return {str(r.get("pair_id") or "") for r in load_jsonl(path) if r.get("pair_id")}


def build_queue(
  samples: dict[tuple[str, str], dict],
  *,
  ids: list[str],
  seed: int,
) -> list[dict]:
  rng = random.Random(seed)
  queue: list[dict] = []
  for rid in ids:
    modes = [m for m in MODES if (rid, m) in samples]
    rng.shuffle(modes)
    for mode in modes:
      row = samples[(rid, mode)]
      queue.append(
        {
          "pair_id": pair_id_of(rid, mode),
          "item_id": rid,
          "mode": mode,
          "draft": str(row.get("draft") or ""),
          "gold": str(row.get("gold") or ""),
          "generated": str(row.get("generated") or ""),
        }
      )
  return queue


def write_machine_positions(queue: list[dict], judgments_path: Path) -> tuple[int, int]:
  done = judged_pair_ids(judgments_path)
  n_b = 0
  n_d = 0
  for item in queue:
    pid = item["pair_id"]
    if pid in done:
      continue
    draft = item["draft"]
    generated = item["generated"]
    gold = item["gold"]

    # 優先順位:
    # 1) 文末改行だけの差 -> eq
    # 2) 句読点置換だけ（望ましくない）-> 劣化 (a)
    # 3) 完全一致（本文一致）-> b
    # 4) 人間の推敲一致 -> d
    if body_equal(draft, generated):
      position, source = "b", "machine_noedit"
      n_b += 1
    elif only_trailing_newline_removed(draft, generated):
      position, source = "eq", "machine_final_newline_eq"
    elif punctuation_variant_to_target_is_degraded(draft, generated):
      position, source = "a", "machine_degraded_punc"
    elif body_equal(gold, generated):
      position, source = "d", "machine_gold"
      n_d += 1
    else:
      continue
    append_jsonl(
      judgments_path,
      {
        "pair_id": pid,
        "item_id": item["item_id"],
        "mode": item["mode"],
        "position": position,
        "source": source,
        "comment": "",
      },
    )
    done.add(pid)
  return n_b, n_d


def create_app(
  *,
  samples_dir: Path,
  judgments_path: Path,
  ids_path: Path | None = None,
  seed: int = 42,
) -> FastAPI:
  app = FastAPI(title="a1-probe-position")
  samples = load_samples(samples_dir)
  if ids_path is not None:
    ids = load_ids(ids_path)
    if not ids:
      raise FileNotFoundError(f"no ids in {ids_path}")
  else:
    seen: set[str] = set()
    ids = []
    for (rid, mode) in samples:
      if mode != "base" or rid in seen:
        continue
      seen.add(rid)
      ids.append(rid)
  queue = build_queue(samples, ids=ids, seed=seed)
  by_id = {item["pair_id"]: item for item in queue}
  write_machine_positions(queue, judgments_path)

  def pending() -> list[dict]:
    done = judged_pair_ids(judgments_path)
    return [item for item in queue if item["pair_id"] not in done]

  @app.get("/")
  def index():
    if not INDEX_HTML.is_file():
      raise HTTPException(404, "missing web/a1_probe_position.html")
    return FileResponse(
      INDEX_HTML,
      headers={"Cache-Control": "no-store, must-revalidate"},
    )

  @app.get("/api/status")
  def status():
    done_rows = load_jsonl(judgments_path)
    done_ids = {str(r.get("pair_id") or "") for r in done_rows if r.get("pair_id")}
    auto_b = sum(
      1
      for r in done_rows
      if r.get("pair_id") in by_id
      and r.get("position") == "b"
      and r.get("source") == "machine_noedit"
    )
    auto_d = sum(
      1
      for r in done_rows
      if r.get("pair_id") in by_id
      and r.get("position") == "d"
      and r.get("source") == "machine_gold"
    )
    queued = [item["pair_id"] for item in queue]
    return {
      "total": len(queue),
      "done": sum(1 for pid in queued if pid in done_ids),
      "remaining": len(pending()),
      "auto_b": auto_b,
      "auto_d": auto_d,
      "samples_dir": str(samples_dir),
      "judgments_path": str(judgments_path),
    }

  @app.get("/api/next")
  def next_item():
    left = pending()
    if not left:
      return {"done": True}
    item = left[0]
    draft = item["draft"]
    generated = item["generated"]
    return {
      "pair_id": item["pair_id"],
      "item_id": item["item_id"],
      "draft": draft,
      "generated": generated,
      "generated_diff_html": draft_to_candidate_diff_html(draft, generated),
    }

  @app.post("/api/judge")
  def judge(body: JudgeBody):
    if body.position not in HUMAN_POSITIONS:
      raise HTTPException(400, "position must be a|eq|c|d")
    item = by_id.get(body.pair_id)
    if not item:
      raise HTTPException(404, f"unknown pair_id {body.pair_id}")
    if body.pair_id in judged_pair_ids(judgments_path):
      return {"ok": True, "duplicate": True}
    append_jsonl(
      judgments_path,
      {
        "pair_id": body.pair_id,
        "item_id": item["item_id"],
        "mode": item["mode"],
        "position": body.position,
        "source": "human",
        "comment": body.comment,
      },
    )
    return {"ok": True, "duplicate": False}

  return app


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dir", default=str(DEFAULT_DIR))
  parser.add_argument("--ids", default="")
  parser.add_argument("--judgments", default=str(DEFAULT_JUDGMENTS))
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8327)
  args = parser.parse_args()
  if args.host in LOOPBACK:
    raise SystemExit(
      f"refusing to bind {args.host!r}; use 0.0.0.0 (or a LAN address) so remote clients can reach it"
    )
  samples_dir = Path(args.dir)
  ids_path = Path(args.ids) if str(args.ids).strip() else None
  if ids_path is not None and not ids_path.is_file():
    raise SystemExit(f"missing {ids_path}")
  try:
    load_samples(samples_dir)
  except FileNotFoundError as exc:
    raise SystemExit(str(exc)) from exc
  app = create_app(
    samples_dir=samples_dir,
    judgments_path=Path(args.judgments),
    ids_path=ids_path,
    seed=args.seed,
  )
  print(
    f"a1-probe-position http://{args.host}:{args.port}/ "
    f"dir={samples_dir} judgments={args.judgments}",
    flush=True,
  )
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
  main()
