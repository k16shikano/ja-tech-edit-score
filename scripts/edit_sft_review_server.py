#!/usr/bin/env python3
"""編集 SFT 学習データの確認・除外・本文修正用ローカル Web。

決定は data/edit_sft/review_state.json に保存し、エクスポートで
train.reviewed.jsonl / heldout.reviewed.jsonl を書く。
原稿本文を載せるため、公開インターネットや不特定向けトンネルには出さない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from export_edit_sft import INSTRUCTION  # noqa: E402

DEFAULT_TRAIN = ROOT / "data" / "edit_sft" / "train.jsonl"
DEFAULT_HELDOUT = ROOT / "data" / "edit_sft" / "heldout.jsonl"
DEFAULT_STATE = ROOT / "data" / "edit_sft" / "review_state.json"
INDEX_HTML = ROOT / "web" / "edit_sft_review.html"

USER_PREFIX = INSTRUCTION + "\n\n"


class ReviewDecision(BaseModel):
  status: str = Field(description="keep | exclude | pending")
  draft: str | None = None
  revised: str | None = None
  note: str = ""


class SaveBody(BaseModel):
  id: str
  status: str = "keep"
  draft: str
  revised: str
  note: str = ""


class ExportBody(BaseModel):
  # pending を keep 扱いしない。学習昇格は明示 keep のみ。
  include_pending_as_keep: bool = False


def draft_from_user_content(content: str) -> str:
  if content.startswith(USER_PREFIX):
    return content[len(USER_PREFIX) :]
  # 旧データ互換: 先頭行が指示なら残り
  if content.startswith(INSTRUCTION):
    parts = content.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else content
  return content


def load_jsonl(path: Path, *, split: str) -> list[dict]:
  rows: list[dict] = []
  if not path.is_file():
    return rows
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      meta = obj.get("meta") or {}
      msgs = obj.get("messages") or []
      if len(msgs) < 2:
        continue
      rid = str(meta.get("id") or "")
      if not rid:
        continue
      rows.append(
        {
          "id": rid,
          "split": split,
          "project_id": str(meta.get("project_id") or ""),
          "source": str(meta.get("source") or "section"),
          "char_similarity": meta.get("char_similarity"),
          "bigram_jaccard": meta.get("bigram_jaccard"),
          "length_ratio": meta.get("length_ratio"),
          "draft_chars": meta.get("draft_chars"),
          "revised_chars": meta.get("revised_chars"),
          "draft_paragraphs": meta.get("draft_paragraphs"),
          "revised_paragraphs": meta.get("revised_paragraphs"),
          "draft": draft_from_user_content(str(msgs[0].get("content") or "")),
          "revised": str(msgs[1].get("content") or ""),
        }
      )
  return rows


def load_state(path: Path) -> dict:
  if not path.is_file():
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  tmp.replace(path)


def build_app(
  *,
  train_path: Path,
  heldout_path: Path,
  state_path: Path,
) -> FastAPI:
  items = load_jsonl(train_path, split="train") + load_jsonl(heldout_path, split="heldout")
  by_id = {row["id"]: row for row in items}
  order = [row["id"] for row in items]
  state = load_state(state_path)

  app = FastAPI(title="edit-sft review (local only)")

  @app.get("/")
  def index() -> FileResponse:
    if not INDEX_HTML.is_file():
      raise HTTPException(500, f"missing {INDEX_HTML}")
    return FileResponse(INDEX_HTML)

  @app.get("/api/meta")
  def meta() -> dict:
    projects = sorted({row["project_id"] for row in items if row["project_id"]})
    counts = {"total": len(items), "train": 0, "heldout": 0, "pending": 0, "keep": 0, "exclude": 0}
    for row in items:
      counts[row["split"]] += 1
      st = (state.get(row["id"]) or {}).get("status", "pending")
      if st not in ("keep", "exclude"):
        st = "pending"
      counts[st] += 1
    return {
      "instruction": INSTRUCTION,
      "n": len(items),
      "projects": projects,
      "counts": counts,
      "state_path": str(state_path),
      "warning": "原稿本文あり。公開インターネットや不特定向けには出さないこと。",
    }

  def filtered_ids(
    *,
    split: str = "",
    project_id: str = "",
    status: str = "",
    q: str = "",
  ) -> list[str]:
    q_norm = q.strip()
    out: list[str] = []
    for rid in order:
      row = by_id[rid]
      st = (state.get(rid) or {}).get("status", "pending")
      if st not in ("keep", "exclude"):
        st = "pending"
      if split and row["split"] != split:
        continue
      if project_id and row["project_id"] != project_id:
        continue
      if status and st != status:
        continue
      if q_norm and q_norm not in row["draft"] and q_norm not in row["revised"] and q_norm not in rid:
        continue
      out.append(rid)
    return out

  def item_summary(rid: str) -> dict:
    row = by_id[rid]
    st = (state.get(rid) or {}).get("status", "pending")
    if st not in ("keep", "exclude"):
      st = "pending"
    return {
      "id": rid,
      "split": row["split"],
      "project_id": row["project_id"],
      "source": row["source"],
      "char_similarity": row["char_similarity"],
      "status": st,
      "draft_chars": len(row["draft"]),
      "revised_chars": len(row["revised"]),
      "preview": row["draft"][:80].replace("\n", " "),
    }

  @app.get("/api/item-ids")
  def list_item_ids(
    split: str = "",
    project_id: str = "",
    status: str = "",
    q: str = "",
  ) -> dict:
    """フィルタ後の全 id（前/次ナビゲーション用。一覧表示の件数制限と独立）。"""
    ids = filtered_ids(split=split, project_id=project_id, status=status, q=q)
    return {"total": len(ids), "ids": ids}

  @app.get("/api/items")
  def list_items(
    split: str = "",
    project_id: str = "",
    status: str = "",
    q: str = "",
    offset: int = 0,
    limit: int = 100,
  ) -> dict:
    ids = filtered_ids(split=split, project_id=project_id, status=status, q=q)
    total = len(ids)
    offset = max(0, offset)
    # 1 ページ最大 200。全件を一度に返す用途は /api/item-ids。
    limit = max(1, min(limit, 200))
    page_ids = ids[offset : offset + limit]
    return {
      "total": total,
      "offset": offset,
      "limit": limit,
      "items": [item_summary(rid) for rid in page_ids],
    }

  @app.get("/api/item/{item_id:path}")
  def get_item(
    item_id: str,
    split: str = "",
    project_id: str = "",
    status: str = "",
    q: str = "",
  ) -> dict:
    row = by_id.get(item_id)
    if not row:
      raise HTTPException(404, "not found")
    dec = state.get(item_id) or {}
    draft = dec["draft"] if "draft" in dec and dec["draft"] is not None else row["draft"]
    revised = dec["revised"] if "revised" in dec and dec["revised"] is not None else row["revised"]
    st = dec.get("status", "pending")
    if st not in ("keep", "exclude", "pending"):
      st = "pending"
    try:
      idx = order.index(item_id)
    except ValueError:
      idx = -1
    # フィルタ後の前後（左一覧・前/次ボタン用）
    filtered = filtered_ids(split=split, project_id=project_id, status=status, q=q)
    try:
      fidx = filtered.index(item_id)
    except ValueError:
      fidx = -1
    return {
      **row,
      "draft": draft,
      "revised": revised,
      "status": st,
      "note": dec.get("note") or "",
      "index": idx,
      "prev_id": order[idx - 1] if idx > 0 else None,
      "next_id": order[idx + 1] if 0 <= idx < len(order) - 1 else None,
      "filter_index": fidx,
      "filter_total": len(filtered),
      "filter_prev_id": filtered[fidx - 1] if fidx > 0 else None,
      "filter_next_id": filtered[fidx + 1] if 0 <= fidx < len(filtered) - 1 else None,
      "original_draft": row["draft"],
      "original_revised": row["revised"],
      "edited": ("draft" in dec and dec["draft"] is not None)
      or ("revised" in dec and dec["revised"] is not None),
    }

  @app.post("/api/save")
  def save(body: SaveBody) -> dict:
    if body.id not in by_id:
      raise HTTPException(404, "not found")
    if body.status not in ("keep", "exclude", "pending"):
      raise HTTPException(400, "status must be keep|exclude|pending")
    draft = body.draft.strip("\n")
    revised = body.revised.strip("\n")
    if body.status != "exclude" and (not draft.strip() or not revised.strip()):
      raise HTTPException(400, "draft/revised empty")
    entry: dict = {
      "status": body.status,
      "note": body.note,
    }
    orig = by_id[body.id]
    if draft != orig["draft"]:
      entry["draft"] = draft
    if revised != orig["revised"]:
      entry["revised"] = revised
    state[body.id] = entry
    save_state(state_path, state)
    return {"ok": True, "id": body.id, "status": body.status}

  @app.post("/api/export")
  def export_reviewed(body: ExportBody) -> dict:
    written = {"train": 0, "heldout": 0, "excluded": 0, "pending_skipped": 0}
    out_paths = {
      "train": state_path.parent / "train.reviewed.jsonl",
      "heldout": state_path.parent / "heldout.reviewed.jsonl",
    }
    handles = {k: p.open("w", encoding="utf-8") for k, p in out_paths.items()}
    try:
      for rid in order:
        row = by_id[rid]
        dec = state.get(rid) or {}
        st = dec.get("status", "pending")
        if st == "exclude":
          written["excluded"] += 1
          continue
        if st == "pending" and not body.include_pending_as_keep:
          written["pending_skipped"] += 1
          continue
        draft = dec["draft"] if "draft" in dec and dec["draft"] is not None else row["draft"]
        revised = (
          dec["revised"] if "revised" in dec and dec["revised"] is not None else row["revised"]
        )
        chat = {
          "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\n{draft}"},
            {"role": "assistant", "content": revised},
          ],
          "meta": {
            "id": rid,
            "project_id": row["project_id"],
            "source": row["source"],
            "char_similarity": row["char_similarity"],
            "bigram_jaccard": row["bigram_jaccard"],
            "length_ratio": row["length_ratio"],
            "review_status": st if st != "pending" else "keep",
            "review_note": dec.get("note") or "",
          },
        }
        handles[row["split"]].write(json.dumps(chat, ensure_ascii=False) + "\n")
        written[row["split"]] += 1
    finally:
      for h in handles.values():
        h.close()
    return {
      "ok": True,
      "written": written,
      "paths": {k: str(v) for k, v in out_paths.items()},
    }

  return app


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
  parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
  parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8310)
  args = parser.parse_args()
  if args.host in ("127.0.0.1", "localhost", "::1"):
    raise SystemExit(
      f"refusing to bind {args.host!r}; use 0.0.0.0 (or a LAN address) so remote clients can reach it"
    )

  app = build_app(
    train_path=args.train,
    heldout_path=args.heldout,
    state_path=args.state,
  )
  print(f"edit-sft review: http://{args.host}:{args.port}/", flush=True)
  print(f"state: {args.state}", flush=True)
  uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
  main()
