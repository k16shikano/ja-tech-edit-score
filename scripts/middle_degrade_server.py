#!/usr/bin/env python3
"""Composer 推敲が下書きより劣化していないかを人が見るローカル Web。

原稿本文を画面に出すため、公開インターネットには出さない。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from section_middle_utils import (
  CHOICE_DEGRADED,
  CHOICE_OK,
  append_jsonl,
  judgments_by_id,
  load_jsonl,
  revisions_by_id,
)
from text_diff_html import draft_to_candidate_diff_html

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "web" / "middle_degrade.html"
DEFAULT_ITEMS = ROOT / "data" / "revision_corpus" / "keep_section.jsonl"
DEFAULT_REVISIONS = ROOT / "data" / "section_middle" / "revisions.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data" / "section_middle" / "judgments.jsonl"


class JudgeBody(BaseModel):
  item_id: str
  choice: str = Field(description="ok | degraded")
  comment: str = ""


def create_app(
  *,
  items_path: Path,
  revisions_path: Path,
  judgments_path: Path,
) -> FastAPI:
  app = FastAPI(title="middle-degrade")
  items = load_jsonl(items_path)
  by_id = {str(r.get("id") or ""): r for r in items if r.get("id")}

  def pending_items() -> list[dict]:
    revisions = revisions_by_id(load_jsonl(revisions_path))
    done = set(judgments_by_id(load_jsonl(judgments_path)))
    out: list[dict] = []
    for item in items:
      item_id = str(item.get("id") or "")
      if not item_id or item_id in done:
        continue
      rev = revisions.get(item_id)
      text = str((rev or {}).get("text") or "").strip()
      if not text:
        continue
      out.append(item)
    return out

  @app.get("/")
  def index():
    if not INDEX_HTML.is_file():
      raise HTTPException(404, "missing web/middle_degrade.html")
    return FileResponse(
      INDEX_HTML,
      headers={"Cache-Control": "no-store, must-revalidate"},
    )

  @app.get("/api/status")
  def status():
    revisions = revisions_by_id(load_jsonl(revisions_path))
    done = judgments_by_id(load_jsonl(judgments_path))
    with_text = sum(
      1
      for item in items
      if str((revisions.get(str(item.get("id") or "")) or {}).get("text") or "").strip()
    )
    pending = pending_items()
    return {
      "total_items": len(items),
      "with_revision": with_text,
      "done": len(done),
      "remaining": len(pending),
      "items_path": str(items_path),
      "revisions_path": str(revisions_path),
      "judgments_path": str(judgments_path),
    }

  @app.get("/api/next")
  def next_item():
    pending = pending_items()
    if not pending:
      return {"done": True}
    item = pending[0]
    item_id = str(item.get("id") or "")
    revisions = revisions_by_id(load_jsonl(revisions_path))
    gen = str((revisions.get(item_id) or {}).get("text") or "")
    draft = str(item.get("source_text") or "")
    return {
      "item_id": item_id,
      "split": str((item.get("meta") or {}).get("split") or ""),
      "source_text": draft,
      "generated_text": gen,
      "diff_html": draft_to_candidate_diff_html(draft, gen),
    }

  @app.post("/api/judge")
  def judge(body: JudgeBody):
    if body.choice not in (CHOICE_OK, CHOICE_DEGRADED):
      raise HTTPException(400, "choice must be ok|degraded")
    item = by_id.get(body.item_id)
    if not item:
      raise HTTPException(404, f"unknown item_id {body.item_id}")
    if body.item_id in judgments_by_id(load_jsonl(judgments_path)):
      return {"ok": True, "duplicate": True}
    row = {
      "item_id": body.item_id,
      "choice": body.choice,
      "comment": body.comment,
      "split": str((item.get("meta") or {}).get("split") or ""),
      "project_id": str(item.get("project_id") or ""),
    }
    append_jsonl(judgments_path, row)
    return {"ok": True, "duplicate": False}

  return app


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--items", default=str(DEFAULT_ITEMS))
  parser.add_argument("--revisions", default=str(DEFAULT_REVISIONS))
  parser.add_argument("--judgments", default=str(DEFAULT_JUDGMENTS))
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8321)
  args = parser.parse_args()
  if args.host in ("127.0.0.1", "localhost", "::1"):
    raise SystemExit(
      f"refusing to bind {args.host!r}; use 0.0.0.0 (or a LAN address) so remote clients can reach it"
    )

  items_path = Path(args.items)
  if not items_path.is_file():
    raise SystemExit(f"missing {items_path}")
  app = create_app(
    items_path=items_path,
    revisions_path=Path(args.revisions),
    judgments_path=Path(args.judgments),
  )
  print(
    f"middle-degrade http://{args.host}:{args.port}/ "
    f"items={items_path} revisions={args.revisions} judgments={args.judgments}",
    flush=True,
  )
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
  main()
