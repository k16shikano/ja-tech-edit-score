#!/usr/bin/env python3
"""ブラインド人手判定用のローカル Web サーバ。

原稿本文を画面に出すため、公開インターネットには出さない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from text_diff_html import draft_to_candidate_diff_html

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "web" / "blind_judge.html"
DEFAULT_PAIRS = ROOT / "data" / "blind_eval" / "pairs.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data" / "blind_eval" / "judgments.jsonl"


INCOMPARABLE_REASONS = (
  ("both_worse", "どちらも下書きより劣化している"),
  ("noedit_vs_worse", "片方は下書きと差分がなく、もう片方は劣化している"),
  ("broken_vs_worse", "片方に致命的な欠陥があり、もう片方は劣化している"),
  ("veto", "全体では片方のほうがよいが、許容できない編集がある"),
  ("other", "その他（コメントへ）"),
)
INCOMPARABLE_REASON_IDS = {k for k, _ in INCOMPARABLE_REASONS}


class JudgeBody(BaseModel):
  pair_id: str
  choice: str = Field(description="a | b | tie | incomparable")
  comment: str = ""
  a_broken: bool = Field(default=False, description="A に致命的な欠落・破壊がある")
  b_broken: bool = Field(default=False, description="B に致命的な欠落・破壊がある")
  a_noedit: bool = Field(default=False, description="A は下書きとの事実上の差分なし")
  b_noedit: bool = Field(default=False, description="B は下書きとの事実上の差分なし")
  incomparable_reason: str = ""


def load_jsonl(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def judged_ids(path: Path) -> set[str]:
  return {str(r.get("pair_id") or "") for r in load_jsonl(path) if r.get("pair_id")}


def create_app(
  *,
  pairs_path: Path,
  judgments_path: Path,
  question: str = "",
  choice_a: str = "A のほうがまし",
  choice_b: str = "B のほうがまし",
  choice_tie: str = "同等",
  choice_incomparable: str = "",
) -> FastAPI:
  app = FastAPI(title="blind-judge")
  pairs = load_jsonl(pairs_path)
  pairs.sort(key=lambda r: int(r.get("order") or 0))
  by_id = {str(p["pair_id"]): p for p in pairs}
  if not question:
    for p in pairs:
      q = str(p.get("question") or "")
      if q:
        question = q
        break

  @app.get("/")
  def index():
    if not INDEX_HTML.is_file():
      raise HTTPException(404, "missing web/blind_judge.html")
    return FileResponse(
      INDEX_HTML,
      headers={"Cache-Control": "no-store, must-revalidate"},
    )

  @app.get("/api/status")
  def status():
    done = judged_ids(judgments_path)
    pending = [p for p in pairs if str(p["pair_id"]) not in done]
    return {
      "total": len(pairs),
      "done": len(done),
      "remaining": len(pending),
      "pairs_path": str(pairs_path),
      "judgments_path": str(judgments_path),
      "question": question,
      "choice_a": choice_a,
      "choice_b": choice_b,
      "choice_tie": choice_tie,
      "choice_incomparable": choice_incomparable,
      "incomparable_reasons": [
        {"id": k, "label": lab} for k, lab in INCOMPARABLE_REASONS
      ],
    }

  @app.get("/api/next")
  def next_pair():
    done = judged_ids(judgments_path)
    for p in pairs:
      if str(p["pair_id"]) not in done:
        draft = str(p.get("context_draft") or "")
        a_text = str(p.get("a_text") or "")
        b_text = str(p.get("b_text") or "")
        return {
          "pair_id": p["pair_id"],
          "order": p.get("order"),
          "context_draft": draft,
          "a_text": a_text,
          "b_text": b_text,
          "a_diff_html": draft_to_candidate_diff_html(draft, a_text),
          "b_diff_html": draft_to_candidate_diff_html(draft, b_text),
          # 出所は画面に出さない
        }
    return {"done": True}

  @app.post("/api/judge")
  def judge(body: JudgeBody):
    if body.choice not in ("a", "b", "tie", "incomparable"):
      raise HTTPException(400, "choice must be a|b|tie|incomparable")
    reason = str(body.incomparable_reason or "")
    if body.choice == "incomparable":
      if reason not in INCOMPARABLE_REASON_IDS:
        raise HTTPException(400, "incomparable requires a reason")
    else:
      reason = ""
    p = by_id.get(body.pair_id)
    if not p:
      raise HTTPException(404, f"unknown pair_id {body.pair_id}")
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    # 再開時の重複防止: 既判定なら上書きせずスキップ
    if body.pair_id in judged_ids(judgments_path):
      return {"ok": True, "duplicate": True}
    row = {
      "pair_id": body.pair_id,
      "item_id": p.get("item_id"),
      "compare_type": p.get("compare_type"),
      "choice": body.choice,
      "comment": body.comment,
      "a_broken": body.a_broken,
      "b_broken": body.b_broken,
      "a_noedit": body.a_noedit,
      "b_noedit": body.b_noedit,
      "incomparable_reason": reason,
      "a_source": p.get("a_source"),
      "b_source": p.get("b_source"),
      "swapped": p.get("swapped"),
    }
    with judgments_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"ok": True, "duplicate": False}

  return app


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
  parser.add_argument("--judgments", default=str(DEFAULT_JUDGMENTS))
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8320)
  parser.add_argument("--question", default="")
  parser.add_argument("--choice-a", default="A のほうがまし")
  parser.add_argument("--choice-b", default="B のほうがまし")
  parser.add_argument("--choice-tie", default="同等")
  parser.add_argument("--choice-incomparable", default="")
  args = parser.parse_args()
  if args.host in ("127.0.0.1", "localhost", "::1"):
    raise SystemExit(
      f"refusing to bind {args.host!r}; use 0.0.0.0 (or a LAN address) so remote clients can reach it"
    )

  pairs_path = Path(args.pairs)
  if not pairs_path.is_file():
    raise SystemExit(
      f"missing {pairs_path}: run scripts/build_blind_pairs.py first "
      "(比較 6 は生成なしでも作れる。1〜5 は生成後)"
    )

  app = create_app(
    pairs_path=pairs_path,
    judgments_path=Path(args.judgments),
    question=args.question,
    choice_a=args.choice_a,
    choice_b=args.choice_b,
    choice_tie=args.choice_tie,
    choice_incomparable=args.choice_incomparable,
  )
  print(
    f"blind-judge http://{args.host}:{args.port}/ "
    f"pairs={pairs_path} judgments={args.judgments}",
    flush=True,
  )
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
  main()
