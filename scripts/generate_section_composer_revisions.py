#!/usr/bin/env python3
"""節ペアの下書きから、評価用と同じ指示文で Composer 推敲を生成する。

出力は追記。既存 id は飛ばして再開する。CURSOR_API_KEY が必要。
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from section_middle_utils import (
  PROMPT_TAG,
  append_jsonl,
  build_revision_prompt,
  load_jsonl,
  strip_code_fence,
)


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def done_ids(path: Path) -> set[str]:
  return {str(r.get("id") or "") for r in load_jsonl(path) if r.get("id")}


def generate_one(draft: str, *, model: str, cwd: Path, api_key: str) -> tuple[str, str]:
  from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions

  prompt = build_revision_prompt(draft)
  try:
    result = Agent.prompt(
      prompt,
      AgentOptions(
        api_key=api_key,
        model=model,
        local=LocalAgentOptions(
          cwd=str(cwd),
          setting_sources=[],
        ),
      ),
    )
  except CursorAgentError as err:
    raise SystemExit(
      f"Cursor SDK startup failed: {err.message} (retryable={err.is_retryable})"
    ) from err
  if result.status == "error":
    raise SystemExit("Cursor agent run failed: status=error")
  text = strip_code_fence(str(result.result or ""))
  return text, str(result.status or "")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    default="data/revision_corpus/keep_section.jsonl",
    help="節の推敲前後ペア",
  )
  parser.add_argument(
    "--out",
    default="data/section_middle/revisions.jsonl",
    help="生成の追記先",
  )
  parser.add_argument("--model", default="composer-2.5")
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--offset", type=int, default=0)
  parser.add_argument("--retries", type=int, default=1, help="空出力のとき追加で試す回数")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  items_path = (root / args.input).resolve()
  out_path = (root / args.out).resolve()
  if not items_path.is_file():
    raise SystemExit(f"missing {items_path}")

  items = load_jsonl(items_path)
  if args.offset > 0:
    items = items[args.offset :]
  done = done_ids(out_path)
  pending = [r for r in items if str(r.get("id") or "") not in done]
  if args.limit > 0:
    pending = pending[: args.limit]
  if not pending:
    print(f"nothing to do: out={out_path}")
    return

  api_key = os.environ.get("CURSOR_API_KEY", "").strip()
  if not api_key:
    raise SystemExit("CURSOR_API_KEY is not set")

  cwd = root / "tmp" / "section-middle-agent-cwd"
  cwd.mkdir(parents=True, exist_ok=True)

  print(
    f"processing {len(pending)} "
    f"(input={items_path}, out={out_path}, model={args.model}, skipped_done={len(done)})"
  )
  ok_count = 0
  empty_count = 0
  timings: list[float] = []

  for idx, item in enumerate(pending, start=1):
    item_id = str(item.get("id") or "")
    draft = str(item.get("source_text") or "")
    started = time.perf_counter()
    text = ""
    run_status = ""
    attempts = 1 + max(0, args.retries)
    for _attempt in range(attempts):
      if not draft.strip():
        break
      text, run_status = generate_one(
        draft, model=args.model, cwd=cwd, api_key=api_key
      )
      if text:
        break
    elapsed = time.perf_counter() - started
    timings.append(elapsed)
    row = {
      "id": item_id,
      "project_id": str(item.get("project_id") or ""),
      "generator": args.model,
      "prompt_tag": PROMPT_TAG,
      "text": text,
      "created_at": utc_now_iso(),
      "run_status": run_status,
    }
    append_jsonl(out_path, row)
    if text:
      ok_count += 1
      print(
        f"[{idx}/{len(pending)}] ok id={item_id} "
        f"elapsed={elapsed:.1f}s chars={len(text)}"
      )
    else:
      empty_count += 1
      print(
        f"[{idx}/{len(pending)}] empty id={item_id} "
        f"elapsed={elapsed:.1f}s status={run_status}"
      )

  avg = sum(timings) / len(timings) if timings else 0.0
  print(
    f"done: ok={ok_count} empty={empty_count} "
    f"avg_elapsed={avg:.1f}s total_elapsed={sum(timings):.1f}s"
  )


if __name__ == "__main__":
  main()
