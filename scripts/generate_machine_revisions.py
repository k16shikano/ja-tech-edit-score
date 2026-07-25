#!/usr/bin/env python3
"""revision_pairs の下書きから Cursor SDK で機械推敲案を生成する。

出力 JSONL は追記方式。既存行の draft_id を読み直して未処理分だけ続行する。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from steering_utils import load_jsonl

PROMPT_TAG = "plain-v1"
PROMPT_TEMPLATE = (
  "次の日本語の技術文書の下書きを、意味を保ったまま推敲してください。"
  "推敲後の本文だけを出力してください。前置きや説明は不要です。\n\n"
  "{draft}"
)

JAPANESE_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]")


def append_jsonl(path: Path, row: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_draft_ids(path: Path) -> set[str]:
  if not path.is_file():
    return set()
  done: set[str] = set()
  for row in load_jsonl(path):
    draft_id = str(row.get("draft_id") or "")
    if draft_id:
      done.add(draft_id)
  return done


def ensure_revision_pairs(pairs_path: Path, root: Path) -> None:
  if pairs_path.is_file() and pairs_path.stat().st_size > 0:
    return
  print(f"missing {pairs_path}; running make steering-pairs", file=sys.stderr)
  subprocess.run(
    ["make", "-C", str(root), "steering-pairs"],
    check=True,
  )
  if not pairs_path.is_file() or pairs_path.stat().st_size == 0:
    raise SystemExit(f"failed to create revision pairs: {pairs_path}")


def validate_revision(draft: str, text: str) -> str | None:
  cleaned = text.strip()
  if not cleaned:
    return "empty"
  if cleaned == draft.strip():
    return "identical_to_draft"
  draft_len = len(draft)
  text_len = len(cleaned)
  if draft_len > 0 and (text_len > draft_len * 2 or text_len < draft_len * 0.5):
    return "length_out_of_range"
  if not JAPANESE_RE.search(cleaned):
    return "no_japanese"
  return None


def build_prompt(draft: str) -> str:
  return PROMPT_TEMPLATE.format(draft=draft)


def generate_one(draft: str, *, model: str, cwd: Path, api_key: str) -> tuple[str, str]:
  """1件生成して (text, run_status) を返す。"""
  from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions

  prompt = build_prompt(draft)
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
    raise SystemExit(f"Cursor agent run failed: status=error")

  text = str(result.result or "").strip()
  return text, str(result.status or "")


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_output_row(
  *,
  pair: dict,
  model: str,
  text: str,
  reject_reason: str | None,
) -> dict:
  draft_id = str(pair["id"])
  row = {
    "id": f"{draft_id}__machine__{PROMPT_TAG}",
    "project_id": str(pair.get("project_id") or ""),
    "draft_id": draft_id,
    "generator": model,
    "prompt_tag": PROMPT_TAG,
    "text": text,
    "created_at": utc_now_iso(),
  }
  if reject_reason:
    row["status"] = "rejected"
    row["reject_reason"] = reject_reason
  return row


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", default="data/revision_pairs.jsonl", help="revision pairs JSONL")
  parser.add_argument("--out", default="data/machine_revisions.jsonl", help="output JSONL (append)")
  parser.add_argument("--model", default="composer-2.5", help="Cursor agent model id")
  parser.add_argument("--limit", type=int, default=0, help="max rows to process (0 = all)")
  parser.add_argument("--offset", type=int, default=0, help="skip first N input rows")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  pairs_path = (root / args.input).resolve()
  out_path = (root / args.out).resolve()

  ensure_revision_pairs(pairs_path, root)
  pairs = load_jsonl(pairs_path)
  if args.offset > 0:
    pairs = pairs[args.offset :]

  done = load_done_draft_ids(out_path)
  pending = [p for p in pairs if str(p.get("id") or "") not in done]
  if args.limit > 0:
    pending = pending[: args.limit]

  if not pending:
    print(f"nothing to do: out={out_path} (already processed or empty selection)")
    return

  api_key = os.environ.get("CURSOR_API_KEY", "").strip()
  if not api_key:
    raise SystemExit("CURSOR_API_KEY is not set")

  print(
    f"processing {len(pending)} pairs "
    f"(input={pairs_path}, out={out_path}, model={args.model}, skipped_done={len(done)})"
  )

  ok_count = 0
  rejected_count = 0
  timings: list[float] = []

  for idx, pair in enumerate(pending, start=1):
    draft = str(pair.get("draft") or "")
    if not draft:
      row = make_output_row(
        pair=pair,
        model=args.model,
        text="",
        reject_reason="empty_draft",
      )
      append_jsonl(out_path, row)
      rejected_count += 1
      print(f"[{idx}/{len(pending)}] skip empty draft: {pair.get('id')}")
      continue

    started = time.perf_counter()
    text, run_status = generate_one(draft, model=args.model, cwd=root, api_key=api_key)
    elapsed = time.perf_counter() - started
    timings.append(elapsed)

    reject_reason = validate_revision(draft, text)
    row = make_output_row(
      pair=pair,
      model=args.model,
      text=text,
      reject_reason=reject_reason,
    )
    append_jsonl(out_path, row)

    if reject_reason:
      rejected_count += 1
      print(
        f"[{idx}/{len(pending)}] rejected ({reject_reason}) "
        f"draft_id={pair.get('id')} elapsed={elapsed:.1f}s status={run_status}"
      )
    else:
      ok_count += 1
      print(
        f"[{idx}/{len(pending)}] ok draft_id={pair.get('id')} "
        f"elapsed={elapsed:.1f}s chars={len(text)}"
      )

  if timings:
    avg = sum(timings) / len(timings)
    print(
      f"done: ok={ok_count} rejected={rejected_count} "
      f"avg_elapsed={avg:.1f}s total_elapsed={sum(timings):.1f}s"
    )
  else:
    print(f"done: ok={ok_count} rejected={rejected_count}")


if __name__ == "__main__":
  main()
