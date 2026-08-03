#!/usr/bin/env python3
"""レビューキュー再出力後に、usable な keep/exclude だけ state へ戻す。

- keep / exclude は id 一致、または (project_id, draft, revised) 一致で引き継ぐ
- pending は引き継がない（未 keep は再取得した候補として見直す）
- 対象は呼び出し側が渡す out-dir（8311 / 8312 など）。キュー間に等級差はない
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_chat_texts(path: Path) -> dict[str, tuple[str, str, str]]:
  """id -> (project_id, draft, revised)"""
  out: dict[str, tuple[str, str, str]] = {}
  if not path.is_file():
    return out
  for line in path.open(encoding="utf-8"):
    if not line.strip():
      continue
    obj = json.loads(line)
    meta = obj.get("meta") or {}
    rid = str(meta.get("id") or "")
    pid = str(meta.get("project_id") or "")
    draft = revised = ""
    for msg in obj.get("messages") or []:
      if msg.get("role") == "user":
        content = str(msg.get("content") or "")
        draft = content.split("\n\n", 1)[-1] if "\n\n" in content else content
      elif msg.get("role") == "assistant":
        revised = str(msg.get("content") or "")
    if rid:
      out[rid] = (pid, draft, revised)
  return out


def load_state(path: Path) -> dict:
  if not path.is_file():
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", required=True, help="review queue dir")
  parser.add_argument(
    "--old-state",
    required=True,
    help="再出力前に退避した review_state.json",
  )
  parser.add_argument(
    "--old-train",
    default="",
    help="再出力前 train.jsonl（本文照合用。なければ state の draft/revised）",
  )
  parser.add_argument(
    "--old-heldout",
    default="",
    help="再出力前 heldout.jsonl",
  )
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  new_train = out_dir / "train.jsonl"
  new_held = out_dir / "heldout.jsonl"
  state_path = out_dir / "review_state.json"

  new_rows = {}
  new_rows.update(load_chat_texts(new_train))
  new_rows.update(load_chat_texts(new_held))
  new_ids = set(new_rows)

  old_state = load_state(Path(args.old_state))
  old_texts: dict[str, tuple[str, str, str]] = {}
  if args.old_train:
    old_texts.update(load_chat_texts(Path(args.old_train)))
  if args.old_heldout:
    old_texts.update(load_chat_texts(Path(args.old_heldout)))

  # content key -> decision (keep/exclude only)
  by_content: dict[tuple[str, str, str], tuple[str, dict]] = {}
  for rid, dec in old_state.items():
    if not isinstance(dec, dict):
      continue
    st = dec.get("status")
    if st not in ("keep", "exclude"):
      continue
    if rid in old_texts:
      by_content[old_texts[rid]] = (st, dec)
    elif "draft" in dec and "revised" in dec:
      pid = str(dec.get("project_id") or (old_texts.get(rid) or ("",))[0] or "")
      by_content[(pid, str(dec.get("draft") or ""), str(dec.get("revised") or ""))] = (
        st,
        dec,
      )

  new_state: dict = {}
  n_id = n_content = 0
  for rid, (pid, draft, revised) in new_rows.items():
    if rid in old_state and isinstance(old_state[rid], dict):
      st = old_state[rid].get("status")
      if st in ("keep", "exclude"):
        new_state[rid] = old_state[rid]
        n_id += 1
        continue
    key = (pid, draft, revised)
    if key in by_content:
      st, dec = by_content[key]
      # 新しい id に決定を載せ替える
      transferred = dict(dec)
      transferred["status"] = st
      new_state[rid] = transferred
      n_content += 1

  state_path.write_text(
    json.dumps(new_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  from collections import Counter

  c = Counter(v.get("status") for v in new_state.values())
  print(
    json.dumps(
      {
        "out_dir": str(out_dir),
        "new_items": len(new_ids),
        "state_entries": len(new_state),
        "carried_by_id": n_id,
        "carried_by_content": n_content,
        "statuses": dict(c),
      },
      ensure_ascii=False,
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
