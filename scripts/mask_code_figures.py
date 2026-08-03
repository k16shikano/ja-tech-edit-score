#!/usr/bin/env python3
"""学習対象外の記法を draft / revised から除く。

- フェンスコード・画像 → 【コード】 / 【図】（位置だけ残す）
- 行頭の ★ …（編集コメント）→ 行ごと削除
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CODE_PLACEHOLDER = "【コード】"
FIGURE_PLACEHOLDER = "【図】"

# 閉じられた fenced code（言語タグ・属性付きも）
FENCE_RE = re.compile(r"```[^\n]*\n[\s\S]*?```", re.MULTILINE)


def _mask_images(text: str) -> str:
  """`![...](...)` を置換。alt 内の `]` やバッククォートに耐える。"""
  out: list[str] = []
  i = 0
  n = len(text)
  while i < n:
    if text.startswith("![", i):
      k = text.find("](", i + 2)
      if k == -1:
        out.append(text[i])
        i += 1
        continue
      m = text.find(")", k + 2)
      if m == -1:
        out.append(text[i])
        i += 1
        continue
      end = m + 1
      if end < n and text[end] == "{":
        c = text.find("}", end)
        if c != -1:
          end = c + 1
      out.append(FIGURE_PLACEHOLDER)
      i = end
      continue
    out.append(text[i])
    i += 1
  return "".join(out)


def _fence_line(line: str) -> bool:
  # 通常の ``` に加え、引用内の `> ``` もコード開始／終了とみなす
  s = line.lstrip()
  if s.startswith(">"):
    s = s[1:].lstrip()
  return s.startswith("```")


def _mask_fences(text: str) -> str:
  """閉じたフェンスを置換し、閉じのない ``` 開始も塊として置換する。"""
  out = FENCE_RE.sub(CODE_PLACEHOLDER, text)
  lines = out.splitlines(keepends=True)
  rebuilt: list[str] = []
  i = 0
  while i < len(lines):
    if _fence_line(lines[i]):
      j = i + 1
      while j < len(lines) and not _fence_line(lines[j]):
        j += 1
      if j < len(lines):
        j += 1
      rebuilt.append(CODE_PLACEHOLDER + ("\n" if lines[i].endswith("\n") else ""))
      i = j
      continue
    rebuilt.append(lines[i])
    i += 1
  return "".join(rebuilt)


def strip_star_comment_lines(text: str) -> str:
  """行頭（先頭空白の直後）が ★ の行は編集コメントなので削除する。

  文中の ★ は残す。
  """
  if not text:
    return text
  kept: list[str] = []
  for line in text.splitlines(keepends=True):
    body = line[:-1] if line.endswith("\n") else line
    if body.lstrip().startswith("★"):
      continue
    kept.append(line)
  return "".join(kept)


def mask_code_and_figures(text: str) -> str:
  """コード・図のマスクと編集コメント行の削除を行う。"""
  if not text:
    return text
  out = _mask_fences(text)
  out = _mask_images(out)
  out = strip_star_comment_lines(out)
  out = re.sub(
    rf"(?:{re.escape(CODE_PLACEHOLDER)}\s*){{2,}}",
    CODE_PLACEHOLDER + "\n\n",
    out,
  )
  out = re.sub(
    rf"(?:{re.escape(FIGURE_PLACEHOLDER)}\s*){{2,}}",
    FIGURE_PLACEHOLDER + "\n\n",
    out,
  )
  out = re.sub(r"\n{3,}", "\n\n", out)
  if text.endswith("\n") or text.strip():
    return out.strip() + "\n"
  return out


def mask_pair(draft: str, revised: str) -> tuple[str, str, dict]:
  nd = mask_code_and_figures(draft)
  nr = mask_code_and_figures(revised)
  return nd, nr, {
    "draft_had_code": "```" in (draft or ""),
    "revised_had_code": "```" in (revised or ""),
    "draft_had_figure": "![" in (draft or ""),
    "revised_had_figure": "![" in (revised or ""),
    "draft_had_star_comment": any(
      ln.lstrip().startswith("★") for ln in (draft or "").splitlines()
    ),
    "revised_had_star_comment": any(
      ln.lstrip().startswith("★") for ln in (revised or "").splitlines()
    ),
    "changed": nd != draft or nr != revised,
  }


def transform_chat_jsonl(path: Path, out_path: Path | None = None) -> dict:
  INSTR_SPLIT = "\n\n"
  n = changed = 0
  rows: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      n += 1
      msgs = obj.get("messages") or []
      if len(msgs) < 2:
        rows.append(obj)
        continue
      user = str(msgs[0].get("content") or "")
      rev = str(msgs[1].get("content") or "")
      if INSTR_SPLIT in user:
        head, draft = user.split(INSTR_SPLIT, 1)
        prefix = head + INSTR_SPLIT
      else:
        prefix, draft = "", user
      new_d, new_r, st = mask_pair(draft, rev)
      if st["changed"]:
        changed += 1
      msgs[0]["content"] = prefix + new_d
      msgs[1]["content"] = new_r
      meta = obj.setdefault("meta", {})
      meta["code_figure_masked"] = True
      rows.append(obj)

  dest = out_path or path
  with dest.open("w", encoding="utf-8") as f:
    for obj in rows:
      f.write(json.dumps(obj, ensure_ascii=False) + "\n")
  return {"n": n, "changed": changed, "out": str(dest)}


def transform_review_state(path: Path) -> dict:
  if not path.is_file():
    return {"n": 0, "changed": 0}
  state = json.loads(path.read_text(encoding="utf-8"))
  changed = 0
  for _id, dec in state.items():
    for key in ("draft", "revised"):
      if key in dec and isinstance(dec[key], str):
        new = mask_code_and_figures(dec[key])
        if new != dec[key]:
          changed += 1
        dec[key] = new
  path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  return {"n": len(state), "changed_fields": changed, "out": str(path)}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--text", type=Path, help="単体テキストを変換して表示")
  parser.add_argument("--jsonl", type=Path, action="append", default=[], help="chat jsonl")
  parser.add_argument("--review-state", type=Path, default=None)
  args = parser.parse_args()
  if args.text:
    print(mask_code_and_figures(args.text.read_text(encoding="utf-8")))
    return
  for p in args.jsonl:
    print(json.dumps(transform_chat_jsonl(p), ensure_ascii=False))
  if args.review_state:
    print(json.dumps(transform_review_state(args.review_state), ensure_ascii=False))


if __name__ == "__main__":
  main()
