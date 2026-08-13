#!/usr/bin/env python3
"""下書きと候補文の差分を HTML にする（ブラインド判定 UI 用）。"""
from __future__ import annotations

import difflib
import html


def _esc(s: str) -> str:
  return html.escape(s, quote=False)


def _norm_line(line: str) -> str:
  """行末スペースは差分判定から除外する。"""
  return line.rstrip()


def _lines_same(old: str, new: str) -> bool:
  return _norm_line(old) == _norm_line(new)


def _line_eq(line: str) -> str:
  return (
    f'<div class="diff-line diff-eq"><span class="diff-gutter"> </span>{_esc(line) or " "}</div>'
  )


def _inline_diff_html(old: str, new: str) -> str:
  """1 行内の文字差分（候補側を主表示。行末スペースは無視）。"""
  if _lines_same(old, new):
    return _esc(new)
  old_core = _norm_line(old)
  new_core = _norm_line(new)
  new_suffix = new[len(new_core) :]
  if old_core == new_core:
    return _esc(new)
  sm = difflib.SequenceMatcher(None, old_core, new_core)
  parts: list[str] = []
  for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
      parts.append(_esc(new_core[j1:j2]))
    elif tag == "delete":
      parts.append(f'<span class="diff-del">{_esc(old_core[i1:i2])}</span>')
    elif tag == "insert":
      parts.append(f'<span class="diff-ins">{_esc(new_core[j1:j2])}</span>')
    elif tag == "replace":
      parts.append(f'<span class="diff-del">{_esc(old_core[i1:i2])}</span>')
      parts.append(f'<span class="diff-ins">{_esc(new_core[j1:j2])}</span>')
  return ("".join(parts) or " ") + _esc(new_suffix)


def _lines_norm_equal(draft_lines: list[str], cand_lines: list[str]) -> bool:
  if len(draft_lines) != len(cand_lines):
    return False
  return all(_lines_same(a, b) for a, b in zip(draft_lines, cand_lines))


def draft_to_candidate_diff_html(draft: str, candidate: str) -> str:
  """候補文を下書きと同じ行順で全文表示し、変更箇所だけ強調する。"""
  cand_lines = candidate.split("\n")
  if not candidate and not cand_lines:
    return '<p class="diff-empty">（空）</p>'

  draft_lines = draft.split("\n")
  if _lines_norm_equal(draft_lines, cand_lines):
    return "".join(_line_eq(line) for line in cand_lines)

  draft_norm = [_norm_line(l) for l in draft_lines]
  cand_norm = [_norm_line(l) for l in cand_lines]
  sm = difflib.SequenceMatcher(None, draft_norm, cand_norm)
  parts: list[str] = []
  for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
      for line in cand_lines[j1:j2]:
        parts.append(_line_eq(line))
    elif tag == "insert":
      for line in cand_lines[j1:j2]:
        parts.append(
          f'<div class="diff-line diff-ins-line"><span class="diff-gutter">＋</span>'
          f'<span class="diff-ins">{_esc(line) or " "}</span></div>'
        )
    elif tag == "replace":
      olds = draft_lines[i1:i2]
      news = cand_lines[j1:j2]
      for k, new_line in enumerate(news):
        old_line = olds[k] if k < len(olds) else ""
        if not old_line:
          parts.append(
            f'<div class="diff-line diff-ins-line"><span class="diff-gutter">＋</span>'
            f'<span class="diff-ins">{_esc(new_line) or " "}</span></div>'
          )
        elif _lines_same(old_line, new_line):
          parts.append(_line_eq(new_line))
        else:
          body = _inline_diff_html(old_line, new_line)
          parts.append(
            f'<div class="diff-line diff-chg"><span class="diff-gutter">±</span>{body}</div>'
          )
    # delete: 候補に無い行は下書きパネル側にあるので候補欄では出さない

  return "".join(parts)
