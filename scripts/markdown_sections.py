#!/usr/bin/env python3
"""Markdown を見出し単位の節に分割する。"""

from __future__ import annotations

import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ANCHOR_SUFFIX_RE = re.compile(r"\s*\{#[^}]+\}\s*$")


def normalize_heading(title: str) -> str:
  title = ANCHOR_SUFFIX_RE.sub("", title).strip()
  return title


def split_sections(text: str) -> dict[str, str]:
  """見出しのパンくずをキーに、節本文（見出し行を含む）を返す。"""
  lines = text.splitlines()
  if not any(line.strip() for line in lines):
    return {}

  sections: dict[str, str] = {}
  stack: list[tuple[int, str]] = []
  current_key: str | None = None
  current_body: list[str] = []

  def flush() -> None:
    nonlocal current_key, current_body
    if current_key is None:
      return
    body = "\n".join(current_body).rstrip()
    if body.strip():
      sections[current_key] = body + "\n"
    current_key = None
    current_body = []

  for line in lines:
    match = HEADING_RE.match(line)
    if match:
      flush()
      level = len(match.group(1))
      title = normalize_heading(match.group(2))
      while stack and stack[-1][0] >= level:
        stack.pop()
      stack.append((level, title))
      current_key = " > ".join(title for _, title in stack)
      current_body = [line]
      continue

    if current_key is None:
      current_key = "__preamble__"
      current_body = []
    current_body.append(line)

  flush()

  if not sections and text.strip():
    return {"__file__": text.rstrip() + "\n"}
  return sections


def paragraph_count(text: str) -> int:
  return len([part for part in text.split("\n\n") if part.strip()])


def line_count(text: str) -> int:
  return len([line for line in text.splitlines() if line.strip()])
