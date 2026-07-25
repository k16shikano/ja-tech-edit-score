#!/usr/bin/env python3
"""文書を文列に分割する（pref-sentseq 学習・推論で共有）。

段落境界は空行（\\n\\n 以上）。段落内は「。」「！」「？」（および閉じ括弧直後）で
文に分割する。見出し行・箇条書き行・コードブロック内の各行は 1 行 = 1 文とする。

段落境界の系列表現（pref-sentseq）は、各段落の先頭文ベクトルに学習可能な
「段落開始」埋め込みを加算する方式（`para_boundary_mode=para_start_embedding`）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SENTSEQ_SPLIT_VERSION = "v1"
PARA_BOUNDARY_MODE = "para_start_embedding"

HEADING_RE = re.compile(r"^#{1,6}\s")
BULLET_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")
CODE_FENCE_LINE_RE = re.compile(r"^\s*```")
CLOSING_BRACKETS = "」』)]"
SENTENCE_END_RE = re.compile(r"[。！？][" + re.escape(CLOSING_BRACKETS) + r"]*")


@dataclass(frozen=True)
class SentenceUnit:
  text: str
  is_para_start: bool


def split_prose_sentences(text: str) -> list[str]:
  text = text.strip()
  if not text:
    return []
  sentences: list[str] = []
  start = 0
  for match in SENTENCE_END_RE.finditer(text):
    end = match.end()
    chunk = text[start:end].strip()
    if chunk:
      sentences.append(chunk)
    start = end
  tail = text[start:].strip()
  if tail:
    sentences.append(tail)
  return sentences


def is_special_line(line: str, *, in_code_fence: bool) -> bool:
  if in_code_fence:
    return True
  stripped = line.strip()
  if not stripped:
    return False
  if HEADING_RE.match(stripped):
    return True
  if BULLET_RE.match(stripped):
    return True
  if CODE_FENCE_LINE_RE.match(stripped):
    return True
  return False


def split_paragraph_units(para: str) -> list[str]:
  """段落を文単位に分割する。"""
  lines = para.splitlines()
  if not lines:
    return []

  units: list[str] = []
  in_code_fence = False
  prose_buf: list[str] = []

  def flush_prose() -> None:
    nonlocal prose_buf
    if not prose_buf:
      return
    chunk = "\n".join(prose_buf).strip()
    prose_buf = []
    if chunk:
      units.extend(split_prose_sentences(chunk))

  for line in lines:
    fence_match = CODE_FENCE_LINE_RE.match(line.strip())
    if fence_match:
      flush_prose()
      in_code_fence = not in_code_fence
      stripped = line.strip()
      if stripped:
        units.append(stripped)
      continue

    if is_special_line(line, in_code_fence=in_code_fence):
      flush_prose()
      stripped = line.strip()
      if stripped:
        units.append(stripped)
      continue

    prose_buf.append(line)

  flush_prose()
  return [u for u in units if u.strip()]


def split_paragraphs(text: str) -> list[str]:
  normalized = text.replace("\r\n", "\n").replace("\r", "\n")
  return [p.strip("\n") for p in re.split(r"\n\n+", normalized) if p.strip()]


def split_document_sentences(text: str) -> list[SentenceUnit]:
  """文書を文列に分割し、段落先頭かどうかを付ける。"""
  units: list[SentenceUnit] = []
  for para in split_paragraphs(text):
    sents = split_paragraph_units(para)
    for idx, sent in enumerate(sents):
      units.append(SentenceUnit(text=sent, is_para_start=(idx == 0)))
  return units


def truncate_sentence_units(
  units: list[SentenceUnit],
  *,
  max_sents: int,
) -> list[SentenceUnit]:
  if max_sents <= 0 or len(units) <= max_sents:
    return units
  return units[:max_sents]
