#!/usr/bin/env python3
"""評価データ E（E1/E2/E3）共通の推敲指示文。"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))

from section_middle_utils import PROMPT_TAG, build_revision_prompt

__all__ = ["PROMPT_TAG", "build_e_user_content", "instruction_text"]


def instruction_text() -> str:
  """``export_edit_sft.INSTRUCTION`` の全文（plan-only-a2.md §3.1 にも転記）。"""
  from export_edit_sft import INSTRUCTION

  return INSTRUCTION


def build_e_user_content(draft: str) -> str:
  """E1（Composer）・E2（adapter）・E3（base Qwen）で同一の user 本文。

  形式: ``INSTRUCTION + "\\n\\n" + draft``（``edit_sft_section`` の messages[0] と同型）。
  Qwen 系（E2/E3）はこの文字列を chat template に載せる。E1 はそのまま Composer に渡す。
  """
  return build_revision_prompt(draft)
