#!/usr/bin/env python3
"""Shell 経由の BRIEF/.md 書き込みと、和欧間空白の bypass を拒否する。"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compose_gate_common import BRIEF_DENY, allow, deny, read_stdin, shell_command
from ja_compose_check import find_violations, format_violation_report
from skill_read_state import require_tech_writing_before_prose_edit

BRIEF_PATH_RE = re.compile(r"BRIEF\.md", re.IGNORECASE)
PROSE_PATH_RE = re.compile(r"[^\s;&|\"']+\.(?:md|mdc)\b", re.IGNORECASE)

WRITE_HINT_RE = re.compile(
    r"(?:"
    r"[>]{1,2}|"
    r"\bcat\b|\btee\b|\bsponge\b|"
    r"\bsed\s+-i\b|"
    r"\b(cp|mv)\b|"
    r"open\s*\(|"
    r"write_text\s*\(|"
    r"Path\s*\([^)]+\)\.write"
    r")",
    re.IGNORECASE,
)


def _extract_prose_paths(command: str) -> list[str]:
    paths: list[str] = []
    for match in PROSE_PATH_RE.finditer(command):
        path = match.group(0)
        if path not in paths:
            paths.append(path)
    return paths


def _writes_brief(command: str) -> bool:
    if not BRIEF_PATH_RE.search(command):
        return False
    return bool(WRITE_HINT_RE.search(command))


def _writes_prose(command: str) -> bool:
    paths = _extract_prose_paths(command)
    if not paths:
        return False
    return bool(WRITE_HINT_RE.search(command))


def main() -> int:
    data = read_stdin()
    command = shell_command(data)
    if not command.strip():
        allow()
        return 0

    if _writes_brief(command):
        deny(BRIEF_DENY)
        return 0

    if not _writes_prose(command):
        allow()
        return 0

    missing_skill = require_tech_writing_before_prose_edit()
    if missing_skill:
        deny(missing_skill)
        return 0

    violations = find_violations(command)
    if violations:
        deny(format_violation_report(violations))
        return 0

    allow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
