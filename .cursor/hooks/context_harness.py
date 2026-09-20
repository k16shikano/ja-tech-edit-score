#!/usr/bin/env python3
"""コンテキスト注入、BRIEF 拒否、スキル Read ゲート、和欧間検査。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compose_gate_common import (
    BRIEF_DENY,
    SKILL_DENY,
    allow,
    deny,
    edit_path,
    event_name,
    is_japanese_prose_target,
    read_stdin,
    tool_input,
    tool_name,
)
from ja_compose_check import find_violations, format_violation_report
from skill_read_state import (
    mark_skill_read,
    require_tech_writing_before_prose_edit,
    reset_session,
    skill_id_from_read_path,
)

REMINDER = """コンテキスト参照（省略禁止。この順ですべて。人間と会話しているエージェント向け）:
1. BRIEF.md（人間管理の大域。必ず読む。エージェントは書かない）
2. この会話（局所。必要な文脈が出るまで遡る。直前の数発だけで足りるとみなすな）
3. .cursor/work-log.md（エージェントの作業ログ。公開してよいことだけ）
会話または作業ログが BRIEF.md と矛盾したら、BRIEF を直さず人間に方針を確認する。
BRIEF 以外だけ、または作業ログだけで答えるな。
サブエージェントにはこの塊を付けない。必要な文脈は親が Task のプロンプトに書け。

【厳格ゲート】日本語プロースを .md/.mdc に書く前に japanese-tech-writing/SKILL.md を Read する（未 Read なら deny）。
BRIEF.md は Write/StrReplace/Shell すべて deny。和欧間の不要空白は deny。Shell による .md 書き込みも同じゲート。
Allow で bypass しない。既存 BRIEF の空白を規範の代わりに使わない。
"""


def edit_new_text(data: dict) -> str:
    inp = tool_input(data)
    tool = tool_name(data)
    if tool == "Write":
        return str(inp.get("contents") or "")
    if tool in {"StrReplace", "EditNotebook"}:
        return str(inp.get("new_string") or "")
    return ""


def read_file(rel: str) -> str:
    path = Path(rel)
    if not path.is_file():
        return f"（{rel} が無い）"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"（{rel} を読めない: {exc}）"


def files_context() -> str:
    brief = read_file("BRIEF.md")
    log = read_file(".cursor/work-log.md")
    return (
        f"{REMINDER}\n"
        f"===== BRIEF.md =====\n{brief}\n"
        f"===== .cursor/work-log.md =====\n{log}\n"
        "===== ここまで注入。会話はチャット履歴を必要なところまで遡れ ====="
    )


def gate_prose_file_edit(data: dict) -> bool:
    """True if allowed."""
    path = edit_path(data)
    tool = tool_name(data)

    if path.name == "BRIEF.md":
        deny(BRIEF_DENY)
        return False

    if tool not in {"Write", "StrReplace", "EditNotebook"}:
        allow()
        return False

    if not is_japanese_prose_target(path):
        allow()
        return False

    missing_skill = require_tech_writing_before_prose_edit()
    if missing_skill:
        deny(missing_skill + " " + SKILL_DENY)
        return False

    violations = find_violations(edit_new_text(data))
    if violations:
        deny(format_violation_report(violations) + "\n\n" + SKILL_DENY)
        return False

    allow()
    return False


def handle_post_read(data: dict) -> None:
    inp = tool_input(data)
    path = str(inp.get("path") or "")
    skill_id = skill_id_from_read_path(path)
    if skill_id:
        mark_skill_read(skill_id)


def main() -> int:
    data = read_stdin()
    event = event_name(data)
    tool = tool_name(data)

    if event == "sessionstart":
        reset_session()

    if event in {"presubmitprompt", "beforesubmitprompt", "sessionstart", "userpromptsubmit"}:
        from compose_gate_common import emit

        emit({"additional_context": files_context()})
        return 0

    if event == "posttooluse" and tool == "Read":
        handle_post_read(data)
        from compose_gate_common import emit

        emit({})
        return 0

    if tool in {"Write", "StrReplace", "Delete", "EditNotebook"}:
        gate_prose_file_edit(data)
        return 0

    from compose_gate_common import emit

    emit({})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
