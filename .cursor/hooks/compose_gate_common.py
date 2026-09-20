#!/usr/bin/env python3
"""compose gate 共通ユーティリティ。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_stdin() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def deny(agent_message: str, user_message: str | None = None) -> None:
    emit(
        {
            "permission": "deny",
            "agent_message": agent_message,
            "user_message": user_message or agent_message,
        }
    )


def allow() -> None:
    emit({"permission": "allow"})


def event_name(data: dict) -> str:
    return str(
        data.get("hook_event_name")
        or data.get("event")
        or data.get("event_name")
        or ""
    ).lower()


def tool_name(data: dict) -> str:
    return str(data.get("tool_name") or "")


def tool_input(data: dict) -> dict:
    raw = data.get("tool_input") or {}
    return raw if isinstance(raw, dict) else {}


def edit_path(data: dict) -> Path:
    inp = tool_input(data)
    return Path(str(inp.get("path") or inp.get("target_notebook") or ""))


def shell_command(data: dict) -> str:
    return str(data.get("command") or tool_input(data).get("command") or "")


def is_japanese_prose_target(path: Path) -> bool:
    if not path.name:
        return False
    if path.suffix.lower() in {".md", ".mdc"}:
        return True
    return path.name in {"BRIEF.md", "README.md"}


BRIEF_DENY = (
    "BRIEF.md は人間が管理する。エージェントは Shell を含め一切書き換えない。"
    "下書きはチャットに出す。"
)

SKILL_DENY = (
    "日本語プロースを .md/.mdc に書く前に、"
    "japanese-tech-writing/SKILL.md を Read する。"
    "整形節（和欧間の空白を含む）を確認してから書く。"
    "Allow で bypass しない。"
)

HOOKS_DIR = Path(__file__).resolve().parent
STATE_DIR = HOOKS_DIR / "state"
