#!/usr/bin/env python3
"""セッション内のスキル Read 状態を記録する。"""

from __future__ import annotations

from compose_gate_common import STATE_DIR

SKILL_MARKERS: dict[str, tuple[str, ...]] = {
    "japanese-tech-writing": (
        "japanese-tech-writing/SKILL.md",
        "japanese-tech-writing\\SKILL.md",
    ),
    "japanese-explanation": (
        "japanese-explanation/SKILL.md",
        "japanese-explanation\\SKILL.md",
    ),
}


def _flag_path(skill_id: str) -> Path:
    return STATE_DIR / f"{skill_id}.read"


def reset_session() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for path in STATE_DIR.glob("*.read"):
        path.unlink(missing_ok=True)


def mark_skill_read(skill_id: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _flag_path(skill_id).write_text("1", encoding="utf-8")


def has_skill_read(skill_id: str) -> bool:
    return _flag_path(skill_id).is_file()


def skill_id_from_read_path(path: str) -> str | None:
    for skill_id, markers in SKILL_MARKERS.items():
        if any(marker in path for marker in markers):
            return skill_id
    return None


def require_tech_writing_before_prose_edit() -> str | None:
    if has_skill_read("japanese-tech-writing"):
        return None
    return (
        "このセッションで japanese-tech-writing/SKILL.md をまだ Read していない。"
        " .md/.mdc への追記・置換の前に Read する。"
    )
