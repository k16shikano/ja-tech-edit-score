#!/usr/bin/env python3
"""和欧文字間の不要な半角空白を、追加・置換テキストから検出する。"""

from __future__ import annotations

import re
from dataclasses import dataclass

LATIN_THEN_JA = re.compile(
    r"(?<![`/\[])"
    r"(?<![A-Za-z0-9])"
    r"([A-Za-z][A-Za-z0-9+\-]*)"
    r" +"
    r"([ぁ-んァ-ヶー一-龠])"
)

JA_THEN_LATIN = re.compile(
    r"([ぁ-んァ-ヶー一-龠])"
    r" +"
    r"([A-Za-z][A-Za-z0-9+\-]*)"
)

NUM_THEN_JA = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)"
    r" +"
    r"([ぁ-んァ-ヶー一-龠])"
)

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")


@dataclass(frozen=True)
class Violation:
    line: int
    snippet: str
    kind: str


def _strip_non_prose(text: str) -> str:
    without_fences = FENCE_RE.sub("", text)
    return INLINE_CODE_RE.sub("", without_fences)


def find_violations(text: str, *, max_items: int = 8) -> list[Violation]:
    if not text or not text.strip():
        return []

    findings: list[Violation] = []
    stripped = _strip_non_prose(text)
    for line_no, line in enumerate(stripped.splitlines(), start=1):
        for kind, pattern in (
            ("latin-ja-space", LATIN_THEN_JA),
            ("ja-latin-space", JA_THEN_LATIN),
            ("num-ja-space", NUM_THEN_JA),
        ):
            for match in pattern.finditer(line):
                left, right = match.group(1), match.group(2)
                snippet = f"{left} {right}"
                findings.append(Violation(line=line_no, snippet=snippet, kind=kind))
                if len(findings) >= max_items:
                    return findings
    return findings


def format_violation_report(violations: list[Violation]) -> str:
    lines = [
        "和欧文字間に読み上げ不要な半角空白がある。japanese-tech-writing の整形節に従い直す。",
        "既存 BRIEF の用法を真似して空白を入れない。",
    ]
    for item in violations:
        lines.append(f"- {item.line} 行付近: 「{item.snippet}」 ({item.kind})")
    return "\n".join(lines)
