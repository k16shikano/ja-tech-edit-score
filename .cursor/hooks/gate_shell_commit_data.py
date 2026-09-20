#!/usr/bin/env python3
"""Shell 経由の git add / git commit で原稿・学習データ載せを拒否する。"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from commit_data_guard import (
    bulk_add_violation,
    check_paths,
    check_staged,
    format_report,
    parse_git_add_paths,
)
from compose_gate_common import allow, deny, read_stdin, shell_command

GIT_COMMIT_RE = re.compile(r"\bgit\s+commit\b", re.IGNORECASE)


def main() -> int:
    data = read_stdin()
    command = shell_command(data)
    if not command.strip():
        allow()
        return 0

    violations: list[str] = []

    add_paths = parse_git_add_paths(command)
    if add_paths is not None:
        if "__git_add_all__" in add_paths:
            violations.append(bulk_add_violation())
        else:
            violations.extend(check_paths(add_paths))

    if GIT_COMMIT_RE.search(command):
        violations.extend(check_staged())

    if violations:
        deny(format_report(violations))
        return 0

    allow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
