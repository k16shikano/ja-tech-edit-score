#!/usr/bin/env python3
"""compose gate の自己テスト。python3 .cursor/hooks/test_compose_gates.py で実行。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = [sys.executable, str(ROOT / ".cursor/hooks/context_harness.py")]
SHELL_GATE = [sys.executable, str(ROOT / ".cursor/hooks/gate_shell_compose.py")]


def run(cmd: list[str], payload: dict) -> dict:
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"hook failed rc={proc.returncode} stderr={proc.stderr}")
    return json.loads(proc.stdout)


def main() -> int:
    run(HARNESS, {"hook_event_name": "sessionStart"})

    assert run(HARNESS, {
        "hook_event_name": "preToolUse",
        "tool_name": "Write",
        "tool_input": {"path": str(ROOT / "docs/t.md"), "contents": "テスト"},
    })["permission"] == "deny"

    run(HARNESS, {
        "hook_event_name": "postToolUse",
        "tool_name": "Read",
        "tool_input": {"path": "/home/k16/.cursor/skills/japanese-tech-writing/SKILL.md"},
    })

    assert run(HARNESS, {
        "hook_event_name": "preToolUse",
        "tool_name": "StrReplace",
        "tool_input": {"path": str(ROOT / "docs/t.md"), "new_string": "LoRA で SFT する"},
    })["permission"] == "deny"

    assert run(HARNESS, {
        "hook_event_name": "preToolUse",
        "tool_name": "Write",
        "tool_input": {"path": str(ROOT / "docs/t.md"), "contents": "LoRAでSFTする"},
    })["permission"] == "allow"

    brief_cmd = "cat > BRIEF.md << 'EOF'\nx\nEOF"
    assert run(SHELL_GATE, {
        "hook_event_name": "beforeShellExecution",
        "command": brief_cmd,
    })["permission"] == "deny"

    run(HARNESS, {"hook_event_name": "sessionStart"})
    assert run(SHELL_GATE, {
        "hook_event_name": "beforeShellExecution",
        "command": f"cat > {ROOT}/docs/t.md << 'EOF'\nok\nEOF",
    })["permission"] == "deny"

    run(HARNESS, {
        "hook_event_name": "postToolUse",
        "tool_name": "Read",
        "tool_input": {"path": "/home/k16/.cursor/skills/japanese-tech-writing/SKILL.md"},
    })
    assert run(SHELL_GATE, {
        "hook_event_name": "beforeShellExecution",
        "command": "cat > docs/t.md << 'EOF'\nComposer による\nEOF",
    })["permission"] == "deny"
    assert run(SHELL_GATE, {
        "hook_event_name": "beforeShellExecution",
        "command": "cat > docs/t.md << 'EOF'\nComposerによる\nEOF",
    })["permission"] == "allow"

    print("all compose gate tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
