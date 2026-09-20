#!/usr/bin/env python3
"""commit_data_guard の自己テスト。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = [sys.executable, str(ROOT / ".cursor/hooks/commit_data_guard.py")]
SHELL_GATE = [sys.executable, str(ROOT / ".cursor/hooks/gate_shell_commit_data.py")]


def run_guard(*args: str) -> int:
    proc = subprocess.run(GUARD + list(args), cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode


def run_shell(command: str) -> dict:
    proc = subprocess.run(
        SHELL_GATE,
        input=json.dumps({"hook_event_name": "beforeShellExecution", "command": command}),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
    )
    return json.loads(proc.stdout)


def main() -> int:
    assert run_guard(
        "--paths",
        "a2_pdpo_experiment_spec/data/A2/items.jsonl",
    ) == 1
    assert run_guard(
        "--paths",
        "a2_pdpo_experiment_spec/data/preferences/train.jsonl",
    ) == 1
    assert run_guard(
        "--paths",
        "a2_pdpo_experiment_spec/data/A2/split_manifest.jsonl",
    ) == 0
    assert run_guard(
        "--paths",
        "a2_pdpo_experiment_spec/data/preferences/.gitkeep",
    ) == 0
    assert run_guard("--paths", "data/revision_corpus/keep_section.jsonl") == 1
    assert run_guard("--paths", "data/examples.template.jsonl") == 0

    assert run_shell("git add a2_pdpo_experiment_spec/data")["permission"] == "deny"
    assert run_shell("git add a2_pdpo_experiment_spec/data/A2/split_manifest.jsonl")["permission"] == "allow"
    assert run_shell("git add -A")["permission"] == "deny"

    print("all commit data guard tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
