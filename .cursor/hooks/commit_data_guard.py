#!/usr/bin/env python3
"""COMMIT-RULES 準拠: 原稿本文・学習データのコミットを拒否する。"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_PATTERNS = (
    "data/.gitkeep",
    "data/README.md",
    "data/examples.schema.json",
    "data/examples.template.jsonl",
    "data/hard_eval.schema.json",
    "data/hard_eval.template.jsonl",
    "data/hard_eval/README.md",
    "data/batch_import_repos.example.txt",
    "outputs/README.md",
    "outputs/pref-static/model.joblib",
    "outputs/pref-static/metrics.json",
    "**/data/.gitkeep",
    "**/data/**/.gitkeep",
    "**/data/README.md",
    "**/split_manifest.jsonl",
    "**/*.schema.json",
    "**/*.template.jsonl",
)

FORBIDDEN_DIR_NAMES = frozenset(
    {
        "revision_corpus",
        "edit_sft_all",
        "edit_sft_section",
        "edit_sft_hunk_nopara",
        "blind_eval",
        "pref_keep_split_hunk",
        "pref_keep_split_section",
        "section_middle",
        "b_generation",
        "a1_probe",
        "a1_probe_interval",
        "pref_a_split",
        "pairsplit",
        "pref_dataset",
        "pref_split",
        "reference_logps",
        "generated",
        "checkpoints",
    }
)

FORBIDDEN_BASENAMES = frozenset(
    {
        "items.jsonl",
        "canonical.jsonl",
        "keep_section.jsonl",
        "keep_hunk_nopara.jsonl",
        "tech-writing-norms.md",
        "section_mining_manifest.json",
    }
)

MANUSCRIPT_TEXT_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf]{20,}")

GIT_ADD_PATH_RE = re.compile(
    r"(?:^|\s)(?:/[^ ]+/)?git\s+add(?:\s+-[^\s]+)*\s+(.+)$",
    re.IGNORECASE,
)


def normalize_path(path: str) -> str:
    p = path.strip().strip('"').strip("'")
    p = p.removeprefix("./")
    return p.replace("\\", "/")


def matches_allowed(rel: str) -> bool:
    for pat in ALLOWED_PATTERNS:
        if fnmatch.fnmatch(rel, pat):
            return True
    return False


def path_violation(rel: str) -> str | None:
    rel = normalize_path(rel)
    if not rel or rel == ".":
        return None

    if matches_allowed(rel):
        return None

    parts = rel.split("/")
    base = parts[-1]

    if base in FORBIDDEN_BASENAMES:
        return f"禁止ファイル名: {rel}"

    for part in parts:
        if part in FORBIDDEN_DIR_NAMES:
            return f"原稿・学習データディレクトリ: {rel} ({part}/)"
        if part.startswith("archive-"):
            return f"退避データ: {rel}"

    if "reference_logps" in parts:
        return f"学習中間成果物: {rel}"

    data_idx = next((i for i, p in enumerate(parts) if p == "data"), None)
    if data_idx is not None:
        tail = "/".join(parts[data_idx:])
        if tail.endswith(".jsonl") and not matches_allowed(rel):
            return f"data/ 配下の jsonl（許可リスト外）: {rel}"
        if base.endswith(".jsonl") and data_idx > 0:
            return f"ネストした data/ 配下の jsonl: {rel}"

    if parts[0] == "data" and not matches_allowed(rel):
        if rel.endswith((".jsonl", ".json", ".md", ".txt")) and not rel.endswith(
            (".schema.json", ".template.jsonl", ".example.txt")
        ):
            return f"ルート data/ 配下（許可リスト外）: {rel}"

    if parts[0] == "outputs" or (len(parts) > 1 and "outputs" in parts):
        if not matches_allowed(rel):
            return f"outputs/ 配下（pref-static 以外）: {rel}"

    return None


def content_violation(rel: str, text: str) -> str | None:
    if not rel.endswith(".jsonl"):
        return None
    if rel.endswith(".template.jsonl"):
        return None
    if matches_allowed(rel) and rel.endswith("split_manifest.jsonl"):
        return None
    for line in text.splitlines():
        m = MANUSCRIPT_TEXT_RE.search(line)
        if m:
            snippet = m.group(0)[:24]
            return f"原稿らしい連続文字列（20字以上）: {rel} …{snippet}…"
    return None


def check_paths(paths: list[str]) -> list[str]:
    violations: list[str] = []
    for raw in paths:
        rel = normalize_path(raw)
        if not rel or rel.startswith("-") or rel in {".", ".."}:
            continue
        pv = path_violation(rel)
        if pv:
            violations.append(pv)
            continue
        full = REPO_ROOT / rel
        if full.is_file():
            try:
                cv = content_violation(rel, full.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            if cv:
                violations.append(cv)
    return violations


def git_staged_paths() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_show_staged(rel: str) -> str:
    return subprocess.check_output(
        ["git", "show", f":{rel}"],
        cwd=REPO_ROOT,
        text=True,
        errors="replace",
    )


def check_staged() -> list[str]:
    violations: list[str] = []
    for rel in git_staged_paths():
        pv = path_violation(rel)
        if pv:
            violations.append(pv)
            continue
        if rel.endswith((".jsonl", ".json", ".md", ".txt")):
            try:
                text = git_show_staged(rel)
            except subprocess.CalledProcessError:
                continue
            cv = content_violation(rel, text)
            if cv:
                violations.append(cv)
    return violations


def parse_git_add_paths(command: str) -> list[str] | None:
    cmd = command.strip()
    if "git" not in cmd or "add" not in cmd:
        return None
    if re.search(r"\bgit\s+add\b", cmd) is None:
        return None
    if re.search(r"\bgit\s+add\b[^;\n|&]*\s+(?:-\w+\s+)*(-A|--all|\.\s*$|\.\s)", cmd):
        return ["__git_add_all__"]
    m = GIT_ADD_PATH_RE.search(cmd.replace("\n", " "))
    if not m:
        return None
    tail = m.group(1).strip()
    paths: list[str] = []
    for token in re.split(r"\s+", tail):
        if token.startswith("-") or token in {"&&", "||", ";", "|"}:
            break
        norm = normalize_path(token)
        if norm == "data" or norm.endswith("/data"):
            return ["__git_add_all__"]
        paths.append(token)
    return paths if paths else None


def bulk_add_violation() -> str:
    return (
        "git add -A / git add . / git add <dir>/data は禁止。"
        "許可ファイルだけをパス指定して add する（docs/COMMIT-RULES.md）。"
    )


def format_report(violations: list[str]) -> str:
    lines = [
        "原稿本文・学習データのコミットは docs/COMMIT-RULES.md で禁止されている。",
        "ネストした */data/ はルート data/ と別パス。git add でディレクトリごと載せない。",
        "",
    ]
    for v in violations[:12]:
        lines.append(f"- {v}")
    if len(violations) > 12:
        lines.append(f"- …他 {len(violations) - 12} 件")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="git ステージ済みを検査")
    parser.add_argument("--paths", nargs="*", help="パスを直接検査")
    parser.add_argument("--command", help="shell コマンドから git add 引数を解析")
    args = parser.parse_args(argv)

    violations: list[str] = []
    if args.command:
        paths = parse_git_add_paths(args.command)
        if paths is not None:
            if "__git_add_all__" in paths:
                violations.append(bulk_add_violation())
            else:
                violations.extend(check_paths(paths))
    if args.paths:
        violations.extend(check_paths(args.paths))
    if args.staged:
        violations.extend(check_staged())

    if violations:
        sys.stderr.write(format_report(violations) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
