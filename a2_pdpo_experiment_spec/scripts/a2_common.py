"""A2 P-DPO experiment shared utilities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

SPEC_ROOT = Path(__file__).resolve().parent.parent
PARENT_ROOT = SPEC_ROOT.parent

DEFAULT_A2_SOURCE = PARENT_ROOT / "data/revision_corpus/keep_section.jsonl"
A2_ITEMS_PATH = SPEC_ROOT / "data/A2/items.jsonl"
SPLIT_MANIFEST_PATH = SPEC_ROOT / "data/A2/split_manifest.jsonl"

THINKING_MARKERS = (
    "<" + "/think>",
    "<think",
    "redacted_thinking",
    "<" + "think>",
)


@dataclass(frozen=True)
class A2Item:
    id: str
    draft: str
    human_revision: str
    metadata: dict[str, Any]


def load_config(path: Path | str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = SPEC_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec_path(rel: str) -> Path:
    return SPEC_ROOT / rel


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SPEC_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PARENT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return "unknown"


def environment_text() -> str:
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
    ]
    for mod in ("torch", "transformers", "peft", "trl", "numpy"):
        try:
            m = __import__(mod)
            lines.append(f"{mod}={getattr(m, '__version__', '?')}")
        except ImportError:
            lines.append(f"{mod}=not_installed")
    return "\n".join(lines) + "\n"


def setup_run_dir(script_name: str, config: dict[str, Any], config_path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = SPEC_ROOT / "outputs" / "runs" / script_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.snapshot.yaml")
    (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit_hash() + "\n", encoding="utf-8")
    meta = {
        "script": script_name,
        "started_at": ts,
        "experiment_name": config.get("experiment", {}).get("name"),
        "seed": config.get("experiment", {}).get("seed"),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink(missing_ok=True)
    latest.symlink_to(run_dir.name)
    return run_dir


def set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def import_a2_from_source(
    source_path: Path | None = None,
    dest_path: Path | None = None,
) -> list[A2Item]:
    source = source_path or DEFAULT_A2_SOURCE
    dest = dest_path or A2_ITEMS_PATH
    if not source.is_file():
        raise FileNotFoundError(f"missing A2 source: {source}")

    items: list[A2Item] = []
    with source.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            draft = (obj.get("source_text") or obj.get("draft") or "").strip()
            human = (obj.get("edited_text") or obj.get("human_revision") or "").strip()
            item_id = (obj.get("id") or f"A2-{idx:04d}").strip()
            meta = {
                k: v
                for k, v in obj.items()
                if k not in ("source_text", "edited_text", "draft", "human_revision")
            }
            items.append(A2Item(id=item_id, draft=draft, human_revision=human, metadata=meta))

    rows = [
        {
            "id": it.id,
            "draft": it.draft,
            "human_revision": it.human_revision,
            "metadata": it.metadata,
        }
        for it in items
    ]
    write_jsonl(dest, rows)
    return items


def load_a2_items(path: Path | None = None) -> list[A2Item]:
    p = path or A2_ITEMS_PATH
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}; run inspect_a2.py first")
    items: list[A2Item] = []
    for row in read_jsonl(p):
        items.append(
            A2Item(
                id=row["id"],
                draft=row["draft"],
                human_revision=row["human_revision"],
                metadata=row.get("metadata") or {},
            )
        )
    return items


def load_split_manifest(path: Path | None = None) -> dict[str, str]:
    p = path or SPLIT_MANIFEST_PATH
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}; run split_a2.py first")
    return {row["a2_id"]: row["split"] for row in read_jsonl(p)}


def items_for_split(items: list[A2Item], manifest: dict[str, str], split: str) -> list[A2Item]:
    return [it for it in items if manifest.get(it.id) == split]


def item_index_map(items: list[A2Item]) -> dict[str, int]:
    ordered = sorted(items, key=lambda x: x.id)
    return {it.id: i for i, it in enumerate(ordered)}


def generation_seed(
    experiment_seed: int,
    item_index: int,
    sample_index: int,
    *,
    offset: int = 0,
) -> int:
    return int(experiment_seed + offset + item_index * 100 + sample_index)


def load_prompt_files(config: dict[str, Any]) -> tuple[str, str]:
    gen = config["generic_revision"]
    system_path = spec_path(gen["prompt_system_file"])
    user_path = spec_path(gen["prompt_user_file"])
    system = system_path.read_text(encoding="utf-8").strip()
    user_tpl = user_path.read_text(encoding="utf-8").strip()
    return system, user_tpl


def prompt_hash(system: str, user_template: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\0")
    h.update(user_template.encode("utf-8"))
    return h.hexdigest()[:16]


def build_user_content(user_template: str, draft: str) -> str:
    return user_template.replace("{{DRAFT}}", draft)


def revision_messages(system: str, user_template: str, draft: str, revision: str | None = None) -> list[dict[str, str]]:
    user_content = build_user_content(user_template, draft)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    if revision is not None:
        messages.append({"role": "assistant", "content": revision})
    return messages


def generated_sample_path(split: str, a2_id: str, sample_index: int) -> Path:
    return spec_path(f"data/generated/{split}/{a2_id}/sample-{sample_index}.json")


def preference_path(split: str) -> Path:
    return spec_path(f"data/preferences/{split}.jsonl")


def validate_generation(text: str, draft: str) -> dict[str, Any]:
    flags: list[str] = []
    cleaned = (text or "").strip()
    if not cleaned:
        flags.append("empty")
    norm_draft = re.sub(r"\s+", "", draft)
    norm_out = re.sub(r"\s+", "", cleaned)
    if cleaned and norm_out == norm_draft:
        flags.append("exact_copy")
    lower = cleaned.lower()
    for marker in THINKING_MARKERS:
        if marker.lower() in lower:
            flags.append("thinking_trace")
            break
    review_patterns = (
        r"^変更点",
        r"^修正点",
        r"^講評",
        r"^要約",
        r"^以下が推敲",
    )
    for pat in review_patterns:
        if re.search(pat, cleaned[:200]):
            flags.append("review_prefix")
            break
    draft_len = max(len(draft), 1)
    out_len = len(cleaned)
    if out_len > 0 and out_len < max(50, int(0.05 * draft_len)):
        flags.append("too_short")
    if out_len > int(3.5 * draft_len) + 500:
        flags.append("too_long")
    status = "ok" if not flags else "failed"
    return {"status": status, "flags": flags}


def count_sentences(text: str) -> int:
    parts = re.split(r"[。！？!?]\s*", text.strip())
    return max(1, len([p for p in parts if p.strip()]))


def surface_features(draft: str, revision: str) -> dict[str, float]:
    src = len(draft)
    rev = len(revision)
    ratio = rev / max(src, 1)
    sents = count_sentences(revision)
    punct = sum(revision.count(c) for c in "、。，．！？!?「」『』（）()")
    return {
        "source_chars": float(src),
        "revision_chars": float(rev),
        "length_ratio": float(ratio),
        "sentence_count": float(sents),
        "avg_sentence_len": float(rev / max(sents, 1)),
        "punctuation_count": float(punct),
    }


def group_preferences_by_source(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["a2_id"]].append(row)
    return dict(grouped)


def add_script_root() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
