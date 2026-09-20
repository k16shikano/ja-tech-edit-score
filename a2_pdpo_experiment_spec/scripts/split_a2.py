#!/usr/bin/env python3
"""Phase 1: fixed train/dev/test split."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from a2_common import (
    A2_ITEMS_PATH,
    SPLIT_MANIFEST_PATH,
    load_a2_items,
    load_config,
    setup_run_dir,
    spec_path,
    write_jsonl,
)


def split_groups(items, seed: int, sizes: dict[str, int]) -> dict[str, str]:
    by_draft: dict[str, list] = defaultdict(list)
    for it in items:
        by_draft[it.draft].append(it)

    groups = list(by_draft.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    assignment: dict[str, str] = {}
    counts = {"train": 0, "dev": 0, "test": 0}
    limits = sizes

    for group in groups:
        remaining = {k: limits[k] - counts[k] for k in limits}
        eligible = [k for k, rem in remaining.items() if rem >= len(group)]
        if not eligible:
            eligible = [max(remaining, key=remaining.get)]
        split = min(eligible, key=lambda k: (-remaining[k], k))
        for it in group:
            assignment[it.id] = split
        counts[split] += len(group)

    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--force", action="store_true", help="regenerate manifest even if it exists")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("split_a2", config, config_path)

    if SPLIT_MANIFEST_PATH.is_file() and not args.force:
        rows = []
        with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        counts = defaultdict(int)
        for row in rows:
            counts[row["split"]] += 1
        print(f"manifest exists: train={counts['train']} dev={counts['dev']} test={counts['test']}")
        print(f"skip regeneration ({SPLIT_MANIFEST_PATH})")
        return

    if not A2_ITEMS_PATH.is_file():
        raise SystemExit(f"missing {A2_ITEMS_PATH}; run inspect_a2.py first")

    items = load_a2_items()
    seed = int(config["experiment"]["seed"])
    sizes = config["data"]["split"]
    expected = {
        "train": int(sizes["train"]),
        "dev": int(sizes["dev"]),
        "test": int(sizes["test"]),
    }

    assignment = split_groups(items, seed, expected)
    counts = defaultdict(int)
    for split in assignment.values():
        counts[split] += 1

    if dict(counts) != expected:
        raise SystemExit(f"split mismatch: got {dict(counts)} expected {expected}")

    rows = [{"a2_id": it.id, "split": assignment[it.id], "seed": seed} for it in items]
    write_jsonl(SPLIT_MANIFEST_PATH, rows)

    summary = {"seed": seed, "counts": dict(counts), "manifest": str(SPLIT_MANIFEST_PATH)}
    (run_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
