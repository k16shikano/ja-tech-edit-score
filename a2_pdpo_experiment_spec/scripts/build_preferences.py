#!/usr/bin/env python3
"""Phase 3: build P-DPO preference records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from a2_common import (
    generated_sample_path,
    items_for_split,
    load_a2_items,
    load_config,
    load_split_manifest,
    preference_path,
    setup_run_dir,
    spec_path,
    write_jsonl,
)


def load_ok_sample(path: Path) -> dict | None:
    if not path.is_file():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("validation", {}).get("status") != "ok":
        return None
    text = (obj.get("text") or "").strip()
    if not text:
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("build_preferences", config, config_path)

    items = load_a2_items()
    manifest = load_split_manifest()
    k = int(config["generic_revision"]["samples_per_source"])
    weight = 1.0 / k
    splits = [args.split] if args.split else ["train", "dev"]

    summary: dict[str, dict] = {}
    violations: list[str] = []

    for split in splits:
        split_items = items_for_split(items, manifest, split)
        rows: list[dict] = []
        missing: list[str] = []

        for it in split_items:
            for sample_index in range(k):
                sample_path = generated_sample_path(split, it.id, sample_index)
                sample = load_ok_sample(sample_path)
                if sample is None:
                    missing.append(f"{it.id}:sample-{sample_index}")
                    continue
                generic = sample["text"].strip()
                if generic == it.draft.strip():
                    violations.append(f"draft-as-rejected:{it.id}:{sample_index}")
                row = {
                    "a2_id": it.id,
                    "split": split,
                    "draft": it.draft,
                    "chosen": it.human_revision,
                    "rejected": generic,
                    "rejected_sample_index": sample_index,
                    "source_weight": weight,
                }
                rows.append(row)

        if missing:
            print(f"warning {split}: missing ok samples {len(missing)}")

        out = preference_path(split)
        write_jsonl(out, rows)
        summary[split] = {
            "records": len(rows),
            "expected": len(split_items) * k,
            "missing_ok_samples": len(missing),
            "path": str(out),
        }

    (run_dir / "preferences_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if violations:
        raise SystemExit(f"violations: {violations[:5]}")


if __name__ == "__main__":
    main()
