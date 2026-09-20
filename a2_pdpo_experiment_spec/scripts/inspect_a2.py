#!/usr/bin/env python3
"""Phase 0: validate A2 dataset."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from a2_common import (
    A2_ITEMS_PATH,
    import_a2_from_source,
    load_a2_items,
    load_config,
    setup_run_dir,
    spec_path,
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--import", dest="do_import", action="store_true", help="force re-import from parent keep_section")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    expected = int(config["data"]["expected_items"])
    run_dir = setup_run_dir("inspect_a2", config, config_path)

    if not A2_ITEMS_PATH.is_file() or args.do_import:
        import_a2_from_source()

    items = load_a2_items()
    ids = [it.id for it in items]
    id_counts = Counter(ids)
    dup_ids = [i for i, c in id_counts.items() if c > 1]

    empty_draft = [it.id for it in items if not it.draft.strip()]
    empty_human = [it.id for it in items if not it.human_revision.strip()]
    exact_pairs = [it.id for it in items if it.draft.strip() == it.human_revision.strip()]

    draft_counts = Counter(it.draft for it in items)
    human_counts = Counter(it.human_revision for it in items)
    dup_drafts = [d for d, c in draft_counts.items() if c > 1]
    dup_humans = [h for h, c in human_counts.items() if c > 1]

    draft_lens = [len(it.draft) for it in items]
    human_lens = [len(it.human_revision) for it in items]
    draft_tokens = [estimate_tokens(it.draft) for it in items]
    human_tokens = [estimate_tokens(it.human_revision) for it in items]

    extreme = [
        {
            "id": it.id,
            "draft_chars": len(it.draft),
            "human_chars": len(it.human_revision),
        }
        for it in items
        if len(it.draft) > 20000 or len(it.human_revision) > 20000
    ]

    report = {
        "count": len(items),
        "expected": expected,
        "count_ok": len(items) == expected,
        "duplicate_ids": dup_ids,
        "empty_draft": empty_draft,
        "empty_human": empty_human,
        "exact_pairs": exact_pairs,
        "duplicate_draft_groups": len(dup_drafts),
        "duplicate_human_groups": len(dup_humans),
        "draft_chars": {
            "min": min(draft_lens),
            "max": max(draft_lens),
            "mean": statistics.mean(draft_lens),
            "median": statistics.median(draft_lens),
        },
        "human_chars": {
            "min": min(human_lens),
            "max": max(human_lens),
            "mean": statistics.mean(human_lens),
            "median": statistics.median(human_lens),
        },
        "draft_tokens_est": {
            "min": min(draft_tokens),
            "max": max(draft_tokens),
            "mean": statistics.mean(draft_tokens),
            "median": statistics.median(draft_tokens),
        },
        "human_tokens_est": {
            "min": min(human_tokens),
            "max": max(human_tokens),
            "mean": statistics.mean(human_tokens),
            "median": statistics.median(human_tokens),
        },
        "extreme_pairs": extreme,
    }

    report_path = run_dir / "inspect_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# A2 inspect report",
        "",
        f"- items: {report['count']} (expected {expected})",
        f"- duplicate ids: {len(dup_ids)}",
        f"- empty draft: {len(empty_draft)}",
        f"- empty human_revision: {len(empty_human)}",
        f"- exact draft==human pairs: {len(exact_pairs)}",
        f"- duplicate draft groups: {report['duplicate_draft_groups']}",
        f"- duplicate human_revision groups: {report['duplicate_human_groups']}",
        f"- draft chars median: {report['draft_chars']['median']:.0f}",
        f"- human chars median: {report['human_chars']['median']:.0f}",
        f"- extreme pairs (>20k chars): {len(extreme)}",
        "",
    ]
    (run_dir / "inspect_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {report_path}")

    hard_fail = (
        report["count"] != expected
        or dup_ids
        or empty_draft
        or empty_human
    )
    if hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
