#!/usr/bin/env python3
"""Phase 4: length leakage diagnostics."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from a2_common import (
    generated_sample_path,
    items_for_split,
    load_a2_items,
    load_config,
    load_split_manifest,
    setup_run_dir,
    spec_path,
    surface_features,
)


def try_classifier(rows: list[dict]) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {"available": False, "reason": "sklearn_not_installed"}

    if len(rows) < 20:
        return {"available": False, "reason": "too_few_rows"}

    x = [[r["features"][k] for k in sorted(rows[0]["features"])] for r in rows]
    y = [r["label"] for r in rows]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "available": True,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "n_rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("diagnose_length_leakage", config, config_path)

    items = load_a2_items()
    manifest = load_split_manifest()
    k = int(config["generic_revision"]["samples_per_source"])

    human_ratios: list[float] = []
    generic_ratios: list[float] = []
    clf_rows: list[dict] = []

    for split in ("train", "dev"):
        for it in items_for_split(items, manifest, split):
            src_len = max(len(it.draft), 1)
            human_ratios.append(len(it.human_revision) / src_len)
            h_feat = surface_features(it.draft, it.human_revision)
            clf_rows.append({"label": 1, "features": h_feat})

            for sample_index in range(k):
                path = generated_sample_path(split, it.id, sample_index)
                if not path.is_file():
                    continue
                obj = json.loads(path.read_text(encoding="utf-8"))
                if obj.get("validation", {}).get("status") != "ok":
                    continue
                text = obj["text"]
                generic_ratios.append(len(text) / src_len)
                g_feat = surface_features(it.draft, text)
                clf_rows.append({"label": 0, "features": g_feat})

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    report = {
        "human_over_draft": stats(human_ratios),
        "generic_over_draft": stats(generic_ratios),
        "surface_classifier": try_classifier(clf_rows),
    }

    out = run_dir / "length_leakage_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
