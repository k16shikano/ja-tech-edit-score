#!/usr/bin/env python3
"""Phase 9 (optional): hidden representation deltas adapter OFF vs ON."""
from __future__ import annotations

import argparse
import json

from a2_common import load_config, setup_run_dir, spec_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("analyze_representations", config, config_path)

    note = {
        "status": "not_run",
        "reason": "Phase 9 runs only after Phase 7/8 show an effect.",
    }
    (run_dir / "representation_note.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(note, ensure_ascii=False))


if __name__ == "__main__":
    main()
