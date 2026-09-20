#!/usr/bin/env python3
"""Phase 5: precompute reference policy log-probabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from a2_common import (
    load_config,
    read_jsonl,
    setup_run_dir,
    spec_path,
    write_jsonl,
)
from model_utils import (
    compute_sequence_logprob,
    load_base_model,
    load_prompts_from_config,
    load_tokenizer,
    ref_logp_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--split", choices=["train", "dev"], default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("precompute_reference_logps", config, config_path)

    training = config["training"]
    model_name = training["base_model"]
    max_length = int(training.get("max_length", 8192))
    system, user_tpl = load_prompts_from_config(config)

    splits = [args.split] if args.split else ["train", "dev"]
    out_dir = spec_path("data/reference_logps")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, bf16=training.get("precision") == "bf16")
    model.eval()

    stats = {"rows": 0, "splits": {}}

    for split in splits:
        pref_path = spec_path(f"data/preferences/{split}.jsonl")
        if not pref_path.is_file():
            raise SystemExit(f"missing {pref_path}")
        prefs = read_jsonl(pref_path)
        if args.limit > 0:
            prefs = prefs[: args.limit]

        out_path = out_dir / f"{split}.jsonl"
        rows: list[dict] = []
        for rec in prefs:
            for side, text in (("chosen", rec["chosen"]), ("rejected", rec["rejected"])):
                key = ref_logp_key(
                    rec["a2_id"],
                    split,
                    side,
                    rec["rejected_sample_index"] if side == "rejected" else None,
                )
                with torch.no_grad():
                    lp = compute_sequence_logprob(
                        model,
                        tokenizer,
                        rec["draft"],
                        text,
                        system,
                        user_tpl,
                        max_length=max_length,
                    )
                rows.append(
                    {
                        "key": key,
                        "a2_id": rec["a2_id"],
                        "split": split,
                        "side": side,
                        "rejected_sample_index": rec.get("rejected_sample_index"),
                        **lp,
                    }
                )

        write_jsonl(out_path, rows)
        stats["rows"] += len(rows)
        stats["splits"][split] = {"records": len(rows), "path": str(out_path)}

    (run_dir / "precompute_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
