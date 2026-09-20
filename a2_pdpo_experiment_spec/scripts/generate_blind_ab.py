#!/usr/bin/env python3
"""Phase 8: blind A/B generation files for human judgment."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from a2_common import (
    items_for_split,
    load_a2_items,
    load_config,
    load_split_manifest,
    revision_messages,
    set_seeds,
    setup_run_dir,
    spec_path,
)
from model_utils import (
    apply_chat_template,
    load_policy_model,
    load_prompts_from_config,
    load_tokenizer,
)


BLIND_QUESTION = (
    "AとBのうち、あなた自身の推敲判断に近いものを選んでください。"
    "どちらとも選べない場合は「選べない」としてください。"
)


def generate_one(model, tokenizer, draft: str, system: str, user_tpl: str, *, max_new_tokens: int) -> str:
    messages = revision_messages(system, user_tpl, draft)
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/pdpo_best")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("generate_blind_ab", config, config_path)
    seed = int(config["experiment"]["seed"])
    set_seeds(seed + 777)

    items = load_a2_items()
    manifest = load_split_manifest()
    test_items = items_for_split(items, manifest, "test")
    if args.limit > 0:
        test_items = test_items[: args.limit]

    training = config["training"]
    system, user_tpl = load_prompts_from_config(config)
    tokenizer = load_tokenizer(training["base_model"])
    max_new = min(int(config["generic_revision"]["sampling"]["max_tokens"]), 4096)

    editor_model = load_policy_model(
        training["base_model"],
        training["lora"],
        adapter_path=spec_path(args.checkpoint),
    )

    rows: list[dict] = []
    mapping: list[dict] = []

    for it in test_items:
        with editor_model.disable_adapter():
            generic_text = generate_one(
                editor_model, tokenizer, it.draft, system, user_tpl, max_new_tokens=max_new
            )
        editor_text = generate_one(
            editor_model, tokenizer, it.draft, system, user_tpl, max_new_tokens=max_new
        )
        if random.random() < 0.5:
            option_a, option_b = generic_text, editor_text
            key = {"A": "generic_off", "B": "editor_on"}
        else:
            option_a, option_b = editor_text, generic_text
            key = {"A": "editor_on", "B": "generic_off"}

        rows.append(
            {
                "a2_id": it.id,
                "draft": it.draft,
                "question": BLIND_QUESTION,
                "option_a": option_a,
                "option_b": option_b,
            }
        )
        mapping.append({"a2_id": it.id, "mapping": key})

    blind_path = run_dir / "blind_ab.jsonl"
    map_path = run_dir / "blind_ab_mapping.json"
    with blind_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {blind_path} ({len(rows)} items)")
    print(f"wrote {map_path}")


if __name__ == "__main__":
    main()
