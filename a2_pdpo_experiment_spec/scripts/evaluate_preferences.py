#!/usr/bin/env python3
"""Phase 7: automatic evaluation on held-out test split."""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from datetime import datetime, timezone

import torch

from a2_common import (
    generation_seed,
    item_index_map,
    items_for_split,
    load_a2_items,
    load_config,
    load_prompt_files,
    load_split_manifest,
    prompt_hash,
    revision_messages,
    setup_run_dir,
    spec_path,
    validate_generation,
)
from model_utils import (
    compute_sequence_logprob,
    load_policy_model,
    load_prompts_from_config,
    load_tokenizer,
)


FRESH_SEED_OFFSET = 1_000_000


def call_llama_chat(endpoint: str, model_name: str, messages, sampling: dict, seed: int) -> str:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model_name,
        "messages": messages,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "seed": seed,
        "max_tokens": sampling["max_tokens"],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def ensure_fresh_generic(it, sample_index, config, idx_map, system, user_tpl, phash) -> dict:
    out_path = spec_path(f"data/generated/test_fresh/{it.id}/sample-{sample_index}.json")
    if out_path.is_file():
        obj = json.loads(out_path.read_text(encoding="utf-8"))
        if obj.get("validation", {}).get("status") == "ok":
            return obj

    gen_cfg = config["generic_revision"]
    seed = generation_seed(
        int(config["experiment"]["seed"]),
        idx_map[it.id],
        sample_index,
        offset=FRESH_SEED_OFFSET,
    )
    messages = revision_messages(system, user_tpl, it.draft)
    backend = gen_cfg["generator"]["backend"]
    if backend == "transformers":
        from generate_generic import call_transformers_chat

        text, _ = call_transformers_chat(
            gen_cfg["generator"]["model_name"],
            messages,
            gen_cfg["sampling"],
            seed,
        )
    else:
        text = call_llama_chat(
            gen_cfg["generator"]["endpoint"],
            gen_cfg["generator"]["model_name"],
            messages,
            gen_cfg["sampling"],
            seed,
        )
    validation = validate_generation(text, it.draft)
    record = {
        "a2_id": it.id,
        "sample_index": sample_index,
        "seed": seed,
        "prompt_hash": phash,
        "text": text.strip(),
        "validation": validation,
        "generation_metadata": {"generated_at": datetime.now(timezone.utc).isoformat()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if validation["status"] != "ok":
        raise RuntimeError(f"fresh generic failed validation for {it.id} sample-{sample_index}")
    return record


def s_theta(model, tokenizer, draft, completion, system, user_tpl, max_length) -> float:
    with model.disable_adapter():
        ref = compute_sequence_logprob(
            model, tokenizer, draft, completion, system, user_tpl, max_length=max_length
        )["total_logprob"]
    pol = compute_sequence_logprob(
        model, tokenizer, draft, completion, system, user_tpl, max_length=max_length
    )["total_logprob"]
    return pol - ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/pdpo_best")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("evaluate_preferences", config, config_path)

    items = load_a2_items()
    manifest = load_split_manifest()
    test_items = items_for_split(items, manifest, "test")
    idx_map = item_index_map(items)
    k = int(config["evaluation"]["fresh_generic_samples_per_test_source"])

    system, user_tpl = load_prompt_files(config)
    phash = prompt_hash(system, user_tpl)

    if not args.skip_generate:
        for it in test_items:
            for sample_index in range(k):
                ensure_fresh_generic(it, sample_index, config, idx_map, system, user_tpl, phash)

    training = config["training"]
    model_name = training["base_model"]
    max_length = int(training.get("max_length", 8192))
    ckpt = spec_path(args.checkpoint)

    tokenizer = load_tokenizer(model_name)
    model = load_policy_model(model_name, training["lora"], adapter_path=ckpt)
    model.eval()

    per_source: list[dict] = []
    all_margins: list[float] = []
    margin_len_pairs: list[tuple[float, float]] = []

    with torch.no_grad():
        for it in test_items:
            pair_margins: list[float] = []
            src_len = max(len(it.draft), 1)
            for sample_index in range(k):
                sample_path = spec_path(f"data/generated/test_fresh/{it.id}/sample-{sample_index}.json")
                sample = json.loads(sample_path.read_text(encoding="utf-8"))
                generic = sample["text"]
                margin = s_theta(model, tokenizer, it.draft, it.human_revision, system, user_tpl, max_length) - s_theta(
                    model, tokenizer, it.draft, generic, system, user_tpl, max_length
                )
                pair_margins.append(margin)
                margin_len_pairs.append((margin, len(generic) / src_len))

            item_margin = statistics.mean(pair_margins)
            all_margins.append(item_margin)
            per_source.append(
                {
                    "a2_id": it.id,
                    "mean_margin": item_margin,
                    "human_wins": item_margin > 0,
                    "pair_margins": pair_margins,
                }
            )

    wins = sum(1 for row in per_source if row["human_wins"])
    margins_only = [m for m, _ in margin_len_pairs]
    len_ratios = [r for _, r in margin_len_pairs]

    report = {
        "n_test_sources": len(test_items),
        "overall_accuracy": wins / max(len(test_items), 1),
        "mean_margin": statistics.mean(all_margins) if all_margins else 0.0,
        "median_margin": statistics.median(all_margins) if all_margins else 0.0,
        "margin_length_pearson": pearson(margins_only, len_ratios),
        "margin_length_spearman": spearman(margins_only, len_ratios),
        "per_source": per_source,
    }

    out = run_dir / "evaluation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_source"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
