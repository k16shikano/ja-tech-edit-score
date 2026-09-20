#!/usr/bin/env python3
"""Evaluate pdpo_best on datasets outside the A2 P-DPO test split."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from a2_common import (
    generation_seed,
    load_config,
    load_prompt_files,
    prompt_hash,
    read_jsonl,
    revision_messages,
    setup_run_dir,
    spec_path,
    validate_generation,
)
from evaluate_preferences import FRESH_SEED_OFFSET, pearson, spearman
from model_utils import compute_sequence_logprob, load_policy_model, load_tokenizer

PARENT_ROOT = spec_path(".").parent
sys.path.insert(0, str(PARENT_ROOT / "scripts"))
from export_edit_sft import INSTRUCTION, INSTRUCTION_LEGACY, INSTRUCTION_V1  # noqa: E402

DATASET_SEED_OFFSET = {
    "a2-section-heldout": 2_000_000,
    "a1-hunk-heldout": 3_000_000,
    "b-valid-gold-vs-composer": 0,
    "c-gold-vs-adapter-selected": 4_000_000,
}
MAX_GEN_ATTEMPTS = 16


@dataclass(frozen=True)
class EvalItem:
    id: str
    draft: str
    human_revision: str
    negatives: tuple[str, ...] = ()
    meta: dict[str, Any] | None = None


def strip_edit_sft_user(user_content: str) -> str:
    for prefix in (INSTRUCTION, INSTRUCTION_V1, INSTRUCTION_LEGACY):
        if user_content.startswith(prefix):
            return user_content[len(prefix) :].lstrip("\n")
    if "\n\n" in user_content:
        return user_content.split("\n\n", 1)[1]
    return user_content


def load_a2_section_heldout() -> list[EvalItem]:
    rows = read_jsonl(PARENT_ROOT / "data/edit_sft_section/heldout.jsonl")
    items: list[EvalItem] = []
    for row in rows:
        item_id = row["meta"]["id"]
        user = row["messages"][0]["content"]
        human = row["messages"][1]["content"]
        items.append(
            EvalItem(
                id=item_id,
                draft=strip_edit_sft_user(user),
                human_revision=human,
                meta={"unit": "section", "corpus": "A2 section heldout (50)"},
            )
        )
    return items


def load_a1_hunk_heldout() -> list[EvalItem]:
    rows = read_jsonl(PARENT_ROOT / "data/edit_sft_hunk_nopara/heldout.jsonl")
    items: list[EvalItem] = []
    for row in rows:
        item_id = row.get("id") or row.get("meta", {}).get("id")
        if "messages" in row:
            user = row["messages"][0]["content"]
            human = row["messages"][1]["content"]
            draft = strip_edit_sft_user(user)
            human_revision = human
        else:
            draft = row["source_text"]
            human_revision = row["edited_text"]
        items.append(
            EvalItem(
                id=item_id,
                draft=draft,
                human_revision=human_revision,
                meta={"unit": "hunk", "corpus": "A1 hunk heldout (210)"},
            )
        )
    return items


def load_b_valid_gold_vs_composer() -> list[EvalItem]:
    rows = read_jsonl(PARENT_ROOT / "data/section_middle/pref_valid.jsonl")
    items: list[EvalItem] = []
    for row in rows:
        if row.get("pair_kind") != "gold_vs_gen":
            continue
        items.append(
            EvalItem(
                id=row["id"],
                draft=row["source_text"],
                human_revision=row["candidate_a"],
                negatives=(row["candidate_b"],),
                meta={
                    "unit": "section",
                    "corpus": "B valid (50)",
                    "pair_kind": "gold_vs_gen",
                    "label": row.get("label"),
                },
            )
        )
    return items


def load_c_gold_vs_adapter_selected() -> list[EvalItem]:
    rows = read_jsonl(PARENT_ROOT / "data/blind_eval/pairs_gold_vs_adapter_selected.jsonl")
    return [
        EvalItem(
            id=row["pair_id"],
            draft=row["context_draft"],
            human_revision=row["a_text"],
            negatives=(row["b_text"],),
            meta={
                "unit": row.get("item_id", "").split(":")[0].replace("keep-", ""),
                "corpus": "C (60)",
                "item_id": row["item_id"],
            },
        )
        for row in rows
    ]


DATASET_LOADERS = {
    "a2-section-heldout": load_a2_section_heldout,
    "a1-hunk-heldout": load_a1_hunk_heldout,
    "b-valid-gold-vs-composer": load_b_valid_gold_vs_composer,
    "c-gold-vs-adapter-selected": load_c_gold_vs_adapter_selected,
}


def s_theta(model, tokenizer, draft, completion, system, user_tpl, max_length) -> float:
    with model.disable_adapter():
        ref = compute_sequence_logprob(
            model, tokenizer, draft, completion, system, user_tpl, max_length=max_length
        )["total_logprob"]
    pol = compute_sequence_logprob(
        model, tokenizer, draft, completion, system, user_tpl, max_length=max_length
    )["total_logprob"]
    return pol - ref


def ensure_fresh_generic(
    *,
    dataset: str,
    item: EvalItem,
    sample_index: int,
    config: dict,
    idx: int,
    system: str,
    user_tpl: str,
    phash: str,
) -> str:
    out_path = spec_path(f"data/generated/external/{dataset}/{item.id}/sample-{sample_index}.json")
    legacy_path = spec_path(f"data/generated/external/section-heldout/{item.id}/sample-{sample_index}.json")
    for candidate in (out_path, legacy_path):
        if candidate.is_file():
            obj = json.loads(candidate.read_text(encoding="utf-8"))
            if obj.get("validation", {}).get("status") == "ok":
                return obj["text"]

    gen_cfg = config["generic_revision"]
    base_offset = DATASET_SEED_OFFSET.get(dataset, FRESH_SEED_OFFSET)
    messages = revision_messages(system, user_tpl, item.draft)
    backend = gen_cfg["generator"]["backend"]
    last_validation: dict | None = None
    base_sampling = dict(gen_cfg["sampling"])
    base_temp = float(base_sampling.get("temperature", 0.7))
    for attempt in range(MAX_GEN_ATTEMPTS):
        seed = generation_seed(
            int(config["experiment"]["seed"]),
            idx,
            sample_index,
            offset=base_offset + attempt * 997,
        )
        attempt_sampling = dict(base_sampling)
        attempt_sampling["temperature"] = min(1.35, base_temp + attempt * 0.1)
        if backend == "transformers":
            from generate_generic import call_transformers_chat

            text, _ = call_transformers_chat(
                gen_cfg["generator"]["model_name"],
                messages,
                attempt_sampling,
                seed,
            )
        else:
            from evaluate_preferences import call_llama_chat

            text = call_llama_chat(
                gen_cfg["generator"]["endpoint"],
                gen_cfg["generator"]["model_name"],
                messages,
                attempt_sampling,
                seed,
            )
        validation = validate_generation(text, item.draft)
        last_validation = validation
        record = {
            "item_id": item.id,
            "sample_index": sample_index,
            "seed": seed,
            "attempt": attempt,
            "prompt_hash": phash,
            "text": text.strip(),
            "validation": validation,
            "generation_metadata": {"generated_at": datetime.now(timezone.utc).isoformat()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if validation["status"] == "ok":
            return record["text"]
    flags = last_validation.get("flags") if last_validation else []
    raise RuntimeError(
        f"generic generation failed for {item.id} sample-{sample_index} after {MAX_GEN_ATTEMPTS} attempts: {flags}"
    )


def generate_fresh_negatives(
    *,
    dataset: str,
    items: list[EvalItem],
    config: dict,
    phash: str,
    k: int,
    system: str,
    user_tpl: str,
    skip_generate: bool,
) -> None:
    if skip_generate:
        return
    for idx, item in enumerate(items):
        if item.negatives:
            continue
        print(f"[{dataset}] generate {idx + 1}/{len(items)} {item.id}", flush=True)
        for sample_index in range(k):
            ensure_fresh_generic(
                dataset=dataset,
                item=item,
                sample_index=sample_index,
                config=config,
                idx=idx,
                system=system,
                user_tpl=user_tpl,
                phash=phash,
            )


def load_negatives(
    *,
    dataset: str,
    item: EvalItem,
    k: int,
) -> list[str]:
    if item.negatives:
        return list(item.negatives)
    negatives: list[str] = []
    for sample_index in range(k):
        sample_path = spec_path(
            f"data/generated/external/{dataset}/{item.id}/sample-{sample_index}.json"
        )
        legacy_path = spec_path(
            f"data/generated/external/section-heldout/{item.id}/sample-{sample_index}.json"
        )
        for candidate in (sample_path, legacy_path):
            if candidate.is_file():
                sample = json.loads(candidate.read_text(encoding="utf-8"))
                if sample.get("validation", {}).get("status") == "ok":
                    negatives.append(sample["text"])
                    break
        else:
            raise FileNotFoundError(f"missing valid generic for {item.id} sample-{sample_index}")
    return negatives


def release_generation_models() -> None:
    import generate_generic

    generate_generic._TRANSFORMERS = None
    generate_generic._LLAMA = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_dataset(
    *,
    dataset: str,
    items: list[EvalItem],
    model,
    tokenizer,
    system,
    user_tpl,
    max_length: int,
    k: int,
) -> dict:
    per_source: list[dict] = []
    all_margins: list[float] = []
    margin_len_pairs: list[tuple[float, float]] = []

    with torch.no_grad():
        for item in items:
            negatives = load_negatives(dataset=dataset, item=item, k=k)

            pair_margins: list[float] = []
            src_len = max(len(item.draft), 1)
            s_human = s_theta(
                model, tokenizer, item.draft, item.human_revision, system, user_tpl, max_length
            )
            for neg in negatives:
                margin = s_human - s_theta(model, tokenizer, item.draft, neg, system, user_tpl, max_length)
                pair_margins.append(margin)
                margin_len_pairs.append((margin, len(neg) / src_len))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            item_margin = statistics.mean(pair_margins)
            all_margins.append(item_margin)
            row = {
                "item_id": item.id,
                "mean_margin": item_margin,
                "human_wins": item_margin > 0,
                "pair_margins": pair_margins,
            }
            if item.meta:
                row["meta"] = item.meta
            per_source.append(row)

    wins = sum(1 for row in per_source if row["human_wins"])
    margins_only = [m for m, _ in margin_len_pairs]
    len_ratios = [r for _, r in margin_len_pairs]

    return {
        "dataset": dataset,
        "description": describe_dataset(dataset),
        "n_sources": len(items),
        "overall_accuracy": wins / max(len(items), 1),
        "mean_margin": statistics.mean(all_margins) if all_margins else 0.0,
        "median_margin": statistics.median(all_margins) if all_margins else 0.0,
        "margin_length_pearson": pearson(margins_only, len_ratios),
        "margin_length_spearman": spearman(margins_only, len_ratios),
        "per_source": per_source,
    }


def describe_dataset(name: str) -> str:
    descriptions = {
        "a2-section-heldout": (
            "学習用データA2の section heldout 50節。"
            "各節で人間の推敲 vs Qwen3-8B generic 3本。"
        ),
        "a1-hunk-heldout": (
            "学習用データA1の hunk heldout 210件。"
            "各 hunk で人間の推敲 vs Qwen3-8B generic 3本。"
        ),
        "b-valid-gold-vs-composer": (
            "学習用データBの valid 50節（gold_vs_gen）。"
            "人間の推敲 vs Composer の推敲（既存。再生成なし）。"
        ),
        "c-gold-vs-adapter-selected": (
            "評価用データC 60件。"
            "人間の推敲 vs 編集SFTアダプタ8本から評価器が選んだ1本（既存。再生成なし）。"
        ),
    }
    return descriptions.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/pdpo_best")
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_LOADERS.keys(), "all"],
        default="all",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("evaluate_external", config, config_path)

    names = list(DATASET_LOADERS.keys()) if args.dataset == "all" else [args.dataset]
    system, user_tpl = load_prompt_files(config)
    phash = prompt_hash(system, user_tpl)
    k = int(config["evaluation"]["fresh_generic_samples_per_test_source"])

    training = config["training"]
    model_name = training["base_model"]
    max_length = int(training.get("max_length", 2048))
    ckpt = spec_path(args.checkpoint)

    prepared: list[tuple[str, list[EvalItem], int]] = []
    for name in names:
        items = DATASET_LOADERS[name]()
        if args.limit > 0:
            items = items[: args.limit]
        if not items:
            continue
        item_k = k if not items[0].negatives else len(items[0].negatives)
        generate_fresh_negatives(
            dataset=name,
            items=items,
            config=config,
            phash=phash,
            k=item_k,
            system=system,
            user_tpl=user_tpl,
            skip_generate=args.skip_generate,
        )
        prepared.append((name, items, item_k))

    release_generation_models()

    tokenizer = load_tokenizer(model_name)
    model = load_policy_model(model_name, training["lora"], adapter_path=ckpt)
    model.eval()

    reports: list[dict] = []
    for name, items, item_k in prepared:
        report = evaluate_dataset(
            dataset=name,
            items=items,
            model=model,
            tokenizer=tokenizer,
            system=system,
            user_tpl=user_tpl,
            max_length=max_length,
            k=item_k,
        )
        reports.append(report)
        summary = {key: report[key] for key in report if key != "per_source"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = run_dir / "external_evaluation.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
