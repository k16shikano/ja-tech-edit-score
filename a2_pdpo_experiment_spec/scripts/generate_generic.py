#!/usr/bin/env python3
"""Phase 2: generic revision generation via llama.cpp OpenAI-compatible API."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from a2_common import (
    build_user_content,
    generated_sample_path,
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


_LLAMA = None
_TRANSFORMERS = None


def get_transformers_model(model_name: str):
    global _TRANSFORMERS
    if _TRANSFORMERS is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _TRANSFORMERS = (model, tokenizer)
    return _TRANSFORMERS


def call_transformers_chat(
    model_name: str,
    messages: list[dict[str, str]],
    sampling: dict[str, Any],
    seed: int,
) -> tuple[str, dict]:
    import torch

    from model_utils import apply_chat_template

    model, tokenizer = get_transformers_model(model_name)
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    max_new = min(int(sampling["max_tokens"]), 4096)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, {"model": model_name, "max_new_tokens": max_new}


def get_embedded_llama(model_path: str):
    global _LLAMA
    if _LLAMA is None:
        from llama_cpp import Llama

        _LLAMA = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )
    return _LLAMA


def call_llama_embedded(
    model_path: str,
    messages: list[dict[str, str]],
    sampling: dict[str, Any],
    seed: int,
) -> tuple[str, dict]:
    llm = get_embedded_llama(model_path)
    payload = llm.create_chat_completion(
        messages=messages,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        seed=seed,
        max_tokens=sampling["max_tokens"],
        repeat_penalty=sampling.get("repeat_penalty", 1.0),
    )
    text = payload["choices"][0]["message"]["content"]
    return text, payload


def call_llama_chat(
    endpoint: str,
    model_name: str,
    messages: list[dict[str, str]],
    sampling: dict[str, Any],
    seed: int,
) -> tuple[str, dict]:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model_name,
        "messages": messages,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "seed": seed,
        "max_tokens": sampling["max_tokens"],
    }
    if sampling.get("top_k") is not None:
        body["top_k"] = sampling["top_k"]
    if sampling.get("repeat_penalty") is not None:
        body["repeat_penalty"] = sampling["repeat_penalty"]

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    text = payload["choices"][0]["message"]["content"]
    return text, payload


def next_attempt_path(base: Path) -> Path:
    if not base.exists():
        return base
    parent = base.parent
    stem = base.stem
    n = 1
    while True:
        candidate = parent / f"{stem}.attempt-{n}.json"
        if not candidate.exists():
            return candidate
        n += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    run_dir = setup_run_dir("generate_generic", config, config_path)

    system, user_tpl = load_prompt_files(config)
    phash = prompt_hash(system, user_tpl)
    gen_cfg = config["generic_revision"]
    sampling = gen_cfg["sampling"]
    backend = gen_cfg["generator"]["backend"]
    model_gguf = gen_cfg["generator"].get("model_gguf", "")
    generator_meta = {
        "backend": backend,
        "model_name": gen_cfg["generator"]["model_name"],
        "quantization": gen_cfg["generator"]["quantization"],
        "endpoint": gen_cfg["generator"].get("endpoint"),
        "model_gguf": model_gguf or None,
        "sampling": sampling,
    }

    items = load_a2_items()
    manifest = load_split_manifest()
    split_items = items_for_split(items, manifest, args.split)
    if args.limit > 0:
        split_items = split_items[: args.limit]

    idx_map = item_index_map(items)
    experiment_seed = int(config["experiment"]["seed"])
    k = int(gen_cfg["samples_per_source"])

    stats = {"created": 0, "skipped_existing": 0, "attempts": 0, "failed_validation": 0}

    for i, it in enumerate(split_items, start=1):
        item_index = idx_map[it.id]
        print(f"[{args.split}] item {i}/{len(split_items)} {it.id}", flush=True)
        for sample_index in range(k):
            out_path = generated_sample_path(args.split, it.id, sample_index)
            if out_path.exists():
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if existing.get("validation", {}).get("status") == "ok":
                    stats["skipped_existing"] += 1
                    continue

            seed = generation_seed(experiment_seed, item_index, sample_index)
            messages = revision_messages(system, user_tpl, it.draft)
            if args.dry_run:
                print(f"would generate {it.id} sample-{sample_index} seed={seed}")
                continue

            stats["attempts"] += 1
            try:
                if backend == "transformers":
                    text, raw = call_transformers_chat(
                        gen_cfg["generator"]["model_name"],
                        messages,
                        sampling,
                        seed,
                    )
                elif backend == "llama.cpp_embedded":
                    if not model_gguf:
                        raise ValueError("model_gguf is required for llama.cpp_embedded")
                    text, raw = call_llama_embedded(model_gguf, messages, sampling, seed)
                else:
                    text, raw = call_llama_chat(
                        gen_cfg["generator"]["endpoint"],
                        gen_cfg["generator"]["model_name"],
                        messages,
                        sampling,
                        seed,
                    )
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                fail_path = next_attempt_path(out_path)
                record = {
                    "a2_id": it.id,
                    "sample_index": sample_index,
                    "seed": seed,
                    "generator": generator_meta,
                    "prompt_hash": phash,
                    "text": "",
                    "validation": {"status": "failed", "flags": ["request_error"]},
                    "generation_metadata": {
                        "error": str(exc),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                fail_path.parent.mkdir(parents=True, exist_ok=True)
                fail_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                stats["failed_validation"] += 1
                continue

            validation = validate_generation(text, it.draft)
            record = {
                "a2_id": it.id,
                "sample_index": sample_index,
                "seed": seed,
                "generator": generator_meta,
                "prompt_hash": phash,
                "text": text.strip(),
                "validation": validation,
                "generation_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_user_preview": build_user_content(user_tpl, it.draft)[:200],
                    "raw_response_id": raw.get("id"),
                },
            }

            target = out_path if validation["status"] == "ok" else next_attempt_path(out_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            if validation["status"] == "ok":
                if target != out_path:
                    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                stats["created"] += 1
            else:
                stats["failed_validation"] += 1

    (run_dir / "generation_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
