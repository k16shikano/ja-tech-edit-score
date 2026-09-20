#!/usr/bin/env python3
"""Phase 6: P-DPO training with frozen reference log-probs."""
from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from a2_common import (
    group_preferences_by_source,
    load_config,
    read_jsonl,
    set_seeds,
    setup_run_dir,
    spec_path,
)
from model_utils import (
    compute_sequence_logprob,
    load_policy_model,
    load_prompts_from_config,
    load_tokenizer,
    ref_logp_key,
)


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def make_optimizer(model, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=lr)
    except Exception:
        return torch.optim.AdamW(params, lr=lr)


def load_reference_map(split: str) -> dict[str, dict]:
    path = spec_path(f"data/reference_logps/{split}.jsonl")
    rows = read_jsonl(path)
    return {row["key"]: row for row in rows}


def s_theta(model, tokenizer, draft, completion, system, user_tpl, max_length) -> float:
    with model.disable_adapter():
        ref = compute_sequence_logprob(
            model, tokenizer, draft, completion, system, user_tpl, max_length=max_length
        )["total_logprob"]
    pol = compute_sequence_logprob(
        model, tokenizer, draft, completion, system, user_tpl, max_length=max_length
    )["total_logprob"]
    return pol - ref


def evaluate_dev(
    model,
    tokenizer,
    system,
    user_tpl,
    prefs,
    ref_map,
    *,
    max_length: int,
    beta: float,
) -> dict:
    grouped = group_preferences_by_source(prefs)
    wins = 0
    margins: list[float] = []

    model.eval()
    with torch.no_grad():
        for a2_id, group in grouped.items():
            pair_margins: list[float] = []
            for rec in group:
                ck = ref_logp_key(a2_id, rec["split"], "chosen")
                rk = ref_logp_key(
                    a2_id,
                    rec["split"],
                    "rejected",
                    rec["rejected_sample_index"],
                )
                pol_ch = compute_sequence_logprob(
                    model,
                    tokenizer,
                    rec["draft"],
                    rec["chosen"],
                    system,
                    user_tpl,
                    max_length=max_length,
                )["total_logprob"]
                pol_re = compute_sequence_logprob(
                    model,
                    tokenizer,
                    rec["draft"],
                    rec["rejected"],
                    system,
                    user_tpl,
                    max_length=max_length,
                )["total_logprob"]
                margin = (pol_ch - ref_map[ck]["total_logprob"]) - (
                    pol_re - ref_map[rk]["total_logprob"]
                )
                pair_margins.append(margin)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            item_margin = sum(pair_margins) / len(pair_margins)
            margins.append(item_margin)
            if item_margin > 0:
                wins += 1

    n = len(grouped)
    return {
        "accuracy": wins / max(n, 1),
        "mean_margin": sum(margins) / max(len(margins), 1),
        "median_margin": sorted(margins)[len(margins) // 2] if margins else 0.0,
        "n_sources": n,
        "beta": beta,
    }


def train_one_beta(
    *,
    config,
    train_prefs,
    dev_prefs,
    ref_train,
    ref_dev,
    beta: float,
    out_dir: Path,
    args,
) -> dict:
    training = config["training"]
    model_name = training["base_model"]
    max_length = int(args.max_length)
    system, user_tpl = load_prompts_from_config(config)

    tokenizer = load_tokenizer(model_name)
    model = load_policy_model(model_name, training["lora"])
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    grouped_train = group_preferences_by_source(train_prefs)
    source_ids = list(grouped_train.keys())

    optimizer = make_optimizer(model, args.learning_rate)

    best = {"accuracy": -1.0, "epoch": -1, "path": ""}
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        release_cuda()
        epoch_loss = 0.0
        perm = source_ids[:]
        import random

        random.shuffle(perm)

        for start in range(0, len(perm), args.items_per_step):
            batch_ids = perm[start : start + args.items_per_step]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            n_items = 0

            for a2_id in batch_ids:
                group = grouped_train[a2_id]
                n_pairs = len(group)
                item_loss = 0.0
                for rec in group:
                    ck = ref_logp_key(a2_id, rec["split"], "chosen")
                    rk = ref_logp_key(
                        a2_id,
                        rec["split"],
                        "rejected",
                        rec["rejected_sample_index"],
                    )
                    ref_ch = ref_train[ck]["total_logprob"]
                    ref_re = ref_train[rk]["total_logprob"]
                    with torch.no_grad():
                        pol_re_val = compute_sequence_logprob(
                            model,
                            tokenizer,
                            rec["draft"],
                            rec["rejected"],
                            system,
                            user_tpl,
                            max_length=max_length,
                            as_tensor=True,
                        )["total_logprob"]
                    s_re_const = pol_re_val - ref_re
                    pol_ch = compute_sequence_logprob(
                        model,
                        tokenizer,
                        rec["draft"],
                        rec["chosen"],
                        system,
                        user_tpl,
                        max_length=max_length,
                        as_tensor=True,
                    )["total_logprob"]
                    s_ch = pol_ch - ref_ch
                    u = beta * (s_ch - s_re_const)
                    loss_ch = -F.logsigmoid(u)
                    if not math.isfinite(float(loss_ch.detach().cpu())):
                        del pol_ch, pol_re_val, loss_ch
                        continue
                    (loss_ch / n_pairs).backward()
                    s_ch_det = s_ch.detach()
                    item_loss += float(loss_ch.detach().cpu())
                    del pol_ch, s_ch, u, loss_ch, pol_re_val, s_re_const
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pol_re = compute_sequence_logprob(
                        model,
                        tokenizer,
                        rec["draft"],
                        rec["rejected"],
                        system,
                        user_tpl,
                        max_length=max_length,
                        as_tensor=True,
                    )["total_logprob"]
                    s_re = pol_re - ref_re
                    u = beta * (s_ch_det - s_re)
                    loss_re = -F.logsigmoid(u)
                    if not math.isfinite(float(loss_re.detach().cpu())):
                        del pol_re, loss_re
                        continue
                    (loss_re / n_pairs).backward()
                    del pol_re, s_re, u, loss_re, s_ch_det
                    release_cuda()
                n_items += 1
                batch_loss += item_loss / max(n_pairs, 1)

            if n_items == 0:
                continue
            optimizer.step()
            epoch_loss += batch_loss / max(n_items, 1)
            release_cuda()

        dev_metrics = evaluate_dev(
            model,
            tokenizer,
            system,
            user_tpl,
            dev_prefs,
            ref_dev,
            max_length=max_length,
            beta=beta,
        )
        model.train()
        release_cuda()
        dev_metrics["epoch"] = epoch
        dev_metrics["train_loss"] = epoch_loss / max(len(perm) / args.items_per_step, 1)
        history.append(dev_metrics)
        print(json.dumps(dev_metrics, ensure_ascii=False))

        ckpt_dir = out_dir / f"beta{beta}" / f"epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        if dev_metrics["accuracy"] >= best["accuracy"]:
            best = {
                "accuracy": dev_metrics["accuracy"],
                "epoch": epoch,
                "path": str(ckpt_dir),
                "beta": beta,
                "mean_margin": dev_metrics["mean_margin"],
            }

    return {"beta": beta, "best": best, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--items-per-step", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="smoke: limit train preference rows")
    args = parser.parse_args()

    config_path = spec_path(args.config)
    config = load_config(config_path)
    if args.max_length <= 0:
        args.max_length = int(config["training"].get("max_length", 4096))
    run_dir = setup_run_dir("train_pdpo", config, config_path)
    set_seeds(int(config["experiment"]["seed"]))

    if not config.get("training_runs", {}).get("run_pdpo", True):
        raise SystemExit("training_runs.run_pdpo is false")

    train_prefs = read_jsonl(spec_path("data/preferences/train.jsonl"))
    dev_prefs = read_jsonl(spec_path("data/preferences/dev.jsonl"))
    if args.limit > 0:
        grouped = group_preferences_by_source(train_prefs)
        keep = list(grouped.keys())[: max(1, args.limit // 3)]
        train_prefs = [r for r in train_prefs if r["a2_id"] in keep]

    ref_train = load_reference_map("train")
    ref_dev = load_reference_map("dev")

    betas = config["training"]["pdpo"]["beta_values"]
    results = []
    for beta in betas:
        beta = float(beta)
        beta_dir = run_dir / f"beta_{beta}"
        beta_dir.mkdir(parents=True, exist_ok=True)
        result = train_one_beta(
            config=config,
            train_prefs=train_prefs,
            dev_prefs=dev_prefs,
            ref_train=ref_train,
            ref_dev=ref_dev,
            beta=beta,
            out_dir=beta_dir,
            args=args,
        )
        results.append(result)

    best_overall = max((r["best"] for r in results), key=lambda x: (x["accuracy"], x["mean_margin"]))
    final_dir = spec_path("checkpoints/pdpo_best")
    final_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    src = Path(best_overall["path"])
    for name in src.iterdir():
        dest = final_dir / name.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if name.is_dir():
            shutil.copytree(name, dest)
        else:
            shutil.copy2(name, dest)

    summary = {"betas": results, "selected": best_overall, "checkpoint_dir": str(final_dir)}
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
