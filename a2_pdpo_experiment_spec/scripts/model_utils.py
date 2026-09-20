"""Model loading and sequence log-probability helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from a2_common import load_prompt_files, revision_messages, spec_path


def apply_chat_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, *, bf16: bool = True):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if bf16 and torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def _transformer_and_lm_head(model):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(inner, "model") and hasattr(inner.model, "model"):
        return inner.model.model, inner.model.lm_head
    if hasattr(inner, "model") and hasattr(inner.model, "lm_head"):
        return inner.model, inner.model.lm_head
    if hasattr(inner, "model"):
        return inner.model, inner.lm_head
    raise TypeError("unsupported model type for logprob")


def load_policy_model(model_name: str, lora_cfg: dict[str, Any], adapter_path: Path | None = None):
    from peft import LoraConfig, PeftModel, get_peft_model

    model = load_base_model(model_name)
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    if adapter_path is not None and adapter_path.is_dir():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    transformer, _ = _transformer_and_lm_head(model)
    transformer.config.use_cache = False
    if hasattr(transformer, "gradient_checkpointing_enable"):
        transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def compute_sequence_logprob(
    model,
    tokenizer,
    draft: str,
    completion: str,
    system: str,
    user_template: str,
    *,
    max_length: int,
    as_tensor: bool = False,
) -> dict[str, float | torch.Tensor]:
    prompt_messages = revision_messages(system, user_template, draft)
    full_messages = revision_messages(system, user_template, draft, completion)

    prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids

    if full_ids.shape[1] > max_length:
        prompt_len = prompt_ids.shape[1]
        if prompt_len >= max_length:
            full_ids = full_ids[:, -max_length:]
            prompt_len = min(prompt_ids.shape[1], max_length - 1)
        else:
            overflow = full_ids.shape[1] - max_length
            full_ids = torch.cat(
                [full_ids[:, :prompt_len], full_ids[:, prompt_len + overflow :]],
                dim=1,
            )
    else:
        prompt_len = prompt_ids.shape[1]

    if prompt_len >= full_ids.shape[1]:
        zero = {
            "total_logprob": float("-inf"),
            "completion_token_count": 0.0,
            "mean_token_logprob": float("-inf"),
        }
        if as_tensor:
            device = next(model.parameters()).device
            zero["total_logprob"] = torch.tensor(float("-inf"), device=device)
            zero["mean_token_logprob"] = torch.tensor(float("-inf"), device=device)
        return zero

    device = next(model.parameters()).device
    input_ids = full_ids.to(device)
    seq_len = input_ids.shape[1]
    completion_len = seq_len - prompt_len
    transformer, lm_head = _transformer_and_lm_head(model)
    transformer.config.use_cache = False

    def _forward_hidden(ids: torch.Tensor) -> torch.Tensor:
        return transformer(input_ids=ids, use_cache=False).last_hidden_state

    if as_tensor and model.training:
        hidden = checkpoint(_forward_hidden, input_ids, use_reentrant=False)
    else:
        hidden = _forward_hidden(input_ids)
    total_t = torch.zeros((), device=device, dtype=hidden.dtype)
    token_chunk = 8
    for start in range(prompt_len - 1, seq_len - 1, token_chunk):
        end = min(start + token_chunk, seq_len - 1)
        logits = lm_head(hidden[:, start:end, :])
        targets = input_ids[:, start + 1 : end + 1]
        total_t = total_t - F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        del logits
    del hidden
    count = completion_len
    mean_t = total_t / max(count, 1)
    if as_tensor:
        return {
            "total_logprob": total_t,
            "completion_token_count": float(count),
            "mean_token_logprob": mean_t,
        }
    total = float(total_t.detach().cpu())
    mean = float(mean_t.detach().cpu())
    return {
        "total_logprob": total,
        "completion_token_count": float(count),
        "mean_token_logprob": mean,
    }


def load_prompts_from_config(config: dict[str, Any]) -> tuple[str, str]:
    return load_prompt_files(config)


def ref_logp_key(a2_id: str, split: str, side: str, sample_index: int | None = None) -> str:
    if side == "chosen":
        return f"{split}|{a2_id}|chosen"
    return f"{split}|{a2_id}|rejected|{sample_index}"


def preference_record_key(rec: dict[str, Any]) -> str:
    return ref_logp_key(
        rec["a2_id"],
        rec["split"],
        "rejected" if rec.get("rejected_sample_index") is not None else "chosen",
        rec.get("rejected_sample_index"),
    )
