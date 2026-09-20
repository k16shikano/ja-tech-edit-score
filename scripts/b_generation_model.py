#!/usr/bin/env python3
"""B 生成実験用の QLoRA モデル読み込みと optimizer 構築。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
  return [p for p in model.parameters() if p.requires_grad]


@contextmanager
def bf16_autocast(device: torch.device):
  if device.type == "cuda":
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      yield
  else:
    yield


def forward_causal_loss(
  model: torch.nn.Module,
  *,
  input_ids: torch.Tensor,
  attention_mask: torch.Tensor,
  labels: torch.Tensor,
  prompt_tokens: int,
) -> torch.Tensor:
  """prompt 以降だけ lm_head を計算し、bf16 autocast で attention を走らせる。"""
  labels_tail = labels[:, prompt_tokens:]
  logits_to_keep = int(labels_tail.shape[1])
  with bf16_autocast(input_ids.device):
    out = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      labels=labels_tail,
      logits_to_keep=logits_to_keep,
    )
  if out.loss is None:
    raise RuntimeError("model returned no loss")
  return out.loss


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
  params = trainable_parameters(model)
  if not params:
    raise RuntimeError("no trainable parameters")
  lr = float(cfg["learning_rate"])
  weight_decay = float(cfg["weight_decay"])
  betas = (float(cfg["adam_beta1"]), float(cfg["adam_beta2"]))
  eps = float(cfg["adam_epsilon"])
  if cfg.get("optimizer_8bit", True):
    import bitsandbytes as bnb

    return bnb.optim.AdamW8bit(
      params,
      lr=lr,
      betas=betas,
      eps=eps,
      weight_decay=weight_decay,
    )
  return torch.optim.AdamW(
    params,
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
  )


def _load_quantized_base(cfg: dict, *, device: torch.device, for_training: bool):
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

  model_id = cfg["model_id"]
  revision = cfg["model_revision"]
  tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
  qcfg = cfg["quantization"]
  bnb = BitsAndBytesConfig(
    load_in_4bit=bool(qcfg["load_in_4bit"]),
    bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=bool(qcfg["bnb_4bit_use_double_quant"]),
    bnb_4bit_compute_dtype=torch.bfloat16,
  )
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    quantization_config=bnb,
    device_map={"": device.index if device.index is not None else 0},
    attn_implementation=cfg.get("attn_implementation", "sdpa"),
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
  )
  if for_training:
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
      model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
      )
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
      model, use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True))
    )
    if hasattr(model, "enable_input_require_grads"):
      model.enable_input_require_grads()
  else:
    model.config.use_cache = True
  return model, tokenizer


def load_model_and_tokenizer(cfg: dict, *, device: torch.device):
  from peft import LoraConfig, get_peft_model

  model, tokenizer = _load_quantized_base(cfg, device=device, for_training=True)
  lora_cfg = cfg["lora"]
  peft_config = LoraConfig(
    r=int(lora_cfg["r"]),
    lora_alpha=int(lora_cfg["lora_alpha"]),
    lora_dropout=float(lora_cfg["lora_dropout"]),
    bias=lora_cfg["bias"],
    task_type=lora_cfg["task_type"],
    target_modules=list(lora_cfg["target_modules"]),
  )
  model = get_peft_model(model, peft_config)
  return model, tokenizer


def load_model_for_inference(cfg: dict, *, device: torch.device, adapter_path: Path | None = None):
  """推論用。adapter_path が None ならベースモデルのみ（xy-base）。"""
  from peft import PeftModel

  model, tokenizer = _load_quantized_base(cfg, device=device, for_training=False)
  if adapter_path is not None:
    model = PeftModel.from_pretrained(model, str(adapter_path))
  model.eval()
  return model, tokenizer
