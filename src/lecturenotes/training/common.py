"""Shared training utilities (tokenization with assistant-only labels, model loading, V100 specifics).

V100 (sm_70) notes
* no bfloat16 -> fp16 autocast + loss scaling; LoRA adapters are kept in fp32
* no FlashAttention-2 -> PyTorch SDPA (memory-efficient kernel works on sm_70)
* long LaTeX targets -> gradient checkpointing; examples longer than max_len are
  *dropped*, never truncated (a truncated target teaches unclosed environments)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model: str
    train_file: str
    val_file: str | None = None
    output_dir: str = "checkpoints/sft"
    max_len: int = 8192
    # optimisation
    lr: float = 1e-4
    epochs: float = 2.0
    per_device_batch: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    seed: int = 42
    # precision / memory
    dtype: str = "fp16"  # fp16 on V100; bf16 on Ampere+
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    deepspeed: str | None = None
    # LoRA (set use_lora: false for full fine-tuning)
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                              "up_proj", "down_proj"])
    init_adapter: str | None = None
    # data
    task_weights: dict[str, float] = field(default_factory=dict)  # oversampling factors per task
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    report_to: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "TrainConfig":
        from ..config import _expand_env

        data = _expand_env(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.extra.update(extra)
        return cfg


def read_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------------------- tokenization

def _chat_ids(tok, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    kw = {}
    try:  # hybrid-thinking templates (Qwen3): train the non-thinking format
        return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                       enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kw)


def encode_chat(tok, messages: list[dict], max_len: int) -> dict | None:
    """input_ids + labels where only the final assistant turn is supervised."""
    prompt = _chat_ids(tok, messages[:-1], add_generation_prompt=True)
    full = _chat_ids(tok, messages, add_generation_prompt=False)
    if hasattr(prompt, "input_ids"):
        prompt, full = prompt["input_ids"], full["input_ids"]
    if full[: len(prompt)] != prompt:
        # template re-renders history differently; fall back to explicit concatenation
        eos = tok.eos_token or ""
        completion = tok(messages[-1]["content"] + eos, add_special_tokens=False)["input_ids"]
        full = list(prompt) + completion
    if len(full) > max_len:
        return None
    labels = [-100] * len(prompt) + list(full[len(prompt):])
    return {"input_ids": list(full), "labels": labels, "length": len(full)}


def build_dataset(tok, rows: list[dict], max_len: int, task_weights: dict[str, float] | None = None,
                  seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out, dropped = [], 0
    for r in rows:
        enc = encode_chat(tok, r["messages"], max_len)
        if enc is None:
            dropped += 1
            continue
        w = (task_weights or {}).get(r.get("task", ""), 1.0)
        reps = int(w) + (1 if rng.random() < (w - int(w)) else 0)
        for _ in range(max(reps, 0)):
            out.append(enc)
    if dropped:
        log.warning("dropped %d/%d examples longer than %d tokens", dropped, len(rows), max_len)
    rng.shuffle(out)
    return out


@dataclass
class PadCollator:
    pad_id: int
    multiple: int = 8

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        n = int(math.ceil(n / self.multiple) * self.multiple)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ----------------------------------------------------------------------------- model

def torch_dtype(name: str):
    import torch

    return {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp32": torch.float32, "float32": torch.float32}[name]


def supports_bf16() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def patch_sdpa_gqa_for_old_gpus() -> bool:
    """On sm < 80 (V100) the only fused SDPA kernel is memory-efficient attention, which rejects
    GQA-shaped K/V (enable_gqa).  transformers passes unexpanded K/V when the mask is None, so torch
    falls back to the *math* kernel and materialises fp32 (heads x L x L) scores: 8.6 GB at 8k tokens
    per layer -> OOM.  Forcing transformers to repeat K/V first keeps the memory-efficient kernel."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        return False
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:
        return False
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
    return True


def load_model_and_tokenizer(cfg: TrainConfig, for_training: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.attn_implementation == "sdpa" and patch_sdpa_gqa_for_old_gpus():
        log.info("sm<80: expanding GQA K/V before SDPA so the memory-efficient kernel is used")

    if cfg.dtype == "bf16" and not supports_bf16():
        log.warning("bf16 requested but the GPU does not support it (V100?) -> using fp16")
        cfg.dtype = "fp16"
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LoRA: frozen base in half precision. Full FT: fp32 master weights (AMP/DeepSpeed handles fp16 compute).
    load_dtype = torch_dtype(cfg.dtype) if cfg.use_lora else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=load_dtype,
                                                 attn_implementation=cfg.attn_implementation)
    model.config.use_cache = not for_training
    if for_training and cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.use_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if for_training:
            model.enable_input_require_grads()
        if cfg.init_adapter:
            model = PeftModel.from_pretrained(model, cfg.init_adapter, is_trainable=for_training)
        else:
            targets = [t for t in cfg.lora_targets if any(n.endswith(t) for n, _ in model.named_modules())]
            if not targets:  # unusual architectures (fused qkv, e.g. Phi-3/4): adapt all linear layers
                targets = "all-linear"
            model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                                     lora_dropout=cfg.lora_dropout, target_modules=targets,
                                                     task_type="CAUSAL_LM"))
        # keep trainable (adapter) params in fp32 for stable fp16 training
        for n, p in model.named_parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        if for_training:
            model.print_trainable_parameters()
    return model, tok


def disable_lora_dropout(model) -> int:
    """Set every LoRA dropout to p=0 (RL needs identical behaviour/training forwards)."""
    import torch

    n = 0
    for name, mod in model.named_modules():
        if "lora_dropout" in name and isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
            n += 1
    return n


def warn_if_reference_is_raw_base(cfg: "TrainConfig") -> None:
    """DPO/GRPO use `disable_adapter()` as the reference policy. With `init_adapter` set, that
    reference is the *raw base* (the SFT adapter is disabled too), so the KL / DPO term pulls the
    policy back toward the un-fine-tuned model. Train on the merged SFT model with a fresh LoRA."""
    if cfg.init_adapter:
        log.warning("init_adapter=%s: the reference policy will be the raw base model, not the SFT model. "
                    "Merge the SFT adapter (training/merge_lora.py) and train a fresh LoRA on top instead.",
                    cfg.init_adapter)


def world_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local
