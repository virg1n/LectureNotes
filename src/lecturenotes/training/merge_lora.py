"""Merge a LoRA adapter into its base model (for vLLM serving without LoRA support).

    python -m lecturenotes.training.merge_lora --base Qwen/Qwen3-4B-Instruct-2507 --adapter checkpoints/sft \
        --out checkpoints/sft-merged
"""

from __future__ import annotations

import argparse


def merge(base: str, adapter: str, out: str, dtype: str = "fp16") -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)  # merge in fp32, then cast
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(td).save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter if _has_tok(adapter) else base).save_pretrained(out)


def _has_tok(path: str) -> bool:
    from pathlib import Path

    return (Path(path) / "tokenizer_config.json").exists()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp16")
    a = ap.parse_args(argv)
    merge(a.base, a.adapter, a.out, a.dtype)
    print(f"merged -> {a.out}")


if __name__ == "__main__":
    main()
