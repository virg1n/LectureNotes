"""In-process transformers backend (debugging, CPU/GPU without a server, LoRA adapters)."""

from __future__ import annotations

import asyncio
import threading
import time

from .base import Completion, GenParams, LLMBackend, Message, strip_reasoning


class HFLocalBackend(LLMBackend):
    name = "hf"

    def __init__(self, model_path: str, *, adapter_path: str | None = None, dtype: str = "auto",
                 device_map: str = "auto", attn_implementation: str = "sdpa", disable_thinking: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model_path
        self.tok = AutoTokenizer.from_pretrained(model_path)
        torch_dtype = {"auto": "auto", "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16,
                       "bfloat16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.lm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map=device_map,
                                                       attn_implementation=attn_implementation)
        if adapter_path:
            from peft import PeftModel

            self.lm = PeftModel.from_pretrained(self.lm, adapter_path)
        self.lm.eval()
        self.disable_thinking = disable_thinking
        self._lock = threading.Lock()

    def _generate_sync(self, messages: list[Message], p: GenParams) -> list[Completion]:
        import torch

        kw = {"enable_thinking": False} if self.disable_thinking else {}
        try:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.lm.device)
        outs = []
        with self._lock, torch.no_grad():
            for i in range(p.n):
                if p.seed is not None:
                    torch.manual_seed(p.seed + i)
                t0 = time.monotonic()
                gen = self.lm.generate(**enc, max_new_tokens=p.max_tokens, do_sample=p.temperature > 0,
                                       temperature=max(p.temperature, 1e-4), top_p=p.top_p,
                                       pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
                new = gen[0, enc["input_ids"].shape[1]:]
                text = self.tok.decode(new, skip_special_tokens=True)
                if p.stop:
                    for s in p.stop:
                        text = text.split(s)[0]
                outs.append(Completion(strip_reasoning(text),
                                       "length" if len(new) >= p.max_tokens else "stop",
                                       enc["input_ids"].shape[1], len(new), time.monotonic() - t0, self.model))
        return outs

    async def agenerate(self, messages: list[Message], params: GenParams) -> list[Completion]:
        return await asyncio.to_thread(self._generate_sync, messages, params)
