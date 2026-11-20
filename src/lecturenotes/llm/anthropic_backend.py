"""Claude backend -- optional teacher for data generation and LLM-as-judge for evaluation.

Notes
* Uses the official ``anthropic`` SDK with streaming (long outputs) and
  ``get_final_message()``.
* Current Claude models reject sampling parameters (temperature/top_p): diversity
  across samples comes from the prompt, so ``GenParams.temperature`` is ignored.
* Server-side refusal fallbacks are enabled (``fallbacks="default"``); a final
  ``stop_reason == "refusal"`` is surfaced as an empty completion.
* Check the provider's terms before using outputs as training data for a model.
"""

from __future__ import annotations

import asyncio
import os
import time

from .base import Completion, GenParams, LLMBackend, Message


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", *, effort: str = "high", max_concurrency: int = 8,
                 use_fallbacks: bool = True):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'lecturenotes[teacher]' to use the Anthropic backend") from e
        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.effort = effort
        self.sem = asyncio.Semaphore(max_concurrency)
        self.use_fallbacks = use_fallbacks

    async def _one(self, system: str | None, msgs: list[Message], p: GenParams) -> Completion:
        kwargs = dict(model=self.model, max_tokens=max(p.max_tokens, 4096), messages=msgs,
                      output_config={"effort": self.effort})
        if system:
            kwargs["system"] = system
        if p.stop:
            kwargs["stop_sequences"] = p.stop
        if self.use_fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        t0 = time.monotonic()
        msg = None
        async with self.sem:
            for attempt in range(5):
                try:
                    if self.use_fallbacks:
                        stream_cm = self.client.beta.messages.stream(**kwargs)
                    else:
                        stream_cm = self.client.messages.stream(**kwargs)
                    async with stream_cm as stream:
                        msg = await stream.get_final_message()
                    break
                except self._anthropic.RateLimitError as e:
                    retry = float(e.response.headers.get("retry-after", "20"))
                    await asyncio.sleep(retry)
                except self._anthropic.APIStatusError as e:
                    if e.status_code >= 500 and attempt < 4:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    raise
                except self._anthropic.APIConnectionError:
                    if attempt < 4:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    raise
        if msg is None:
            raise RuntimeError("Anthropic request still rate-limited after 5 attempts")
        if msg.stop_reason == "refusal":
            return Completion("", "refusal", msg.usage.input_tokens, msg.usage.output_tokens,
                              time.monotonic() - t0, msg.model)
        text = "".join(b.text for b in msg.content if b.type == "text")
        finish = "length" if msg.stop_reason == "max_tokens" else "stop"
        return Completion(text, finish, msg.usage.input_tokens, msg.usage.output_tokens,
                          time.monotonic() - t0, msg.model)

    async def agenerate(self, messages: list[Message], params: GenParams) -> list[Completion]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        msgs = [m for m in messages if m["role"] != "system"]
        return list(await asyncio.gather(*(self._one(system, msgs, params) for _ in range(params.n))))


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
