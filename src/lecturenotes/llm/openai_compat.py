"""OpenAI-compatible chat backend: vLLM, SGLang, llama.cpp server, TGI, LM Studio, OpenRouter..."""

from __future__ import annotations

import asyncio
import os
import random
import time

import httpx

from .base import Completion, GenParams, LLMBackend, Message, strip_reasoning


class OpenAICompatBackend(LLMBackend):
    name = "openai"

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "lecturenotes", *,
                 api_key: str | None = None, timeout: float = 600.0, max_concurrency: int = 16,
                 json_mode: str = "response_format", extra_body: dict | None = None,
                 disable_thinking: bool = True, max_retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("LN_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout = timeout
        self.sem = asyncio.Semaphore(max_concurrency)
        self.json_mode = json_mode  # "response_format" | "guided_json" | "none"
        self.extra_body = dict(extra_body or {})
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=30.0),
                                             headers={"Authorization": f"Bearer {self.api_key}"})
        return self._client

    def _payload(self, messages: list[Message], p: GenParams) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": p.temperature, "top_p": p.top_p,
                "max_tokens": p.max_tokens, "n": p.n}
        if p.stop:
            body["stop"] = p.stop
        if p.seed is not None:
            body["seed"] = p.seed
        if p.json_schema is not None:
            if self.json_mode == "response_format":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "output", "schema": p.json_schema}}
            elif self.json_mode == "guided_json":
                body["guided_json"] = p.json_schema
        if self.disable_thinking:
            # Qwen3-style hybrid models: request the non-thinking chat template (ignored elsewhere)
            body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        body.update(self.extra_body)
        body.update(p.extra)
        return body

    async def agenerate(self, messages: list[Message], params: GenParams) -> list[Completion]:
        body = self._payload(messages, params)
        url = f"{self.base_url}/chat/completions"
        delay = 2.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            async with self.sem:
                t0 = time.monotonic()
                try:
                    r = await self._get_client().post(url, json=body)
                    if r.status_code == 400 and "chat_template_kwargs" in body and "chat_template_kwargs" in r.text:
                        body.pop("chat_template_kwargs")  # server does not support it
                        continue
                    if r.status_code in (408, 409, 429) or r.status_code >= 500:
                        raise httpx.HTTPStatusError(f"server returned {r.status_code}", request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    lat = time.monotonic() - t0
                    usage = data.get("usage") or {}
                    out = []
                    for ch in data["choices"]:
                        msg = ch.get("message") or {}
                        text = msg.get("content") or ""
                        out.append(Completion(
                            text=strip_reasoning(text), finish_reason=ch.get("finish_reason") or "stop",
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0) // max(len(data["choices"]), 1),
                            latency=lat, model=data.get("model", self.model)))
                    return out
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    last_err = e
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                        raise
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 60)
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last_err}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model, "base_url": self.base_url}
