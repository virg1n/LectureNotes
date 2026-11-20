"""Backend-agnostic LLM interface."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


@dataclass
class GenParams:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 4096
    n: int = 1
    stop: list[str] | None = None
    seed: int | None = None
    json_schema: dict | None = None  # request constrained JSON when the backend supports it
    extra: dict = field(default_factory=dict)

    def with_(self, **kw) -> "GenParams":
        return replace(self, **kw)


@dataclass
class Completion:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    model: str = ""
    call_id: str | None = None  # set by the tracing wrapper

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks emitted by reasoning-capable models."""
    text = _THINK.sub("", text)
    if "</think>" in text:  # opening tag was in the prompt (Qwen3 thinking templates)
        text = text.split("</think>", 1)[1]
    return text.strip("\n")


class LLMBackend(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    async def agenerate(self, messages: list[Message], params: GenParams) -> list[Completion]:
        ...

    def generate(self, messages: list[Message], params: GenParams | None = None) -> list[Completion]:
        return asyncio.run(self.agenerate(messages, params or GenParams()))

    async def aclose(self) -> None:  # pragma: no cover
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model}
