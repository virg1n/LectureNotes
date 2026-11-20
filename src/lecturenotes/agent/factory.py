"""Build LLM backends / clients from configuration."""

from __future__ import annotations

import os

from ..config import AgentConfig, BackendConfig
from ..llm.base import GenParams, LLMBackend
from ..llm.client import DEFAULT_PARAMS, LLMClient, Tracer


def build_backend(cfg: BackendConfig) -> LLMBackend:
    if cfg.type == "openai":
        from ..llm.openai_compat import OpenAICompatBackend

        key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        return OpenAICompatBackend(cfg.base_url, cfg.model, api_key=key, timeout=cfg.timeout,
                                   max_concurrency=cfg.max_concurrency, json_mode=cfg.json_mode,
                                   extra_body=cfg.extra_body, disable_thinking=cfg.disable_thinking)
    if cfg.type == "hf":
        from ..llm.hf_local import HFLocalBackend

        return HFLocalBackend(cfg.model, adapter_path=cfg.adapter_path, dtype=cfg.dtype,
                              disable_thinking=cfg.disable_thinking)
    if cfg.type == "anthropic":
        from ..llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend(cfg.model, effort=cfg.effort, max_concurrency=cfg.max_concurrency)
    if cfg.type == "mock":
        from ..llm.mock import MockBackend

        return MockBackend(inject_errors=cfg.inject_errors)
    raise ValueError(f"unknown backend type {cfg.type}")


def build_client(cfg: AgentConfig, tracer: Tracer | None = None) -> LLMClient:
    backends = {name: build_backend(b) for name, b in cfg.backends.items()}
    params = {task: DEFAULT_PARAMS.get(task, GenParams()).with_(**over) for task, over in cfg.params.items()}
    return LLMClient(backends, cfg.routes, tracer=tracer, params=params)
