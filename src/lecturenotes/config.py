"""Agent configuration (YAML-loadable)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    type: Literal["openai", "hf", "anthropic", "mock"] = "openai"
    model: str = "lecturenotes"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str | None = None
    max_concurrency: int = 16
    timeout: float = 600.0
    json_mode: Literal["none", "response_format", "guided_json"] = "none"
    disable_thinking: bool = True
    extra_body: dict = Field(default_factory=dict)
    # hf backend
    adapter_path: str | None = None
    dtype: str = "auto"
    # anthropic backend
    effort: str = "high"
    # mock backend
    inject_errors: bool = True


class HarnessConfig(BaseModel):
    latex_engine: str = "pdflatex"
    compile_timeout: float = 90.0
    figure_timeout: float = 45.0
    samples_per_section: int = 1          # best-of-n drafts, reranked by the verifier reward
    adaptive_sampling: bool = True        # draw 1 first; the remaining n-1 only if reward < accept_reward
    accept_reward: float = 0.7
    max_latex_repairs: int = 3
    max_figure_repairs: int = 2
    regenerate_failed_figures: bool = True
    drop_failed_figures: bool = True      # False -> keep a "figure unavailable" placeholder
    ensure_figures: bool = True           # generate planned figures the draft left out
    max_quarantine_rounds: int = 6
    revise_rounds: int = 1
    length_tolerance: float = 0.35
    min_coverage: float = 0.75
    section_concurrency: int = 4
    figure_workers: int = 4
    cheatsheet_max_condense: int = 2
    toc_min_sections: int = 4


class AgentConfig(BaseModel):
    backends: dict[str, BackendConfig] = Field(default_factory=lambda: {"student": BackendConfig()})
    routes: dict[str, str] = Field(default_factory=lambda: {"*": "student"})
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    params: dict[str, dict] = Field(default_factory=dict)  # per-task GenParams overrides
    cache_dir: str | None = None

    @classmethod
    def load(cls, path: str | Path | None) -> "AgentConfig":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(_expand_env(data))


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(obj):
    """Expand ${VAR} and ${VAR:-default} in string values."""
    if isinstance(obj, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj
