"""Task-aware LLM client: routes each task to a backend, applies per-task decoding
defaults and records every call in a trace.

A trace record = (task, messages, completion, params, backend) + *outcome
annotations* written later by the harness (accepted? reward? verifier summary?).
Accepted records are directly usable as SFT examples, and (best, worst) pairs
for the same prompt as preference data -- the agent produces its own training
data whenever a strong teacher drives it.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .. import PROMPT_VERSION
from .base import Completion, GenParams, LLMBackend, Message

DEFAULT_PARAMS: dict[str, GenParams] = {
    "plan": GenParams(temperature=0.4, max_tokens=2500),
    "write_section": GenParams(temperature=0.6, max_tokens=5000),
    "revise_section": GenParams(temperature=0.5, max_tokens=5000),
    "repair_latex": GenParams(temperature=0.2, max_tokens=1500),
    "repair_figure": GenParams(temperature=0.3, max_tokens=2000),
    "figure": GenParams(temperature=0.5, max_tokens=2000),
    "cheatsheet": GenParams(temperature=0.4, max_tokens=5000),
    "condense_cheatsheet": GenParams(temperature=0.3, max_tokens=5000),
    "web_query": GenParams(temperature=0.3, max_tokens=40),
    "pick_image": GenParams(temperature=0.0, max_tokens=120),
}


class Tracer:
    def __init__(self, path: str | Path | None = None, run_meta: dict | None = None):
        self.path = Path(path) if path else None
        self.run_meta = run_meta or {}
        self.records: dict[str, dict] = {}
        self._lock = threading.Lock()

    def log(self, *, task: str, messages: list[Message], params: GenParams, completions: list[Completion],
            backend: dict, meta: dict | None = None) -> list[str]:
        ids = []
        with self._lock:
            for c in completions:
                cid = uuid.uuid4().hex[:12]
                c.call_id = cid
                self.records[cid] = {
                    "id": cid, "task": task, "prompt_version": PROMPT_VERSION, "time": time.time(),
                    "messages": messages, "output": c.text, "finish_reason": c.finish_reason,
                    "usage": {"prompt": c.prompt_tokens, "completion": c.completion_tokens},
                    "latency": round(c.latency, 3), "backend": backend,
                    "params": {k: v for k, v in asdict(params).items() if k not in ("json_schema",)},
                    "meta": dict(meta or {}), "outcome": {}, "run": self.run_meta,
                }
                ids.append(cid)
        return ids

    def annotate(self, call_id: str | None, **outcome) -> None:
        if not call_id:
            return
        with self._lock:
            rec = self.records.get(call_id)
            if rec is not None:
                rec["outcome"].update(outcome)

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "w", encoding="utf-8") as f:
            for r in self.records.values():
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    def usage(self) -> dict:
        with self._lock:
            out: dict[str, dict] = {}
            for r in self.records.values():
                t = out.setdefault(r["task"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency": 0.0})
                t["calls"] += 1
                t["prompt_tokens"] += r["usage"]["prompt"]
                t["completion_tokens"] += r["usage"]["completion"]
                t["latency"] += r["latency"]
            return out


class LLMClient:
    def __init__(self, backends: dict[str, LLMBackend], routes: dict[str, str] | None = None,
                 tracer: Tracer | None = None, params: dict[str, GenParams] | None = None):
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self.routes = dict(routes or {})
        self.tracer = tracer or Tracer()
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    def backend_for(self, task: str) -> LLMBackend:
        name = self.routes.get(task) or self.routes.get("*") or next(iter(self.backends))
        return self.backends[name]

    async def complete(self, task: str, messages: list[Message], *, params: GenParams | None = None,
                       meta: dict | None = None, **overrides) -> list[Completion]:
        p = params or self.params.get(task, GenParams())
        if overrides:
            p = p.with_(**overrides)
        be = self.backend_for(task)
        if hasattr(be, "agenerate_task"):
            comps = await be.agenerate_task(task, messages, p)  # type: ignore[attr-defined]
        else:
            comps = await be.agenerate(messages, p)
        self.tracer.log(task=task, messages=messages, params=p, completions=comps, backend=be.describe(), meta=meta)
        return comps

    async def complete_one(self, task: str, messages: list[Message], **kw) -> Completion:
        return (await self.complete(task, messages, **kw))[0]

    async def aclose(self) -> None:
        for b in self.backends.values():
            await b.aclose()
