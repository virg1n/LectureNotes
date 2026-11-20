"""Shared pieces for on-policy RL (GRPO):
prompt pools, harness rewards (process pool), rollout engines (vLLM pool or HF generate)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = {"write_section": 4096, "repair_latex": 1536, "repair_figure": 2048, "figure": 2048,
                  "plan": 2500, "cheatsheet": 4096, "web_query": 40, "pick_image": 120, "revise_section": 4096, "condense_cheatsheet": 4096}


@dataclass
class RLPrompt:
    task: str
    messages: list[dict]  # system + user (no assistant)
    meta: dict = field(default_factory=dict)
    group: str = ""


# ----------------------------------------------------------------------------- prompt parsing
# Prompts are produced by prompts/templates.py, so their fields can be recovered reliably.

def parse_section_prompt(user: str) -> dict:
    m = re.search(r"Length: about (\d+) words", user)
    kp = re.search(r"Key points to explain: (.*)", user)
    figs = re.findall(r"^- id `([^`]+)` \((\w+)\)", user, re.M)
    title = re.search(r"Start with \\section\{(.*)\}\. Output only", user)
    lang = re.search(r"Write in (\w+)\.", user)
    return {"target_words": int(m.group(1)) if m else None,
            "key_points": [k.strip() for k in kp.group(1).split(";") if k.strip() and k.strip() != "-"] if kp else [],
            "expected_figures": len(figs), "title": title.group(1) if title else None,
            "language": "ru" if lang and lang.group(1) == "Russian" else "en"}


def parse_repair_prompt(user: str) -> dict:
    m = re.search(r"## Source \(line numbers added for reference\)\n(.*?)\n\n## Output", user, re.S)
    src = "\n".join(re.sub(r"^\s*\d+\| ?", "", l) for l in (m.group(1).splitlines() if m else []))
    return {"source": src}


def parse_figure_prompt(user: str) -> dict:
    m = re.search(r"Figure id: ([\w-]+); kind: (\w+)", user)
    return {"fig_id": m.group(1) if m else "fig", "kind": m.group(2) if m else "matplotlib"}


def parse_plan_prompt(user: str) -> dict:
    w = re.search(r"Target length: (\d+) words of prose in (\d+)-(\d+) sections", user)
    sl = re.search(r"exactly one section: \[([^\]]*)\]", user)
    return {"total_words": int(w.group(1)) if w else 2000,
            "n_sections": (int(w.group(2)), int(w.group(3))) if w else (3, 6),
            "slides": [int(x) for x in sl.group(1).split(",") if x.strip()] if sl else None}


def parse_cheatsheet_prompt(user: str) -> dict:
    p = re.search(r"fits on (\d+) landscape", user)
    focus = re.search(r"Must include: (.*)\.\n", user)
    return {"pages": int(p.group(1)) if p else 2, "facts": focus.group(1).split("; ") if focus else []}


def parse_web_query_prompt(user: str) -> dict:
    m = re.search(r"showing: (.+)", user)
    return {"description": m.group(1).strip() if m else ""}


PARSERS = {"write_section": parse_section_prompt, "revise_section": parse_section_prompt,
           "repair_latex": parse_repair_prompt, "repair_figure": parse_figure_prompt, "figure": parse_figure_prompt,
           "plan": parse_plan_prompt, "cheatsheet": parse_cheatsheet_prompt, "web_query": parse_web_query_prompt,
           "pick_image": lambda user: {"n_candidates": len(re.findall(r"^\[\d+\] ", user, re.M))}}


def load_prompt_pool(files: list[str], tasks: dict[str, float] | None = None, max_per_task: int = 20000,
                     seed: int = 0) -> dict[str, list[RLPrompt]]:
    rng = random.Random(seed)
    pool: dict[str, list[RLPrompt]] = {}
    seen = set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                task = r["task"]
                if tasks and task not in tasks:
                    continue
                msgs = [m for m in r["messages"] if m["role"] != "assistant"]
                user = msgs[-1]["content"]
                key = hash(user)
                if key in seen or task not in PARSERS:
                    continue
                seen.add(key)
                meta = PARSERS[task](user)
                meta.update(r.get("meta") or {})  # exact fields stored at build time beat prompt parsing
                if task == "pick_image" and r["messages"][-1]["role"] == "assistant":
                    meta["gold_output"] = r["messages"][-1]["content"]  # the teacher's choice
                if task == "repair_latex" and r["messages"][-1]["role"] == "assistant":
                    meta["gold_output"] = r["messages"][-1]["content"]
                pool.setdefault(task, []).append(RLPrompt(task, msgs, meta, r.get("group", "")))
    for t in pool:
        rng.shuffle(pool[t])
        pool[t] = pool[t][:max_per_task]
    return pool


# ----------------------------------------------------------------------------- rewards (worker processes)

_W = None


def _worker_verifier():
    global _W
    if _W is None:
        from ..harness.verifier import Verifier
        from ..latex.capabilities import detect_capabilities

        _W = Verifier(caps=detect_capabilities(), figure_workers=1)
    return _W


def compute_reward(task: str, meta: dict, completion: str, finish_reason: str = "stop") -> tuple[float, dict]:
    """Harness reward for one completion (runs inside a worker process)."""
    import tempfile

    from ..figures.blocks import parse_blocks
    from ..harness.edits import apply_model_repair, strip_wrapping
    from ..harness.rewards import cheatsheet_reward, figure_reward, plan_reward, repair_reward, score_section
    from ..harness.verifier import SectionContext
    from ..schemas import Outline
    from ..utils import extract_json_object

    if finish_reason == "length":
        return -1.0, {"truncated": True}
    v = _worker_verifier()
    try:
        if task in ("write_section", "revise_section"):
            ctx = SectionContext(sid="s1", title=meta.get("title"), target_words=meta.get("target_words"),
                                 key_points=meta.get("key_points", []), language=meta.get("language", "en"))
            rep = v.verify_section(strip_wrapping(completion), ctx)
            return score_section(v, rep, ctx, expected_figures=meta.get("expected_figures", 0))
        if task == "repair_latex":
            before_src = meta["source"]
            before = v.verify_section(before_src, SectionContext(sid="s1"), render_figures=False)
            out = apply_model_repair(before_src, completion, expect_prefix=None)
            if out.applied == 0:
                return -1.0, {"applied": 0}
            after = v.verify_section(out.source, SectionContext(sid="s1"), render_figures=False)
            gold = None
            if meta.get("gold_output"):
                g = apply_model_repair(before_src, meta["gold_output"], expect_prefix=None)
                gold = g.source if g.applied else None
            return repair_reward(before, after, before.source, after.source, gold), {"compile_ok": after.compile_ok}
        if task in ("figure", "repair_figure"):
            blocks, _ = parse_blocks(completion)
            if not blocks:
                return -1.0, {"no_block": True}
            b = blocks[0]
            b.id, b.kind = meta.get("fig_id", b.id), meta.get("kind", b.kind)
            res = v.renderer.render(b, Path(tempfile.gettempdir()) / f"lnrl-{os.getpid()}")
            return figure_reward(res), {"ok": res.ok, "passed": res.passed}
        if task == "plan":
            try:
                from ..agent.planner import _normalize_plan

                outline = Outline.model_validate(_normalize_plan(extract_json_object(completion)))
            except Exception:
                outline = None
            return plan_reward(outline, total_words=meta["total_words"], n_sections=tuple(meta["n_sections"]),
                               selected_slides=meta.get("slides"))
        if task == "pick_image":
            # agreement with the teacher's judgement (text-only metadata, same as at inference)
            from ..prompts.templates import parse_pick
            n = meta.get("n_candidates") or 8
            got, ok = parse_pick(completion, n)
            gold, gok = parse_pick(meta.get("gold_output", ""), n)
            if not ok:
                return -1.0, {"parsed": False}
            if not gok:
                return 0.0, {"no_gold": True}
            if got == gold:
                return 1.0, {"agree": True}
            return (-0.5 if (got is None) != (gold is None) else -0.3), {"agree": False}
        if task == "web_query":
            # found a freely licensed image -> success; short concrete queries preferred.
            # (relevance is not judged here: an image-caption similarity model would be the next step)
            from ..figures.web import fetch_image
            from ..prompts.templates import clean_query

            q = clean_query(completion)
            n_words = len(q.split())
            if not q or n_words > 8:
                return -1.0, {"query": q}
            img = fetch_image(q, Path(tempfile.gettempdir()) / f"lnrl-web-{os.getpid()}", v.renderer.cache_dir.parent / "web")
            r = (1.0 if img else -0.5) - 0.05 * max(0, n_words - 5)
            return r, {"query": q, "found": bool(img)}
        if task == "cheatsheet":
            from ..agent.cheatsheet import clean_cheatsheet

            ctx = SectionContext(sid="cheatsheet", mode="cheatsheet", key_points=meta.get("facts", []))
            rep = v.verify_section(clean_cheatsheet(completion), ctx)
            return cheatsheet_reward(rep, page_budget=meta.get("pages", 2), facts=meta.get("facts", []))
    except Exception as e:  # a crashing verifier must not crash training
        return -1.0, {"error": f"{type(e).__name__}: {e}"[:200]}
    return 0.0, {"unknown_task": task}


class RewardPool:
    def __init__(self, workers: int):
        self.ex = ProcessPoolExecutor(max_workers=workers)

    def score(self, items: list[tuple[str, dict, str, str]]) -> list[tuple[float, dict]]:
        futs = [self.ex.submit(compute_reward, *it) for it in items]
        out = []
        for f in futs:
            try:
                out.append(f.result(timeout=600))
            except Exception as e:
                out.append((-1.0, {"error": str(e)[:200]}))
        return out

    def close(self):
        self.ex.shutdown(cancel_futures=True)


# ----------------------------------------------------------------------------- rollout engines

_DIST_ENV = {"MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "ROLE_RANK",
             "ROLE_WORLD_SIZE", "ROLE_NAME", "OMP_NUM_THREADS"}


class VLLMPool:
    """One vLLM OpenAI server per rollout GPU (data parallel), restartable with new weights."""

    def __init__(self, gpus: list[int], base_port: int = 8100, model: str | None = None, extra_args: list[str] | None = None,
                 python: str | None = None, log_dir: str = "logs/vllm", max_model_len: int = 12288):
        self.gpus = gpus
        self.ports = [base_port + i for i in range(len(gpus))]
        self.extra = extra_args or []
        self.python = python or sys.executable
        self.procs: list[subprocess.Popen] = []
        self.model = model
        self.log_dir = Path(log_dir)
        self.max_model_len = max_model_len

    @property
    def urls(self) -> list[str]:
        return [f"http://127.0.0.1:{p}/v1" for p in self.ports]

    def start(self, model: str, served_name: str = "policy") -> None:
        self.stop()
        self.model = model
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for gpu, port in zip(self.gpus, self.ports):
            # started from inside torchrun: drop its rendezvous vars, or vLLM tries to join the trainer's group
            env = {k: v for k, v in os.environ.items()
                   if k not in _DIST_ENV and not k.startswith(("TORCHELASTIC_", "TORCH_NCCL_", "GROUP_"))}
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_ALLOW_RUNTIME_LORA_UPDATING="True")
            cmd = [self.python, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name",
                   served_name, "--port", str(port), "--dtype", "half", "--max-model-len", str(self.max_model_len),
                   "--gpu-memory-utilization", "0.88", "--disable-log-requests", *self.extra]
            logf = open(self.log_dir / f"vllm-gpu{gpu}.log", "a")
            self.procs.append(subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT))
        self.wait_ready()

    def wait_ready(self, timeout: float = 900) -> None:
        import httpx

        t0 = time.time()
        pending = set(self.urls)
        while pending and time.time() - t0 < timeout:
            for u in list(pending):
                try:
                    if httpx.get(u.replace("/v1", "/health"), timeout=5).status_code == 200:
                        pending.discard(u)
                except Exception:
                    pass
            if any(p.poll() is not None for p in self.procs):
                raise RuntimeError(f"a vLLM server exited; see {self.log_dir}")
            time.sleep(5)
        if pending:
            raise TimeoutError(f"vLLM servers not ready: {pending}")

    def load_lora(self, name: str, path: str) -> None:
        import httpx

        for u in self.urls:
            r = httpx.post(u.replace("/v1", "/v1/load_lora_adapter"), json={"lora_name": name, "lora_path": path}, timeout=300)
            if r.status_code != 200:
                raise RuntimeError(f"load_lora_adapter failed on {u}: {r.text[:300]}")

    def unload_lora(self, name: str) -> None:
        import httpx

        for u in self.urls:
            try:
                httpx.post(u.replace("/v1", "/v1/unload_lora_adapter"), json={"lora_name": name}, timeout=120)
            except Exception as e:  # best effort
                log.warning("unload_lora_adapter failed on %s: %s", u, e)

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []


async def rollout_openai(urls: list[str], model: str, prompts: list[RLPrompt], n: int, temperature: float,
                         top_p: float = 0.95, per_server: int = 4, timeout: float = 3600.0) -> list[list[tuple[str, str]]]:
    """Sample n completions per prompt, spreading prompts round-robin over servers.

    `per_server` bounds in-flight requests (each carries n sequences): a V100 KV cache holds only
    ~20-30 long sequences, and over-subscription turns into client timeouts and silently lost groups."""
    from ..llm.base import GenParams
    from ..llm.openai_compat import OpenAICompatBackend

    backends = [OpenAICompatBackend(u, model, max_concurrency=per_server, timeout=timeout, max_retries=2,
                                    disable_thinking=True) for u in urls]
    failures = 0

    async def one(i: int, p: RLPrompt):
        nonlocal failures
        be = backends[i % len(backends)]
        params = GenParams(temperature=temperature, top_p=top_p, n=n, max_tokens=MAX_NEW_TOKENS.get(p.task, 4096))
        try:
            comps = await be.agenerate(p.messages, params)
            return [(c.text, c.finish_reason) for c in comps]
        except Exception as e:
            failures += 1
            log.warning("rollout failed: %s", e)
            return [("", "error")] * n

    res = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    for b in backends:
        await b.aclose()
    if failures:
        log.warning("rollout: %d/%d prompts failed (groups get reward -1 and are dropped if uniform)",
                    failures, len(prompts))
    return list(res)
