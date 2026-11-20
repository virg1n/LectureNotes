"""End-to-end benchmark: run the agent on held-out topics and aggregate harness metrics.

    python -m lecturenotes.eval.benchmark --config configs/agent/student_vllm.yaml --out eval_runs/sft-v1
    python -m lecturenotes.eval.benchmark --compare eval_runs/base eval_runs/sft-v1   # metric table

Metrics (all computed by the harness, no LLM needed):
  success        a PDF was produced
  draft_compile  share of first drafts that compiled without any model repair (raw model quality)
  figures_ok     rendered-and-lint-clean figures / planned figures
  len_err        mean |log(words / budget)| (0 = perfect length control)
  coverage       key-point coverage
  reward         mean section reward
  cs_fit         cheat sheets within their page budget
  repairs        model repair calls per section;  tokens / time per lecture
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
from pathlib import Path

from ..agent.orchestrator import LectureAgent
from ..config import AgentConfig
from ..data.topics import AUDIENCES, EVAL_TOPICS
from ..schemas import LectureRequest
from ..utils import iter_jsonl, slugify


def eval_requests(n: int | None = None, seed: int = 123) -> list[LectureRequest]:
    rng = random.Random(seed)
    lengths = ["4 pages", "6 pages", "8 pages", "2000 words"]
    reqs = [LectureRequest(topic=t, audience=rng.choice(AUDIENCES), expected_length=rng.choice(lengths),
                           make_cheatsheet=True, cheatsheet_pages=rng.choice([1, 2])) for t in EVAL_TOPICS]
    decks = sorted(Path(__file__).parent.glob("decks/*.pdf"))
    for d in decks:  # slide-conditioned cases (optional)
        reqs.append(LectureRequest(topic=d.stem.replace("_", " "), presentation=d, expected_length="6 pages"))
    return reqs[:n] if n else reqs


def run_metrics(run_dir: Path) -> dict | None:
    rp = run_dir / "report.json"
    if not rp.exists():
        return None
    r = json.loads(rp.read_text(encoding="utf-8"))
    secs = r["sections"]
    drafts = []
    repairs = 0
    tr = run_dir / "trace.jsonl"
    if tr.exists():
        for rec in iter_jsonl(tr):
            o = rec.get("outcome", {})
            if rec["task"] == "write_section" and o.get("selected"):
                drafts.append(bool(o.get("verifier", {}).get("compile_ok")))
            if rec["task"] in ("repair_latex", "repair_figure"):
                repairs += 1
    t = r["totals"]
    usage = r.get("llm_usage", {})
    # denominator = figures the system set out to make; dropping failed figures must not look like success
    dropped = sum(len(s.get("dropped_figures") or []) for s in secs)
    planned = t["figures"] + dropped
    op = run_dir / "outline.json"
    if op.exists():
        outline = json.loads(op.read_text(encoding="utf-8"))
        planned = max(planned, sum(len(s.get("figures", [])) for s in outline.get("sections", [])))
    return {
        "topic": r["request"]["topic"], "success": bool(r["document"]["ok"]),
        "pages": r["document"]["pages"], "draft_compile": sum(drafts) / max(len(drafts), 1),
        "figures_ok": min(1.0, t["figures_ok"] / planned) if planned else 1.0,
        "len_err": abs(math.log(max(t["length_ratio"], 1e-3))), "coverage": t["mean_coverage"],
        "reward": t["mean_reward"], "repairs_per_section": repairs / max(len(secs), 1),
        "quarantined": t["quarantined_units"],
        "cs_fit": (r["cheatsheet"]["pages"] or 99) <= r["cheatsheet"]["budget"] if r.get("cheatsheet") else None,
        "tokens": sum(u["prompt_tokens"] + u["completion_tokens"] for u in usage.values()),
        "seconds": r["seconds"],
    }


def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean (eval sets are small: report uncertainty)."""
    if len(vals) < 2:
        return (vals[0], vals[0]) if vals else (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def aggregate(rows: list[dict]) -> dict:
    agg = {"n": len(rows)}
    for k in ("success", "draft_compile", "figures_ok", "len_err", "coverage", "reward", "repairs_per_section",
              "quarantined", "tokens", "seconds"):
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        agg[k] = round(statistics.mean(vals), 4) if vals else None
        if k in ("reward", "draft_compile", "figures_ok") and vals:
            lo, hi = bootstrap_ci(vals)
            agg[f"{k}_ci95"] = [round(lo, 3), round(hi, 3)]
    cs = [r["cs_fit"] for r in rows if r.get("cs_fit") is not None]
    agg["cs_fit"] = round(sum(cs) / len(cs), 4) if cs else None
    return agg


async def run_suite(cfg: AgentConfig, out: Path, reqs: list[LectureRequest], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(req: LectureRequest):
        d = out / slugify(req.topic, 60)
        if not (d / "report.json").exists():
            async with sem:
                try:
                    await LectureAgent(cfg).run(req, d)
                except Exception as e:
                    (d / "error.txt").parent.mkdir(parents=True, exist_ok=True)
                    (d / "error.txt").write_text(repr(e))
        return run_metrics(d) or {"topic": req.topic, "success": False}

    return list(await asyncio.gather(*(one(r) for r in reqs)))


def compare(dirs: list[Path]) -> str:
    keys = ["success", "draft_compile", "figures_ok", "len_err", "coverage", "reward", "repairs_per_section",
            "cs_fit", "tokens", "seconds"]
    lines = ["| system | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for d in dirs:
        rows = [m for m in (run_metrics(p) for p in sorted(d.iterdir()) if p.is_dir()) if m]
        agg = aggregate(rows)
        lines.append(f"| {d.name} | " + " | ".join(str(agg.get(k)) for k in keys) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--compare", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.compare:
        print(compare(a.compare))
        return
    cfg = AgentConfig.load(a.config)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = asyncio.run(run_suite(cfg, a.out, eval_requests(a.n), a.concurrency))
    agg = aggregate(rows)
    (a.out / "summary.json").write_text(json.dumps({"aggregate": agg, "runs": rows}, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
