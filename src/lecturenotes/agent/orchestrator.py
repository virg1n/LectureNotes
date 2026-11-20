"""Top-level agent: request -> outline -> sections (parallel) -> document -> cheat sheet -> report."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AgentConfig
from ..figures.blocks import parse_blocks
from ..figures.runner import FigureRenderer
from ..harness.edits import quarantine, remove_refs
from ..harness.verifier import SectionContext, Verifier
from ..ingest.slides import SlideDeck, load_deck
from ..latex.assembler import BuildDir, escape_text, flatten, render_cheatsheet, render_notes
from ..latex.capabilities import TexCapabilities, detect_capabilities
from ..latex.compiler import LatexCompiler
from ..llm.client import LLMClient, Tracer
from ..schemas import LectureRequest, Outline
from .budget import LengthBudget, parse_length
from .cheatsheet import CheatsheetBuilder, CheatsheetOutcome
from .factory import build_client
from .planner import make_plan
from .section import SectionOutcome, SectionWorker, rename_duplicate_labels

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    out_dir: Path
    notes_pdf: Path | None
    notes_tex: Path | None
    cheatsheet_pdf: Path | None
    report: dict = field(default_factory=dict)


def _dedupe_figure_ids(sources: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    """Figure ids must be unique document-wide (labels fig:<id>)."""
    seen: set[str] = set()
    changed: set[str] = set()
    out = {}
    for sid, src in sources.items():
        blocks, _ = parse_blocks(src)
        for b in blocks:
            if b.id in seen:
                new = f"{b.id}-{sid}"
                src = re.sub(rf'(<figure\b[^>]*\bid\s*=\s*["\']){re.escape(b.id)}(["\'])', rf"\g<1>{new}\g<2>", src, count=1)
                src = re.sub(rf"\\(ref|autoref)\{{fig:{re.escape(b.id)}\}}", rf"\\\1{{fig:{new}}}", src)
                changed.add(sid)
                seen.add(new)
            else:
                seen.add(b.id)
        out[sid] = src
    return out, changed


class LectureAgent:
    def __init__(self, config: AgentConfig | None = None, client: LLMClient | None = None):
        self.cfg = config or AgentConfig()
        self._client = client

    async def run(self, req: LectureRequest, out_dir: str | Path) -> RunResult:
        t0 = time.monotonic()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        hc = self.cfg.harness
        tracer = Tracer(out / "trace.jsonl", run_meta={"topic": req.topic, "started": dt.datetime.now().isoformat()})
        client = self._client or build_client(self.cfg, tracer)
        client.tracer = tracer

        compiler = LatexCompiler(hc.latex_engine, timeout=hc.compile_timeout)
        caps = await asyncio.to_thread(detect_capabilities, compiler)
        if req.language == "ru" and not caps.has("cyrillic"):
            raise RuntimeError("Russian output needs Cyrillic LaTeX support (TeX Live: texlive-lang-cyrillic + "
                               "cm-super; TinyTeX: tlmgr install babel-russian cyrillic lh cm-super)")

        deck: SlideDeck | None = None
        if req.presentation:
            deck = await asyncio.to_thread(load_deck, req.presentation, out / "slides")
            deck = deck.select(req.slides)
            log.info("loaded %d slides", len(deck))
        budget = parse_length(req.expected_length, len(deck) if deck is not None else None)

        build = BuildDir.create(out / "build")
        renderer = FigureRenderer(self.cfg.cache_dir, timeout=hc.figure_timeout, compiler=compiler, caps=caps,
                                  language=req.language)
        verifier = Verifier(compiler=compiler, caps=caps, renderer=renderer, language=req.language,
                            slide_images=deck.images() if deck is not None else None,
                            figure_workers=hc.figure_workers)
        verifier.topic = req.topic

        # ---------------------------------------------------------------- plan
        outline, plan_info = await make_plan(client, req, budget, deck, caps)
        (out / "outline.json").write_text(outline.model_dump_json(indent=2), encoding="utf-8")
        log.info("outline: %d sections, %d words", len(outline.sections), budget.words)

        # ---------------------------------------------------------------- sections
        sem = asyncio.Semaphore(hc.section_concurrency)
        worker = SectionWorker(client=client, verifier=verifier, cfg=hc, req=req, outline=outline, deck=deck,
                               caps=caps, figures_dir=build.figures_dir)

        async def run_one(sec):
            async with sem:
                try:
                    return await worker.run(sec)
                except Exception as e:  # one broken section (e.g. a backend error) must not sink the lecture
                    log.exception("section %s failed; using a minimal fallback section", sec.id)
                    items = "\n".join(f"  \\item {escape_text(k)}" for k in sec.key_points) or "  \\item --"
                    src = f"\\section{{{escape_text(sec.title)}}}\n\\begin{{itemize}}\n{items}\n\\end{{itemize}}\n"
                    rep = await asyncio.to_thread(
                        verifier.verify_section, src,
                        SectionContext(sid=sec.id, title=sec.title, language=req.language),
                        figures_dir=build.figures_dir)
                    return SectionOutcome(sec, src, rep, -1.0, [{"step": "failed", "error": f"{type(e).__name__}: {e}"[:300]}])

        outcomes: list[SectionOutcome] = list(await asyncio.gather(*(run_one(s) for s in outline.sections)))

        # ---------------------------------------------------------------- assemble
        sources = {o.plan.id: o.source for o in outcomes}
        sources, changed = _dedupe_figure_ids(sources)
        sources = rename_duplicate_labels(sources)
        processed: dict[str, str] = {}
        reports = {o.plan.id: o.report for o in outcomes}
        all_labels = {lab for r in reports.values() for lab in r.labels} | {
            b.label for r in reports.values() for b in r.figure_blocks}
        for sid, src in sources.items():
            if sid in changed or src != reports[sid].source:
                ctx = SectionContext(sid=sid, title=None, language=req.language, known_labels=all_labels)
                reports[sid] = await asyncio.to_thread(verifier.verify_section, src, ctx,
                                                       figures_dir=build.figures_dir,
                                                       prior_results=reports[sid].figure_results)
            processed[sid] = reports[sid].processed

        doc_info = await asyncio.to_thread(self._compile_document, build, compiler, caps, req, outline, processed)
        notes_pdf = out / "notes.pdf"
        if doc_info["pdf"]:
            shutil.copy2(doc_info["pdf"], notes_pdf)
        notes_tex = self._export_tex(out, build)

        # ---------------------------------------------------------------- cheat sheet
        cs: CheatsheetOutcome | None = None
        cs_pdf = None
        if req.make_cheatsheet:
            cs = await CheatsheetBuilder(client=client, verifier=verifier, cfg=hc, req=req,
                                         outline=outline).build(sources)
            cs_pdf = await asyncio.to_thread(self._compile_cheatsheet, out, compiler, caps, req, outline, cs)

        # ---------------------------------------------------------------- report
        report = self._report(req, budget, outline, plan_info, outcomes, reports, doc_info, cs, client,
                              time.monotonic() - t0, caps)
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "report.md").write_text(render_report_md(report), encoding="utf-8")
        tracer.flush()
        await client.aclose()
        return RunResult(out, notes_pdf if notes_pdf.exists() else None, notes_tex, cs_pdf, report)

    # ------------------------------------------------------------------ document
    def _compile_document(self, build: BuildDir, compiler: LatexCompiler, caps: TexCapabilities,
                          req: LectureRequest, outline: Outline, processed: dict[str, str]) -> dict:
        meta_parts = [p for p in (req.course, req.author, req.date or dt.date.today().strftime("%B %Y")) if p]
        main = render_notes(
            title=outline.title, subtitle=outline.subtitle, meta=" -- ".join(meta_parts),
            abstract=outline.abstract, objectives=outline.learning_objectives, prerequisites=outline.prerequisites,
            sections=list(processed), language=req.language,
            toc=len(processed) >= self.cfg.harness.toc_min_sections, caps=caps)
        build.write_main(main)
        dropped: list[str] = []
        res = None
        for attempt in range(4):
            for sid, tex in processed.items():
                build.write_section(sid, tex)
            res = compiler.compile(build.root, "main.tex", passes=2, max_passes=3, halt_on_error=False)
            if res.ok:
                break
            # document-level failure: quarantine the offending units in the responsible section
            fixed_any = False
            for e in res.errors:
                m = re.search(r"sections/(s\d+)\.tex", e.file or "")
                if m and m.group(1) in processed:
                    new, removed = quarantine(processed[m.group(1)], [e.line])
                    if new != processed[m.group(1)]:
                        processed[m.group(1)] = new
                        dropped += removed
                        fixed_any = True
            if not fixed_any:
                break
        assert res is not None
        undefined = sorted(set(res.log.undefined_refs))
        if res.ok and undefined:
            # "see Theorem ??" is worse than no pointer: drop references nothing defines, recompile once
            for sid in processed:
                processed[sid] = remove_refs(processed[sid], undefined)
                build.write_section(sid, processed[sid])
            res2 = compiler.compile(build.root, "main.tex", passes=2, max_passes=3, halt_on_error=False)
            if res2.ok:
                res = res2
        return {
            "pdf": res.pdf_path if res.ok else None, "ok": res.ok, "pages": res.log.pages,
            "errors": [e.short() for e in res.errors[:10]], "warnings": len(res.log.warnings),
            "overfull": len(res.log.overfull(8.0)), "undefined_refs": sorted(set(res.log.undefined_refs)),
            "removed_undefined_refs": undefined if res.ok else [],
            "doc_quarantined": len(dropped), "compile_seconds": round(res.duration, 2),
        }

    @staticmethod
    def _export_tex(out: Path, build: BuildDir) -> Path | None:
        main = build.root / "main.tex"
        if not main.exists():
            return None
        tex = flatten(main.read_text(encoding="utf-8"), build.sections_dir, sty_inline=True)
        dst = out / "notes.tex"
        dst.write_text(tex, encoding="utf-8")
        figs = out / "figures"
        if figs.exists():
            shutil.rmtree(figs)
        # export only figures the document references (unselected best-of-n drafts leave files behind)
        for rel in set(re.findall(r"\\lnfigure(?:here)?\{[^}]*\}\{([^}]+)\}", tex)):
            src = build.figures_dir / rel
            if src.exists():
                (figs / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, figs / rel)
        figs.mkdir(exist_ok=True)
        return dst

    def _compile_cheatsheet(self, out: Path, compiler: LatexCompiler, caps: TexCapabilities, req: LectureRequest,
                            outline: Outline, cs: CheatsheetOutcome) -> Path | None:
        bd = BuildDir.create(out / "build_cheatsheet")
        bd.write_section("cheatsheet", cs.report.processed)
        meta = " -- ".join(p for p in (req.course, outline.subtitle) if p)
        bd.write_main(render_cheatsheet(title=outline.title, meta=meta, language=req.language, columns=cs.columns,
                                        fontsize=cs.fontsize, caps=caps))
        res = compiler.compile(bd.root, "main.tex", passes=1, halt_on_error=False)
        if not res.ok or not res.pdf_path:
            log.warning("cheat sheet failed to compile: %s", [e.short() for e in res.errors[:3]])
            return None
        dst = out / "cheatsheet.pdf"
        shutil.copy2(res.pdf_path, dst)
        (out / "cheatsheet.tex").write_text(flatten((bd.root / "main.tex").read_text(encoding="utf-8"),
                                                    bd.sections_dir, sty_inline=True), encoding="utf-8")
        return dst

    # ------------------------------------------------------------------ report
    @staticmethod
    def _report(req, budget: LengthBudget, outline: Outline, plan_info, outcomes: list[SectionOutcome], reports,
                doc_info, cs: CheatsheetOutcome | None, client: LLMClient, seconds: float, caps) -> dict:
        secs = []
        for o in outcomes:
            s = o.summary()
            r = reports[o.plan.id]
            s.update(words=r.stats.words, figures=len(r.figure_blocks), figures_ok=r.figures_ok)
            secs.append(s)
        total_words = sum(s["words"] for s in secs)
        return {
            "request": json.loads(req.model_dump_json()),
            "budget": {"pages": budget.pages, "words": budget.words, "sections": budget.n_sections,
                       "source": budget.source},
            "plan": {**plan_info, "sections": len(outline.sections), "title": outline.title},
            "document": doc_info,
            "sections": secs,
            "totals": {
                "words": total_words, "length_ratio": round(total_words / max(budget.words, 1), 3),
                "figures": sum(s["figures"] for s in secs), "figures_ok": sum(s["figures_ok"] for s in secs),
                "mean_coverage": round(sum(s["coverage"] for s in secs) / max(len(secs), 1), 3),
                "mean_reward": round(sum(s["reward"] for s in secs) / max(len(secs), 1), 3),
                "autofixes": sum(s["n_fixes"] for s in secs),
                "quarantined_units": sum(s["quarantined_units"] for s in secs),
            },
            "cheatsheet": None if cs is None else {
                "pages": cs.pages, "budget": req.cheatsheet_pages, "layout": [cs.columns, cs.fontsize],
                "reward": round(cs.reward, 3), "compile_ok": cs.report.compile_ok, "steps": cs.history},
            "llm_usage": client.tracer.usage(),
            "tex": {"engine": caps.engine, "distribution": caps.distribution,
                    "missing_features": [k for k, v in caps.features.items() if not v]},
            "seconds": round(seconds, 1),
        }


def render_report_md(r: dict) -> str:
    d = r["document"]
    t = r["totals"]
    lines = [
        f"# Run report: {r['plan']['title']}",
        "",
        f"- document compiled: **{d['ok']}**, pages: {d['pages']}, overfull boxes: {d['overfull']}, "
        f"undefined refs: {len(d['undefined_refs'])}",
        f"- words: {t['words']} (budget {r['budget']['words']}, ratio {t['length_ratio']})",
        f"- figures: {t['figures_ok']}/{t['figures']} rendered, mean key-point coverage {t['mean_coverage']}",
        f"- autofixes: {t['autofixes']}, quarantined units: {t['quarantined_units']}, plan fallback: {r['plan']['fallback']}",
        f"- time: {r['seconds']} s",
        "",
        "| section | words / target | coverage | figures | reward | steps |",
        "|---|---|---|---|---|---|",
    ]
    for s in r["sections"]:
        lines.append(f"| {s['id']} {s['title'][:40]} | {s['words']} / {s['target_words']} | {s['coverage']} | "
                     f"{s['figures_ok']}/{s['figures']} | {s['reward']} | {' > '.join(s['steps'])} |")
    if r.get("cheatsheet"):
        c = r["cheatsheet"]
        lines += ["", f"Cheat sheet: {c['pages']} page(s) (budget {c['budget']}), layout {c['layout']}, "
                      f"reward {c['reward']}"]
    lines += ["", "LLM usage:", ""]
    for task, u in r["llm_usage"].items():
        lines.append(f"- {task}: {u['calls']} calls, {u['prompt_tokens']} prompt / {u['completion_tokens']} "
                     f"completion tokens, {u['latency']:.1f}s")
    return "\n".join(lines) + "\n"
