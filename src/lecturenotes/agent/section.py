"""Per-section pipeline: an explicit state machine driven by verifier feedback.

    DRAFT (best-of-n, reranked by reward)
      -> REPAIR_LATEX   (model SEARCH/REPLACE edits, accepted only if errors decrease)
      -> QUARANTINE     (deterministic removal of failing top-level units; guarantees compilation)
      -> REPAIR_FIGURES (model fixes code from runtime/lint feedback; regenerate from description; drop)
      -> REVISE         (length / key-point coverage / structure feedback; accepted only if reward improves)
      -> DONE

Every model call is annotated in the trace with what happened to its output.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import HarnessConfig
from ..figures.blocks import format_block, parse_blocks, remove_block, replace_block_source
from ..harness.edits import apply_model_repair, quarantine, strip_wrapping
from ..harness.rewards import score_section
from ..harness.verifier import SectionContext, SectionReport, Verifier
from ..ingest.slides import SlideDeck
from ..latex.assembler import escape_text
from ..latex.capabilities import TexCapabilities
from ..llm.client import LLMClient
from ..prompts import templates as P
from ..schemas import FigurePlan, LectureRequest, Outline, SectionPlan

log = logging.getLogger(__name__)


@dataclass
class SectionOutcome:
    plan: SectionPlan
    source: str
    report: SectionReport
    reward: float
    history: list[dict] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    dropped_figures: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        s = self.report.summary()
        # deterministic fixes happen on the draft; later verifications see already-fixed text
        s["n_fixes"] = sum(h.get("n_fixes", 0) for h in self.history) + s["n_fixes"]
        return {
            "id": self.plan.id, "title": self.plan.title, "target_words": self.plan.target_words,
            **s, "reward": round(self.reward, 3),
            "steps": [h["step"] for h in self.history], "quarantined_units": len(self.quarantined),
            "dropped_figures": self.dropped_figures,
        }


class SectionWorker:
    def __init__(self, *, client: LLMClient, verifier: Verifier, cfg: HarnessConfig, req: LectureRequest,
                 outline: Outline, deck: SlideDeck | None, caps: TexCapabilities | None, figures_dir: Path,
                 known_labels: set[str] | None = None):
        self.client = client
        self.verifier = verifier
        self.cfg = cfg
        self.req = req
        self.outline = outline
        self.deck = deck
        self.caps = caps
        self.figures_dir = figures_dir
        self.known_labels = known_labels or set()

    # ------------------------------------------------------------------ helpers
    def _ctx(self, sec: SectionPlan) -> SectionContext:
        return SectionContext(sid=sec.id, title=sec.title, target_words=sec.target_words, key_points=sec.key_points,
                              known_labels=self.known_labels, language=self.req.language)

    async def _verify(self, src: str, sec: SectionPlan, prior: SectionReport | None = None) -> SectionReport:
        return await asyncio.to_thread(
            self.verifier.verify_section, src, self._ctx(sec), figures_dir=self.figures_dir,
            prior_results=prior.figure_results if prior else None)

    async def _reward(self, rep: SectionReport, sec: SectionPlan) -> float:
        r, _ = await asyncio.to_thread(score_section, self.verifier, rep, self._ctx(sec),
                                       expected_figures=len(sec.figures), figures_dir=self.figures_dir)
        return r

    # ------------------------------------------------------------------ pipeline
    async def run(self, sec: SectionPlan) -> SectionOutcome:
        hist: list[dict] = []
        rep = await self._draft(sec, hist)
        rep = await self._repair_latex(sec, rep, hist)
        quarantined: list[str] = []
        if not rep.compile_ok:
            rep, quarantined = await self._quarantine(sec, rep, hist)
        rep, dropped = await self._repair_figures(sec, rep, hist)
        rep = await self._select_web_images(sec, rep, hist)
        if self.cfg.ensure_figures:
            rep = await self._ensure_figures(sec, rep, hist)
        for _ in range(self.cfg.revise_rounds):
            rep2 = await self._revise(sec, rep, hist)
            if rep2 is None:
                break
            rep = rep2
        reward = await self._reward(rep, sec)
        return SectionOutcome(sec, rep.source, rep, reward, hist, quarantined, dropped)

    # ------------------------------------------------------------------ DRAFT
    async def _draft(self, sec: SectionPlan, hist: list[dict]) -> SectionReport:
        slide_text = self.deck.text_for(sec.slides) if (self.deck is not None and sec.slides) else None
        msgs = P.section_messages(self.req, self.outline, sec, slide_text=slide_text, caps=self.caps,
                                  has_slide_images=bool(self.deck is not None and self.deck.images()))
        n = max(1, self.cfg.samples_per_section)
        first = 1 if (self.cfg.adaptive_sampling and n > 1) else n
        scored: list = []

        async def sample(k: int) -> None:
            comps = await self.client.complete("write_section", msgs, n=k, meta={"section": sec.id})
            reports = await asyncio.gather(*(self._verify(strip_wrapping(c.text), sec) for c in comps))
            for c, r in zip(comps, reports):
                rw = await self._reward(r, sec)
                scored.append((rw, c, r))
                self.client.tracer.annotate(c.call_id, reward=rw, verifier=r.summary(), truncated=c.truncated)

        await sample(first)
        # adaptive best-of-n: only pay for more drafts when the first one is not good enough
        if n > first and max(s[0] for s in scored) < self.cfg.accept_reward:
            await sample(n - first)
        scored.sort(key=lambda t: t[0], reverse=True)
        best_rw, best_c, best = scored[0]
        for rw, c, _ in scored:
            self.client.tracer.annotate(c.call_id, selected=(c is best_c))
        hist.append({"step": "draft", "n": len(scored), "rewards": [round(s[0], 3) for s in scored], **best.summary()})
        return best

    # ------------------------------------------------------------------ REPAIR_LATEX
    async def _repair_latex(self, sec: SectionPlan, rep: SectionReport, hist: list[dict]) -> SectionReport:
        for k in range(self.cfg.max_latex_repairs):
            if rep.compile_ok and not rep.latex_errors:
                break
            msgs = P.repair_latex_messages(rep.source, rep.latex_feedback())
            c = await self.client.complete_one("repair_latex", msgs, temperature=0.2 + 0.25 * k,
                                               meta={"section": sec.id, "round": k})
            out = apply_model_repair(rep.source, c.text)
            if out.applied == 0:
                self.client.tracer.annotate(c.call_id, accepted=False, reason=f"no applicable edit ({out.mode})")
                hist.append({"step": "repair_latex", "accepted": False, "reason": "no edit"})
                continue
            new = await self._verify(out.source, sec, prior=rep)
            before, after = len(rep.latex_errors), len(new.latex_errors)
            # a "repair" that deletes a large part of the section is not a repair (quarantine does that better)
            kept = new.stats.chars >= 0.85 * rep.stats.chars
            improved = kept and ((new.compile_ok and not rep.compile_ok) or after < before)
            self.client.tracer.annotate(c.call_id, accepted=improved, errors_before=before, errors_after=after,
                                        compile_ok=new.compile_ok, edit_mode=out.mode, content_kept=kept)
            hist.append({"step": "repair_latex", "accepted": improved, "errors": [before, after]})
            if improved:
                rep = new
        return rep

    # ------------------------------------------------------------------ QUARANTINE
    async def _quarantine(self, sec: SectionPlan, rep: SectionReport, hist: list[dict]) -> tuple[SectionReport, list[str]]:
        removed_all: list[str] = []
        for _ in range(self.cfg.max_quarantine_rounds):
            bad = [i.line for i in rep.compile_issues] or [i.line for i in rep.latex_errors] or [None]
            src, removed = quarantine(rep.source, bad)
            if src == rep.source:
                break
            removed_all += removed
            rep = await self._verify(src, sec, prior=rep)
            if rep.compile_ok:
                break
        if not rep.compile_ok:
            # last resort: a minimal, certainly-compilable section from the plan
            items = "\n".join(f"  \\item {escape_text(k)}" for k in sec.key_points) or "  \\item (content unavailable)"
            src = f"\\section{{{escape_text(sec.title)}}}\n\\begin{{itemize}}\n{items}\n\\end{{itemize}}\n"
            removed_all.append(rep.source)
            rep = await self._verify(src, sec)
        hist.append({"step": "quarantine", "removed_units": len(removed_all), "compile_ok": rep.compile_ok})
        return rep, removed_all

    # ------------------------------------------------------------------ REPAIR_FIGURES
    async def _try_figure(self, task: str, msgs: list[dict], fig_id: str, kind: str, sec: SectionPlan, k: int):
        c = await self.client.complete_one(task, msgs, temperature=0.3 + 0.2 * k,
                                           meta={"section": sec.id, "figure": fig_id, "round": k})
        blocks, _ = parse_blocks(c.text)
        if not blocks:
            self.client.tracer.annotate(c.call_id, accepted=False, reason="no <figure> block in output")
            return None
        nb = blocks[0]
        nb.id, nb.kind = fig_id, (nb.kind if nb.kind in ("matplotlib", "tikz") else kind)
        res = await asyncio.to_thread(self.verifier.renderer.render, nb, self.figures_dir / sec.id)
        self.client.tracer.annotate(c.call_id, accepted=res.passed, figure_ok=res.ok,
                                    figure_errors=[i.code for i in res.errors])
        return (nb, res) if res.passed else None

    async def _repair_figures(self, sec: SectionPlan, rep: SectionReport, hist: list[dict]) -> tuple[SectionReport, list[str]]:
        failed = rep.failed_figures
        if not failed:
            return rep, []
        src = rep.source
        dropped: list[str] = []
        # process from the end so earlier block offsets stay valid
        for b in sorted(failed, key=lambda b: b.start, reverse=True):
            res = rep.figure_results[b.id]
            fixed = None
            if b.kind == "web":  # no code to repair: ask for a better search query
                fixed = await self._web_figure(sec, b.id, b.caption or b.code, b.caption, b.width,
                                               context=src[max(0, b.start - 1500): b.start], failed=[b.code])
            elif b.kind != "slide":
                for k in range(self.cfg.max_figure_repairs):
                    msgs = P.repair_figure_messages(b.id, b.kind, b.caption, b.code, res.feedback(), caps=self.caps)
                    fixed = await self._try_figure("repair_figure", msgs, b.id, b.kind, sec, k)
                    if fixed:
                        break
                if fixed is None and self.cfg.regenerate_failed_figures:
                    plan = next((f for f in sec.figures if f.id == b.id), None)
                    fp = plan or FigurePlan(id=b.id, kind=b.kind, description=b.caption or sec.title)
                    msgs = P.figure_messages(self.req.topic, sec.title, fp, context=src[max(0, b.start - 1500): b.start],
                                             caps=self.caps)
                    fixed = await self._try_figure("figure", msgs, b.id, fp.kind, sec, 0)
            if fixed is not None:
                nb, _ = fixed
                nb.width = b.width
                src = replace_block_source(src, b, format_block(nb, caption=nb.caption or b.caption))
                hist.append({"step": "repair_figure", "figure": b.id, "accepted": True})
            elif self.cfg.drop_failed_figures:
                src = remove_block(src, b)
                dropped.append(b.id)
                hist.append({"step": "drop_figure", "figure": b.id})
        new = await self._verify(src, sec, prior=rep)
        if not new.compile_ok and rep.compile_ok:  # a new caption broke compilation: keep the old source
            return rep, dropped
        return new, dropped

    # ------------------------------------------------------------------ WEB IMAGES
    async def _web_figure(self, sec: SectionPlan, fid: str, description: str, caption: str, width: float | None,
                          *, context: str = "", failed: list[str] | None = None, rounds: int = 2):
        """Model writes a search query (task web_query) -> harness searches Wikimedia Commons. Queries that find
        an image are annotated accepted=True in the trace and become training data for web_query."""
        from ..figures.blocks import FigureBlock

        failed = [f for f in (failed or []) if f]
        for k in range(rounds):
            msgs = P.web_query_messages(self.req.topic, sec.title, description, context=context, failed=failed)
            c = await self.client.complete_one("web_query", msgs, temperature=0.3 + 0.3 * k,
                                               meta={"section": sec.id, "figure": fid})
            q = P.clean_query(c.text)
            if not q or q in failed:
                self.client.tracer.annotate(c.call_id, accepted=False, reason="empty or repeated query")
                continue
            got = await self._choose_image(sec, fid, caption or description, q, width, context=context)
            self.client.tracer.annotate(c.call_id, accepted=got is not None, query=q)
            if got is not None:
                return got
            failed.append(q)
        return None

    async def _choose_image(self, sec: SectionPlan, fid: str, caption: str, query: str, width: float | None, *,
                            context: str = ""):
        """Candidates for `query` -> model picks one or "none" (task pick_image) -> pinned block + render.

        With the teacher behind the client these answers are the labels the student learns from; with the
        student they are its own relevance judgement. Both models only see text metadata (title,
        description, categories, size), so training and inference conditions match."""
        from ..figures import web
        from ..figures.blocks import FigureBlock

        cache = self.verifier.renderer.cache_dir.parent / "web"
        cands = await asyncio.to_thread(web.candidates, query, 6, cache)
        if not cands:
            return None
        msgs = P.pick_image_messages(self.req.topic, sec.title, caption, web.describe_candidates(cands),
                                     context=context)
        c = await self.client.complete_one("pick_image", msgs, meta={"section": sec.id, "figure": fid, "query": query})
        choice, ok = P.parse_pick(c.text, len(cands))
        if not ok or choice is None:
            self.client.tracer.annotate(c.call_id, accepted=ok, choice=None, reason="none fits" if ok else "unparsable")
            return None
        title = cands[choice - 1]["title"]
        nb = FigureBlock(id=fid, kind="web", caption=caption, code=query, width=width or 0.6, start=0, end=0,
                         start_line=1, end_line=1, attrs={"file": title, "query": query})
        res = await asyncio.to_thread(self.verifier.renderer.render, nb, self.figures_dir / sec.id)
        self.client.tracer.annotate(c.call_id, accepted=res.passed, choice=choice, file=title)
        return (nb, res) if res.passed else None

    async def _select_web_images(self, sec: SectionPlan, rep: SectionReport, hist: list[dict]) -> SectionReport:
        """Every web figure written in a draft gets a relevance check: pick among candidates, re-query if
        none fits, drop the figure as a last resort."""
        todo = [b for b in rep.figure_blocks if b.kind == "web" and not (b.attrs or {}).get("file")]
        if not todo or not rep.compile_ok:
            return rep
        src = rep.source
        for b in sorted(todo, key=lambda b: b.start, reverse=True):
            ctx = src[max(0, b.start - 1500): b.start]
            got = await self._choose_image(sec, b.id, b.caption or b.code, b.code, b.width, context=ctx)
            if got is None:
                got = await self._web_figure(sec, b.id, b.caption or b.code, b.caption, b.width, context=ctx,
                                             failed=[b.code])
            if got is not None:
                src = replace_block_source(src, b, format_block(got[0], caption=b.caption or got[0].caption))
                hist.append({"step": "pick_image", "figure": b.id, "file": got[0].attrs.get("file")})
            else:
                src = remove_block(src, b)
                hist.append({"step": "drop_web_image", "figure": b.id})
        new = await self._verify(src, sec, prior=rep)
        return new if new.compile_ok else rep

    # ------------------------------------------------------------------ ENSURE_FIGURES
    async def _ensure_figures(self, sec: SectionPlan, rep: SectionReport, hist: list[dict]) -> SectionReport:
        """Planned figures the draft did not include are generated separately and inserted, so the
        notes stay visual even when the writer forgets the figure blocks."""
        if not rep.compile_ok:
            return rep
        have = {b.id for b in rep.figure_blocks if rep.figure_results.get(b.id) and rep.figure_results[b.id].passed}
        if have and len(have) >= min(len(sec.figures), 1):
            return rep
        missing = [f for f in sec.figures if f.id not in have][: max(1, len(sec.figures) - len(have))]
        src = rep.source
        added = 0
        for fp in missing:
            if fp.kind == "web":
                got = await self._web_figure(sec, fp.id, fp.description, fp.description[:200], 0.6, context=src[:2000])
            else:
                msgs = P.figure_messages(self.req.topic, sec.title, fp, context=src[:3000], caps=self.caps)
                got = await self._try_figure("figure", msgs, fp.id, fp.kind, sec, 0)
            if got is None:
                continue
            nb, _ = got
            block = format_block(nb, caption=nb.caption or fp.description[:200])
            ref = f"Figure~\\ref{{fig:{nb.id}}} illustrates this.\n"
            m = re.search(r"\\begin\{summary\}", src)
            pos = m.start() if m else len(src.rstrip())
            src = src[:pos].rstrip("\n") + "\n" + block + "\n" + ref + src[pos:]
            added += 1
        if not added:
            return rep
        new = await self._verify(src, sec, prior=rep)
        ok = new.compile_ok
        hist.append({"step": "ensure_figures", "added": added, "accepted": ok})
        return new if ok else rep

    # ------------------------------------------------------------------ REVISE
    def _content_feedback(self, sec: SectionPlan, rep: SectionReport) -> list[str]:
        fb = []
        tol = self.cfg.length_tolerance
        if rep.length_ratio is not None:
            if rep.length_ratio < 1 - tol:
                fb.append(f"The section has about {rep.stats.words} words of prose but should have about "
                          f"{sec.target_words}. Expand it: more explanation and intuition, a worked example, "
                          "connections to the rest of the lecture.")
            elif rep.length_ratio > 1 + 1.5 * tol:
                fb.append(f"The section has about {rep.stats.words} words but should have about {sec.target_words}. "
                          "Condense it without losing key points.")
        if rep.coverage < self.cfg.min_coverage and rep.missing_terms:
            fb.append("Explain these key points, which are currently missing or only mentioned: "
                      + "; ".join(rep.missing_terms[:6]) + ".")
        if not rep.stats.boxes and rep.stats.words > 150:
            fb.append("Add at least one definition, example or keyidea box.")
        if any(f.code.startswith("md_") for f in rep.fixes):
            fb.append("Do not use Markdown syntax; use LaTeX only.")
        return fb

    async def _revise(self, sec: SectionPlan, rep: SectionReport, hist: list[dict]) -> SectionReport | None:
        if not rep.compile_ok:
            return None
        fb = self._content_feedback(sec, rep)
        if not fb:
            return None
        msgs = P.revise_messages(rep.source, fb, language=self.req.language)
        c = await self.client.complete_one("revise_section", msgs, meta={"section": sec.id})
        out = apply_model_repair(rep.source, c.text)
        if out.applied == 0:
            self.client.tracer.annotate(c.call_id, accepted=False, reason="unusable output")
            hist.append({"step": "revise", "accepted": False})
            return None
        new = await self._verify(out.source, sec, prior=rep)
        old_r, new_r = await self._reward(rep, sec), await self._reward(new, sec)
        ok = new.compile_ok and new_r > old_r and len(new.failed_figures) <= len(rep.failed_figures)
        self.client.tracer.annotate(c.call_id, accepted=ok, reward_before=old_r, reward_after=new_r, feedback=fb)
        hist.append({"step": "revise", "accepted": ok, "reward": [round(old_r, 3), round(new_r, 3)]})
        return new if ok else None


def rename_duplicate_labels(sources: dict[str, str]) -> dict[str, str]:
    """Make \\label names unique across sections (the model writes sections independently)."""
    seen: set[str] = set()
    out = {}
    for sid, src in sources.items():
        labels = re.findall(r"\\label\{([^}]+)\}", src)
        for lab in labels:
            if lab in seen:
                new = f"{lab}-{sid}"
                src = src.replace(f"\\label{{{lab}}}", f"\\label{{{new}}}", 1)
                src = re.sub(r"\\(ref|eqref|autoref|pageref)\{" + re.escape(lab) + r"\}", rf"\\\1{{{new}}}", src)
                lab = new
            seen.add(lab)
        out[sid] = src
    return out
