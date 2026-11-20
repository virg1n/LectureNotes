"""Cheat sheet: extract facts from the finished notes -> dense boxes -> fit to a page budget."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from ..config import HarnessConfig
from ..harness.edits import apply_model_repair, quarantine, strip_wrapping
from ..harness.rewards import cheatsheet_reward
from ..harness.verifier import SectionContext, SectionReport, Verifier
from ..llm.client import LLMClient
from ..prompts import templates as P
from ..schemas import LectureRequest, Outline

FACT_ENVS = ("definition", "theorem", "lemma", "proposition", "corollary", "keyidea", "warning", "summary")
LAYOUTS = [(3, "footnotesize"), (3, "scriptsize"), (4, "scriptsize"), (4, "tiny")]


_REF_PHRASE = re.compile(r"\s*\(?(?:see |cf\. )?(?:Figure|Fig\.|Table|Section|Sec\.|Eq\.|Equation|Definition|"
                         r"Theorem|Example)~?\s*\\(?:eq|auto)?ref\{[^}]*\}\)?|\s*\\(?:eq|auto)?ref\{[^}]*\}")


def _compact(tex: str, limit: int) -> str:
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)
    tex = _REF_PHRASE.sub("", tex)  # cross-references are meaningless on a cheat sheet
    tex = re.sub(r"<figure\b.*?</figure>", "", tex, flags=re.S | re.I)
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.M)
    tex = re.sub(r"\s+", " ", tex).strip()
    return tex if len(tex) <= limit else tex[:limit].rsplit(" ", 1)[0] + " ..."


def extract_facts(sources: dict[str, str], outline: Outline) -> tuple[str, list[str]]:
    lines: list[str] = []
    terms: list[str] = []
    for sid, src in sources.items():
        m = re.search(r"\\section\*?\{([^}]*)\}", src)
        lines.append(f"## {m.group(1) if m else sid}")
        for env in FACT_ENVS:
            for mm in re.finditer(rf"\\begin\{{{env}\}}(?:\[([^\]]*)\])?(.*?)\\end\{{{env}\}}", src, re.S):
                title = (mm.group(1) or "").strip()
                lines.append(f"[{env}] {title + ': ' if title else ''}{_compact(mm.group(2), 380)}")
                if title and env in ("definition", "theorem", "lemma", "proposition"):
                    terms.append(title)
        eqs = re.findall(r"\\\[(.*?)\\\]|\\begin\{(?:equation|align)\*?\}(.*?)\\end\{(?:equation|align)\*?\}", src, re.S)
        for a, b in eqs[:4]:
            lines.append(f"[formula] {_compact(a or b, 220)}")
        for t in re.findall(r"\\term\{([^}]*)\}", src)[:8]:
            terms.append(t)
    focus = list(outline.cheatsheet_focus)
    seen, uniq = set(), []
    for t in terms + focus:
        k = t.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(t)
    return "\n".join(lines), uniq[:30]


def clean_cheatsheet(text: str) -> str:
    text = _REF_PHRASE.sub("", strip_wrapping(text))
    boxes = re.findall(r"\\begin\{csbox\}.*?\\end\{csbox\}", text, re.S)
    if boxes:
        return "\n\n".join(b.strip() for b in boxes) + "\n"
    body = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}", "", text).strip()
    return f"\\begin{{csbox}}{{Summary}}\n{body}\n\\end{{csbox}}\n"


@dataclass
class CheatsheetOutcome:
    source: str
    report: SectionReport
    columns: int
    fontsize: str
    reward: float
    history: list[dict] = field(default_factory=list)

    @property
    def pages(self) -> int | None:
        return self.report.pages


class CheatsheetBuilder:
    def __init__(self, *, client: LLMClient, verifier: Verifier, cfg: HarnessConfig, req: LectureRequest,
                 outline: Outline):
        self.client, self.verifier, self.cfg, self.req, self.outline = client, verifier, cfg, req, outline
        self.terms: list[str] = []

    async def _verify(self, src: str, layout: tuple[int, str]) -> SectionReport:
        ctx = SectionContext(sid="cheatsheet", title=None, key_points=self.terms, language=self.req.language,
                             mode="cheatsheet", columns=layout[0], fontsize=layout[1])
        return await asyncio.to_thread(self.verifier.verify_section, src, ctx)

    async def build(self, sources: dict[str, str]) -> CheatsheetOutcome:
        budget = max(1, self.req.cheatsheet_pages)
        facts, self.terms = extract_facts(sources, self.outline)
        hist: list[dict] = []
        msgs = P.cheatsheet_messages(self.outline.title, facts, pages=budget, language=self.req.language,
                                     focus=self.outline.cheatsheet_focus)
        c = await self.client.complete_one("cheatsheet", msgs)
        src = clean_cheatsheet(c.text)
        li = 0
        rep = await self._verify(src, LAYOUTS[li])
        # the raw output is judged on its own merits (it becomes SFT data only if *it* is good)
        first_r, _ = cheatsheet_reward(rep, page_budget=budget, facts=self.terms)
        self.client.tracer.annotate(c.call_id, reward=first_r, pages=rep.pages, compile_ok=rep.compile_ok)

        # LaTeX repair, then quarantine as a guarantee
        for k in range(self.cfg.max_latex_repairs):
            if rep.compile_ok:
                break
            rc = await self.client.complete_one("repair_latex", P.repair_latex_messages(rep.source, rep.latex_feedback()),
                                                temperature=0.2 + 0.2 * k, meta={"section": "cheatsheet"})
            out = apply_model_repair(rep.source, rc.text, expect_prefix=None)
            if out.applied:
                new = await self._verify(out.source, LAYOUTS[li])
                ok = new.compile_ok or len(new.latex_errors) < len(rep.latex_errors)
                self.client.tracer.annotate(rc.call_id, accepted=ok, compile_ok=new.compile_ok)
                if ok:
                    rep = new
            hist.append({"step": "repair_latex", "compile_ok": rep.compile_ok})
        for _ in range(self.cfg.max_quarantine_rounds):
            if rep.compile_ok:
                break
            new_src, removed = quarantine(rep.source, [i.line for i in rep.compile_issues] or [None], keep_header=False)
            if new_src == rep.source or not new_src.strip():
                break
            rep = await self._verify(new_src, LAYOUTS[li])
            hist.append({"step": "quarantine", "removed": len(removed)})

        # fit to the page budget: cheap layout changes first, then model condensation
        condensed = 0
        while rep.compile_ok and (rep.pages or 1) > budget:
            if li + 1 < len(LAYOUTS) and (li < 2 or condensed >= self.cfg.cheatsheet_max_condense):
                li += 1
                rep = await self._verify(rep.source, LAYOUTS[li])
                hist.append({"step": "layout", "layout": LAYOUTS[li], "pages": rep.pages})
                continue
            if condensed >= self.cfg.cheatsheet_max_condense:
                break
            cc = await self.client.complete_one("condense_cheatsheet", P.condense_messages(
                rep.source, pages=rep.pages or budget + 1, budget=budget, language=self.req.language))
            new = await self._verify(clean_cheatsheet(cc.text), LAYOUTS[li])
            condensed += 1
            ok = new.compile_ok and (new.pages or 99) < (rep.pages or 99)
            self.client.tracer.annotate(cc.call_id, accepted=ok, pages_before=rep.pages, pages_after=new.pages)
            hist.append({"step": "condense", "accepted": ok, "pages": new.pages})
            if ok:
                rep = new
            elif li + 1 >= len(LAYOUTS):
                break
        reward, info = cheatsheet_reward(rep, page_budget=budget, facts=self.terms)
        self.client.tracer.annotate(c.call_id, final_reward=reward, final_pages=rep.pages)
        return CheatsheetOutcome(rep.source, rep, LAYOUTS[li][0], LAYOUTS[li][1], reward, hist)
