"""Planning: request (+ slides) -> validated Outline, with retries and a deterministic fallback."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..ingest.slides import SlideDeck
from ..latex.capabilities import TexCapabilities
from ..llm.client import LLMClient
from ..prompts.templates import plan_messages
from ..schemas import FigurePlan, LectureRequest, Outline, SectionPlan
from ..utils import extract_json_object
from .budget import LengthBudget

log = logging.getLogger(__name__)


def _normalize_plan(data: Any) -> dict:
    """Accept common deviations of small models (wrong key names, nesting)."""
    if isinstance(data, list):
        data = {"sections": data}
    if "outline" in data and isinstance(data["outline"], dict):
        data = data["outline"]
    data.setdefault("title", data.pop("lecture_title", None) or data.pop("topic", None) or "Lecture notes")
    secs = data.get("sections") or data.get("chapters") or data.get("parts") or []
    out = []
    for s in secs:
        if isinstance(s, str):
            s = {"title": s}
        s = dict(s)
        s.setdefault("title", s.pop("name", None) or s.pop("heading", None) or "Section")
        if "keypoints" in s:
            s["key_points"] = s.pop("keypoints")
        if "words" in s and "target_words" not in s:
            s["target_words"] = s.pop("words")
        figs = s.get("figures") or []
        s["figures"] = [f if isinstance(f, dict) else {"id": str(f)[:30], "description": str(f)} for f in figs]
        out.append(s)
    data["sections"] = out
    return data


def postprocess(outline: Outline, budget: LengthBudget, deck: SlideDeck | None,
                caps: TexCapabilities | None) -> Outline:
    lo, hi = budget.n_sections
    if len(outline.sections) > hi + 2:
        outline.sections = outline.sections[: hi + 2]
    # rescale target words to the budget, keeping the model's relative allocation
    total = sum(s.target_words for s in outline.sections) or 1
    for s in outline.sections:
        s.target_words = max(120, int(round(budget.words * s.target_words / total, -1)))
    if deck is not None and len(deck):
        valid = set(deck.indices)
        for s in outline.sections:
            s.slides = sorted(set(x for x in s.slides if x in valid))
        assigned = {x for s in outline.sections for x in s.slides}
        # unassigned slides go to the section whose slides are nearest in order
        for idx in sorted(valid - assigned):
            best, dist = outline.sections[0], 10 ** 9
            for k, s in enumerate(outline.sections):
                d = min((abs(idx - x) for x in s.slides), default=None)
                if d is None:
                    d = abs(idx - (k + 0.5) * len(valid) / len(outline.sections) - min(valid))
                if d < dist:
                    best, dist = s, d
            best.slides = sorted(set(best.slides) | {idx})
    if caps is not None and not caps.has("tikz"):
        for s in outline.sections:
            for f in s.figures:
                f.kind = "matplotlib"
    for s in outline.sections:
        s.figures = s.figures[:3]
        if not s.figures:  # visual notes: every section gets at least one planned figure
            kind = "tikz" if (caps is None or caps.has("tikz")) and re.search(
                r"structure|overview|architecture|process|cycle|algorithm|tree|graph|relation|pipeline", s.title, re.I) \
                else "matplotlib"
            desc = f"a {'diagram' if kind == 'tikz' else 'plot'} illustrating {', '.join(s.key_points[:2]) or s.title}"
            s.figures = [FigurePlan(id=f"{s.id}-fig", kind=kind, description=desc)]
    return Outline.model_validate(outline.model_dump())


def fallback_outline(req: LectureRequest, budget: LengthBudget, deck: SlideDeck | None) -> Outline:
    """Deterministic outline used when the model cannot produce valid JSON."""
    n = budget.mid_sections
    if deck is not None and len(deck):
        per = max(1, -(-len(deck.slides) // n))  # ceil division: consecutive slides per section
        groups = [deck.slides[i : i + per] for i in range(0, len(deck.slides), per)]
        secs = []
        for g in groups:
            titles = [s.title for s in g if s.title] or [f"Slides {g[0].index}-{g[-1].index}"]
            secs.append(SectionPlan(title=re.sub(r"\s+", " ", titles[0])[:80], key_points=titles[:6],
                                    slides=[s.index for s in g], target_words=budget.words // len(groups)))
    else:
        names = ["Motivation and overview", "Core definitions", "Main results", "Methods and techniques",
                 "Worked examples", "Common pitfalls", "Summary and outlook"]
        pick = names[: max(2, n - 1)] + [names[-1]]
        secs = [SectionPlan(title=t, key_points=[req.topic], target_words=budget.words // len(pick)) for t in pick]
    secs[0].figures = [FigurePlan(id="overview", kind="matplotlib", description=f"illustration of {req.topic}")]
    return Outline(title=req.topic, abstract=f"Lecture notes on {req.topic}.", sections=secs)


async def make_plan(client: LLMClient, req: LectureRequest, budget: LengthBudget, deck: SlideDeck | None,
                    caps: TexCapabilities | None = None, attempts: int = 3) -> tuple[Outline, dict]:
    digest = deck.digest() if deck is not None and len(deck) else None
    msgs = plan_messages(req, total_words=budget.words, n_sections=budget.n_sections, slide_digest=digest,
                         selected_slides=deck.indices if digest else None)
    info: dict = {"attempts": 0, "fallback": False, "errors": []}
    for k in range(attempts):
        info["attempts"] += 1
        c = await client.complete_one("plan", msgs, temperature=0.4 + 0.2 * k)
        try:
            data = _normalize_plan(extract_json_object(c.text))
            outline = Outline.model_validate(data)
            client.tracer.annotate(c.call_id, accepted=True)
            return postprocess(outline, budget, deck, caps), info
        except Exception as e:  # invalid JSON or schema
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            info["errors"].append(msg)
            client.tracer.annotate(c.call_id, accepted=False, error=msg)
            log.warning("plan attempt %d failed: %s", k + 1, msg)
    info["fallback"] = True
    return postprocess(fallback_outline(req, budget, deck), budget, deck, caps), info
