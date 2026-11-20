"""Scalar rewards built from verifier reports (used by RL, rejection sampling, reranking, data filters).

Design principles
* Ordering we want, and test (tests/test_latex_integration.py::test_reward_ordering):
      clean full section  >  full section with a local error  >  thin / empty answers
* "Substance" gate: every component is scaled by min(1, words / (0.5 * target)), so an empty
  \\section{} or three keyword bullets cannot collect structure/coverage points.
* Compile failures are *salvaged*: the failing top-level units are quarantined, the compilable rest is
  scored normally, then multiplied by the kept fraction and charged a fixed penalty. A long good
  section with one bad macro therefore beats an empty one but never a clean one.
* Anti-hacking: repetition penalty, deterministic-fix penalty, overfull boxes, missing planned figures.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import asdict, dataclass

from ..figures.runner import FigureResult
from ..schemas import Outline
from .verifier import SectionContext, SectionReport


@dataclass
class RewardWeights:
    figures: float = 0.25
    length: float = 0.2
    coverage: float = 0.25
    structure: float = 0.3
    fix_penalty: float = 0.015
    fix_penalty_cap: float = 0.15
    overfull_penalty: float = 0.03
    overfull_cap: float = 0.15
    # stricter than the original 0.08 / 3.0: RL policies drift towards self-repetition otherwise
    repetition_threshold: float = 0.05
    repetition_scale: float = 5.0
    compile_fail_penalty: float = 0.35
    cliche_free: int = 1          # one "In summary," is fine
    cliche_penalty: float = 0.03
    cliche_cap: float = 0.15
    min_substance_words: int = 120  # used when no target is known


def length_score(ratio: float | None) -> float:
    if ratio is None or ratio <= 0:
        return 0.0 if ratio is not None else 1.0
    return math.exp(-2.0 * abs(math.log(ratio)))  # 1.0 at target, 0.25 at x2 or x0.5


def _target_words(rep: SectionReport) -> float | None:
    if rep.length_ratio and rep.stats.words:
        return rep.stats.words / rep.length_ratio
    return None


def structure_score(rep: SectionReport) -> tuple[float, dict]:
    s = rep.stats
    target = _target_words(rep) or 400
    md_leaks = sum(1 for f in rep.fixes if f.code.startswith("md_"))
    checks = {  # exemptions depend on the *target*, which the model cannot game
        "has_section": bool(re.match(r"\s*\\section\*?\{", rep.source)),
        "has_box": sum(s.boxes.values()) >= 1,
        "box_variety": len(s.boxes) >= 2 or target < 250,
        "has_math_or_code": (s.display_math + s.inline_math + s.code_listings) > 0,
        "no_markdown": md_leaks == 0,
        "paragraph_prose": s.words >= 0.3 * target,
    }
    return sum(checks.values()) / len(checks), checks


def figure_score(rep: SectionReport, expected: int = 0) -> float:
    n = len(rep.figure_blocks)
    if n == 0:
        return 0.0 if expected > 0 else 1.0
    good = 0.0
    for b in rep.figure_blocks:
        r = rep.figure_results.get(b.id)
        if r is not None and r.passed:
            warn = sum(1 for i in r.issues if i.severity == "warning")
            good += max(0.6, 1.0 - 0.1 * warn)
    return good / max(n, expected)


def section_reward(rep: SectionReport, *, expected_figures: int = 0, w: RewardWeights | None = None,
                   salvage: tuple[SectionReport, float] | None = None) -> tuple[float, dict]:
    """Reward of one section report. For a non-compiling report pass `salvage` =
    (report of the quarantined, compilable remainder, kept fraction); see `score_section`."""
    w = w or RewardWeights()
    info: dict = {"compile_ok": rep.compile_ok}
    if not rep.compile_ok:
        n_err = max(1, len(rep.compile_issues))
        if salvage is None:
            r = -1.0 + 0.5 / n_err
        else:
            srep, kept = salvage
            sr, sinfo = section_reward(srep, expected_figures=expected_figures, w=w)
            r = max(-1.0, sr * kept - w.compile_fail_penalty)
            info.update(salvaged_reward=sr, kept=round(kept, 3))
        info.update(n_errors=n_err, reward=r)
        return r, info
    target = _target_words(rep) or w.min_substance_words * 2
    substance = min(1.0, rep.stats.words / (0.5 * target))
    comp = {
        "figures": figure_score(rep, expected_figures),
        "length": length_score(rep.length_ratio),
        "coverage": rep.coverage,
    }
    comp["structure"], info["structure_checks"] = structure_score(rep)
    total_w = w.figures + w.length + w.coverage + w.structure
    base = (w.figures * comp["figures"] + w.length * comp["length"] + w.coverage * comp["coverage"]
            + w.structure * comp["structure"]) / total_w
    pen_fix = min(w.fix_penalty_cap, w.fix_penalty * len(rep.fixes))
    pen_over = min(w.overfull_cap, w.overfull_penalty * rep.overfull)
    pen_rep = max(0.0, rep.repetition - w.repetition_threshold) * w.repetition_scale
    pen_cl = min(w.cliche_cap, w.cliche_penalty * max(0, getattr(rep.stats, "cliches", 0) - w.cliche_free))
    r = base * substance - pen_fix - pen_over - pen_rep - pen_cl
    info.update(components=comp, substance=round(substance, 3),
                penalties={"fixes": pen_fix, "overfull": pen_over, "repetition": pen_rep, "cliches": pen_cl}, reward=r)
    return r, info


def salvage_report(verifier, rep: SectionReport, ctx: SectionContext, max_rounds: int = 4,
                   figures_dir=None) -> tuple[SectionReport, float] | None:
    """Quarantine failing units until the section compiles; return (report, kept word fraction)."""
    from .edits import quarantine

    cur = rep
    for _ in range(max_rounds):
        bad = [i.line for i in cur.compile_issues] or [i.line for i in cur.latex_errors] or [None]
        src, _removed = quarantine(cur.source, bad)
        if src == cur.source:
            break
        cur = verifier.verify_section(src, ctx, figures_dir=figures_dir, prior_results=rep.figure_results)
        if cur.compile_ok:
            return cur, min(1.0, cur.stats.words / max(rep.stats.words, 1))
    return None


def score_section(verifier, rep: SectionReport, ctx: SectionContext, *, expected_figures: int = 0,
                  w: RewardWeights | None = None, figures_dir=None) -> tuple[float, dict]:
    """section_reward with salvage of non-compiling reports (runs extra compilations only then)."""
    salv = None if rep.compile_ok else salvage_report(verifier, rep, ctx, figures_dir=figures_dir)
    return section_reward(rep, expected_figures=expected_figures, w=w, salvage=salv)


def figure_reward(res: FigureResult) -> float:
    if not res.ok:
        return -1.0 if res.error and "Syntax" in res.error else -0.8
    errs = sum(1 for i in res.issues if i.severity == "error")
    warns = sum(1 for i in res.issues if i.severity == "warning")
    if errs:
        return -0.3 - 0.1 * min(errs, 4)
    return max(0.3, 1.0 - 0.12 * warns)


def _nonws(s: str) -> int:
    return len(re.sub(r"\s+", "", s))


def _canon(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def repair_reward(before: SectionReport, after: SectionReport, before_src: str, after_src: str,
                  gold_src: str | None = None) -> float:
    """Fix every error with a minimal edit and without deleting content.

    * content loss is measured in non-whitespace characters (formulas count), beyond a 10-char allowance
    * edit size is character-level (a two-line typo fix is cheap)
    * with a known gold repair (synthetic data), distance to the gold result is charged too
    """
    lost = max(0, _nonws(before_src) - _nonws(after_src))
    loss_pen = min(0.6, max(0, lost - 10) / 150)
    change = 1.0 - difflib.SequenceMatcher(None, before_src, after_src, autojunk=False).ratio()
    edit_pen = min(0.2, 2.0 * change)
    if not after.compile_ok:
        nb, na = len(before.latex_errors), len(after.latex_errors)
        base = 0.2 * (nb - na) / nb if (nb and na < nb) else -0.5
        return base - loss_pen - edit_pen
    r = 1.0 - loss_pen - edit_pen
    if gold_src is not None:
        a, g = _canon(after_src), _canon(gold_src)
        if a != g:
            sim = difflib.SequenceMatcher(None, a, g, autojunk=False).ratio()
            r -= min(0.4, 20 * (1 - sim))
    return r


def plan_reward(outline: Outline | None, *, total_words: int, n_sections: tuple[int, int],
                selected_slides: list[int] | None = None) -> tuple[float, dict]:
    if outline is None:
        return -1.0, {"valid": False}
    info: dict = {"valid": True}
    secs = outline.sections
    lo, hi = n_sections
    c_n = 1.0 if lo <= len(secs) <= hi else math.exp(-0.5 * min(abs(len(secs) - lo), abs(len(secs) - hi)))
    tw = sum(s.target_words for s in secs)
    c_words = length_score(tw / max(total_words, 1))
    c_kp = sum(1 for s in secs if 2 <= len(s.key_points) <= 8) / len(secs)
    nfig = sum(len(s.figures) for s in secs)
    c_fig = 1.0 if len(secs) * 0.3 <= nfig <= len(secs) * 2 else 0.4
    titles = [s.title.strip().lower() for s in secs]
    c_unique = len(set(titles)) / len(titles)
    c_meta = (bool(outline.abstract.strip()) + bool(outline.learning_objectives) + bool(outline.cheatsheet_focus)) / 3
    comps = {"n_sections": c_n, "words": c_words, "key_points": c_kp, "figures": c_fig, "unique": c_unique,
             "metadata": c_meta}
    if selected_slides:
        assigned = {n for s in secs for n in s.slides}
        comps["slides"] = len(assigned & set(selected_slides)) / len(set(selected_slides))
    r = sum(comps.values()) / len(comps)
    info["components"] = comps
    return r, info


def cheatsheet_reward(rep: SectionReport, *, page_budget: int, facts: list[str]) -> tuple[float, dict]:
    if not rep.compile_ok:
        return -1.0 + 0.5 / max(1, len(rep.compile_issues)), {"compile_ok": False}
    pages = rep.pages or 1
    c_fit = 1.0 if pages <= page_budget else math.exp(-1.5 * (pages - page_budget))
    boxes = rep.source.count("\\begin{csbox}")
    c_boxes = 1.0 if 6 <= boxes <= 30 else 0.5
    c_cov = rep.coverage if facts else 1.0
    r = 0.45 * c_fit + 0.2 * c_boxes + 0.35 * c_cov - min(0.15, 0.015 * len(rep.fixes))
    return r, {"pages": pages, "boxes": boxes, "coverage": c_cov, "reward": r}


def to_dict(w: RewardWeights) -> dict:
    return asdict(w)
