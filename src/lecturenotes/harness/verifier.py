"""Verifier: one call turns a model-written LaTeX fragment into a structured report.

    source --autofix--> fixed source --parse figure blocks--> render figures (sandbox, parallel)
           --substitute blocks (line-preserving)--> compile fragment in isolation (draft mode)
           --> lint/compile/figure issues with line numbers + stats + coverage

The same report drives (a) targeted repair prompts at inference time, (b) the
reward for RL and (c) acceptance filtering of teacher-generated training data.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..figures.blocks import FigureBlock, parse_blocks, replace_blocks
from ..figures.runner import FigureRenderer, FigureResult
from ..latex.assembler import BuildDir, escape_text, render_fragment
from ..latex.autofix import Fix, autofix
from ..latex.capabilities import TexCapabilities
from ..latex.compiler import LatexCompiler, default_compiler
from ..latex.logparse import TexMessage
from ..latex.textstats import TexStats, coverage, repetition_rate, tex_stats
from ..schemas import Issue

COMPILE_HINTS = {
    "undefined_cs": "this command does not exist here; use standard LaTeX/amsmath commands or the macros listed in the instructions",
    "missing_dollar": "a math-only symbol (_, ^, \\alpha, \\frac, ...) is outside math mode, or a formula is not closed",
    "env_mismatch": "\\begin{...} and \\end{...} do not match",
    "missing_brace": "unbalanced braces, or & / \\\\ used outside a table/alignment",
    "runaway": "an argument is missing its closing brace (or contains a blank line)",
    "missing_file": "do not reference external files",
    "unicode": "replace the Unicode character by a LaTeX command",
    "tikz": "check TikZ option names and syntax",
    "misplaced": "& and \\\\ are only allowed inside tables and align-like environments",
    "math_only": "this command must be inside $...$",
    "no_line": "\\\\ cannot start a paragraph or follow another \\\\; remove it",
    "illegal_param": "write \\# for a literal #",
    "double_script": "use braces: x_{i_j} or x^{a^b}",
    "timeout": "the fragment loops forever; simplify macros/TikZ",
}
STRUCTURAL_LINT = {"unclosed_brace", "unknown_env", "nested_math", "math_env_outside_math", "bad_env",
                   "unclosed_figure", "cyrillic", "md_fence", "bad_math_delim", "unicode_math"}


@dataclass
class SectionContext:
    sid: str = "s1"
    title: str | None = None
    target_words: int | None = None
    key_points: list[str] = field(default_factory=list)
    known_labels: set[str] = field(default_factory=set)
    language: str = "en"
    mode: str = "section"  # "section" | "cheatsheet"
    columns: int = 3  # cheat sheet layout
    fontsize: str = "footnotesize"


@dataclass
class SectionReport:
    source: str
    processed: str
    compile_ok: bool
    lint_issues: list[Issue]
    compile_issues: list[Issue]
    figure_blocks: list[FigureBlock]
    figure_results: dict[str, FigureResult]
    figure_parse_issues: list[Issue]
    fixes: list[Fix]
    stats: TexStats
    coverage: float = 1.0
    missing_terms: list[str] = field(default_factory=list)
    length_ratio: float | None = None
    repetition: float = 0.0
    overfull: int = 0
    pages: int | None = None
    duration: float = 0.0
    labels: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- views
    @property
    def lint_errors(self) -> list[Issue]:
        return [i for i in self.lint_issues if i.severity == "error"]

    @property
    def latex_errors(self) -> list[Issue]:
        return self.compile_issues + [i for i in self.lint_errors if i.code in STRUCTURAL_LINT]

    @property
    def failed_figures(self) -> list[FigureBlock]:
        return [b for b in self.figure_blocks if b.id in self.figure_results and not self.figure_results[b.id].passed]

    @property
    def figures_ok(self) -> int:
        return sum(1 for r in self.figure_results.values() if r.passed)

    @property
    def ok(self) -> bool:
        return self.compile_ok and not self.failed_figures

    def latex_feedback(self, max_items: int = 8) -> str:
        items = sorted(self.latex_errors, key=lambda i: (i.line or 10 ** 6))
        lines = [i.render() for i in items[:max_items]]
        return "\n".join(lines) if lines else "(no LaTeX errors)"

    def summary(self) -> dict:
        return {
            "compile_ok": self.compile_ok, "n_compile_errors": len(self.compile_issues),
            "n_lint_errors": len(self.lint_errors), "n_fixes": len(self.fixes),
            "figures": len(self.figure_blocks), "figures_ok": self.figures_ok,
            "words": self.stats.words, "length_ratio": self.length_ratio, "coverage": round(self.coverage, 3),
            "repetition": round(self.repetition, 3), "overfull": self.overfull,
        }


def _apply_relevance(res: FigureResult, verdict: dict) -> None:
    res.relevance = verdict["score"] / 5
    why = verdict.get("reason") or ""
    if verdict["score"] <= 2:
        res.issues.append(Issue(source="figure", severity="error", code="irrelevant_figure", figure_id=res.id,
                                message=f"the figure does not show what the caption describes ({why}); draw the "
                                        f"concept itself or change the figure"))
    elif verdict["score"] == 3:
        res.issues.append(Issue(source="figure", severity="warning", code="weak_figure", figure_id=res.id,
                                message=f"the figure only loosely illustrates the caption ({why})"))


def _render_key(b) -> str:
    return b.code + "|" + repr(sorted((b.attrs or {}).items()))


_NO_FLOAT_ENVS = {"definition", "theorem", "lemma", "proposition", "corollary", "example", "remark", "keyidea",
                  "warning", "summary", "exercise", "solution", "intuition", "note", "lnoverview", "csbox", "minipage",
                  "tcolorbox", "tabular", "multicols", "multicols*"}
_ENV_TOKEN = re.compile(r"\\(begin|end)\{([^}]+)\}")


def inside_box(src: str, pos: int) -> bool:
    """Is position `pos` inside an environment where floats are illegal?"""
    stack: list[str] = []
    for m in _ENV_TOKEN.finditer(re.sub(r"(?<!\\)%.*$", "", src[:pos], flags=re.M)):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif stack and stack[-1] == m.group(2):
            stack.pop()
    return any(e in _NO_FLOAT_ENVS for e in stack)


def compile_issue(e: TexMessage, line_offset: int = 0) -> Issue:
    line = (e.line - line_offset) if e.line else None
    msg = e.message
    if e.category == "undefined_cs" and e.culprit:
        msg = f"Undefined control sequence {e.culprit}"
    if e.context and e.category not in ("timeout",):
        msg += f"  (near: {e.context.strip()[:100]})"
    return Issue(source="compile", code=e.category, message=msg, line=line, hint=COMPILE_HINTS.get(e.category))


class Verifier:
    def __init__(self, *, compiler: LatexCompiler | None = None, caps: TexCapabilities | None = None,
                 renderer: FigureRenderer | None = None, language: str = "en",
                 slide_images: dict[int, Path] | None = None, figure_workers: int = 4,
                 work_root: str | Path | None = None, keep_dirs: bool = False):
        self.compiler = compiler or default_compiler()
        self.caps = caps
        self.language = language
        self.renderer = renderer or FigureRenderer(compiler=self.compiler, caps=caps, language=language)
        self.slide_images = slide_images or {}
        self.figure_workers = figure_workers
        self.work_root = Path(work_root) if work_root else None
        self.keep_dirs = keep_dirs
        # optional VLM check "does the figure show what the caption says" (LN_VLM_URL; see figures/relevance.py)
        from ..figures.relevance import FigureJudge

        self.judge = FigureJudge.from_env(self.renderer.cache_dir.parent / "figjudge")
        self.topic = ""

    # ---------------------------------------------------------------- figures
    def render_figures(self, blocks: list[FigureBlock], out_dir: Path, section: str = "") -> dict[str, FigureResult]:
        if not blocks:
            return {}
        with ThreadPoolExecutor(max_workers=self.figure_workers) as ex:
            futs = {b.id: ex.submit(self.renderer.render, b, out_dir, self.slide_images) for b in blocks}
            results = {k: f.result() for k, f in futs.items()}
            if self.judge is not None:
                todo = [b for b in blocks if b.kind != "slide" and results[b.id].passed and results[b.id].png_path]
                verdicts = dict(zip([b.id for b in todo], ex.map(
                    lambda b: self.judge.judge(results[b.id].png_path, b.caption, section, self.topic), todo)))
                for bid, v in verdicts.items():
                    if v:
                        _apply_relevance(results[bid], v)
            return results

    @staticmethod
    def figure_line(b: FigureBlock, r: FigureResult | None, subdir: str = "", inline: bool = False) -> str:
        cap = escape_text(b.caption, allow_math=True) if b.caption else ""
        if r is not None and r.ok and r.credit:
            cap = (cap + " " if cap else "") + "{\\footnotesize(" + escape_text(r.credit, allow_math=False) + ")}"
        here = "here" if inline else ""
        if r is not None and r.ok and r.pdf_path is not None:
            path = f"{subdir}/{r.pdf_path.name}" if subdir else r.pdf_path.name
            return f"\\lnfigure{here}{{{r.include_options(b)}}}{{{path}}}{{{cap}}}{{{b.label}}}"
        return f"\\lnmissingfigure{here}{{{cap}}}{{{b.label}}}"

    # ---------------------------------------------------------------- main entry
    def verify_section(self, source: str, ctx: SectionContext | None = None, *, render_figures: bool = True,
                       compile: bool = True, figures_dir: Path | None = None,
                       prior_results: dict[str, FigureResult] | None = None) -> SectionReport:
        ctx = ctx or SectionContext(language=self.language)
        t0 = time.monotonic()
        af = autofix(source, language=ctx.language, section_title=ctx.title, mode=ctx.mode,
                     known_labels=ctx.known_labels)
        src = af.source
        blocks, fig_parse_issues = parse_blocks(src)
        lint_issues = af.lint.issues if af.lint else []

        tmp_root = Path(tempfile.mkdtemp(prefix=f"lnv-{ctx.sid}-", dir=self.work_root))
        try:
            # per-section subdirectory: sections are processed concurrently and may reuse figure ids
            fig_dir = (figures_dir or (tmp_root / "figs")) / ctx.sid
            fig_dir.mkdir(parents=True, exist_ok=True)
            results: dict[str, FigureResult] = {}
            if render_figures:
                todo = []
                for b in blocks:
                    pr = (prior_results or {}).get(b.id)
                    # reuse only if the rendered file still exists (a previous verification may have used a
                    # scratch dir that is gone); re-rendering hits the renderer's content cache anyway
                    alive = not pr.ok or (pr.pdf_path is not None and pr.pdf_path.exists()) if pr else False
                    # the key includes the attributes: a web block pinned to a chosen file (`file=`) must not
                    # reuse the render of the same query before the choice (that showed the first search hit)
                    if pr is not None and pr.kind == b.kind and getattr(pr, "_code", None) == _render_key(b) and alive:
                        results[b.id] = pr
                    else:
                        todo.append(b)
                results.update(self.render_figures(todo, fig_dir, section=ctx.title or ""))
                for b in blocks:
                    setattr(results[b.id], "_code", _render_key(b))
            processed = replace_blocks(src, blocks, lambda b: self.figure_line(
                b, results.get(b.id), ctx.sid, inline=ctx.mode == "cheatsheet" or inside_box(src, b.start)))

            compile_ok, compile_issues, overfull, pages = True, [], 0, None
            if compile:
                bd = BuildDir.create(tmp_root / "build")
                bd.write_section(ctx.sid, processed)
                (bd.figures_dir / ctx.sid).mkdir(parents=True, exist_ok=True)
                for r in results.values():
                    if r.ok and r.pdf_path and r.pdf_path.exists():
                        shutil.copy2(r.pdf_path, bd.figures_dir / ctx.sid / r.pdf_path.name)
                bd.write_main(render_fragment(ctx.sid, language=ctx.language, cheatsheet=ctx.mode == "cheatsheet",
                                              caps=self.caps, columns=ctx.columns, fontsize=ctx.fontsize))
                cr = self.compiler.compile(bd.root, "main.tex", passes=1, draft=ctx.mode != "cheatsheet",
                                           halt_on_error=False)
                section_file = f"sections/{ctx.sid}.tex"
                for e in cr.errors:
                    if e.file and e.file.endswith(section_file):
                        compile_issues.append(compile_issue(e))
                    else:
                        iss = compile_issue(e)
                        iss.line = None
                        compile_issues.append(iss)
                compile_ok = cr.ok
                overfull = len([b for b in cr.log.overfull(8.0) if not b.file or b.file.endswith(section_file)])
                pages = cr.log.pages
        finally:
            # Rendered figures live in `figures_dir` when the caller wants them; scratch is always removed.
            if not self.keep_dirs:
                shutil.rmtree(tmp_root, ignore_errors=True)

        stats = tex_stats(src)
        cov, missing = coverage(ctx.key_points, src)
        report = SectionReport(
            source=src, processed=processed, compile_ok=compile_ok, lint_issues=lint_issues,
            compile_issues=compile_issues, figure_blocks=blocks, figure_results=results,
            figure_parse_issues=fig_parse_issues, fixes=af.fixes, stats=stats, coverage=cov,
            missing_terms=missing,
            length_ratio=(stats.words / ctx.target_words) if ctx.target_words else None,
            repetition=repetition_rate(src), overfull=overfull, pages=pages,
            duration=time.monotonic() - t0, labels=af.lint.labels if af.lint else [],
        )
        return report
