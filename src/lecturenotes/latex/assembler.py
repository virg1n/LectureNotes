"""Render documents from templates and manage build directories."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jinja2

from .capabilities import TexCapabilities
from .compiler import STY_FILE, TEMPLATE_DIR

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    block_start_string="<%", block_end_string="%>",
    variable_start_string="<<", variable_end_string=">>",
    comment_start_string="<#", comment_end_string="#>",
    trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
    autoescape=False, undefined=jinja2.StrictUndefined,
)

_SPECIALS = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}"}


def escape_text(s: str, allow_math: bool = True) -> str:
    """Escape LaTeX specials in short plain-text fields (titles, objectives) but keep
    $...$ math and existing escapes intact."""
    if not s:
        return ""
    s = str(s).replace("\r", "")
    parts = re.split(r"(\$[^$]+\$)", s) if allow_math else [s]
    out = []
    for p in parts:
        if allow_math and len(p) > 1 and p.startswith("$") and p.endswith("$"):
            out.append(p)
            continue
        buf = []
        i = 0
        while i < len(p):
            c = p[i]
            if c == "\\" and i + 1 < len(p):
                # keep simple macros like \LaTeX, \& as-is
                buf.append(p[i : i + 2])
                i += 2
                continue
            buf.append(_SPECIALS.get(c, c))
            i += 1
        out.append("".join(buf))
    return "".join(out)


def _with_flags(src: str, caps: TexCapabilities | None) -> str:
    if caps is None:
        return src
    flags = caps.flags_tex()
    if not flags:
        return src
    return re.sub(r"(\\usepackage(?:\[[^\]]*\])?\{lecturenotes\})", lambda m: flags + "\n" + m.group(1), src, count=1)


def render_notes(*, title: str, subtitle: str = "", meta: str = "", abstract: str = "",
                 objectives: Iterable[str] = (), prerequisites: Iterable[str] = (),
                 sections: Iterable[str], language: str = "en", toc: bool = True,
                 short_title: str | None = None, caps: TexCapabilities | None = None) -> str:
    tpl = _env.get_template("notes.tex.j2")
    src = tpl.render(
        title=escape_text(title), subtitle=escape_text(subtitle), meta=escape_text(meta),
        abstract=escape_text(abstract), objectives=[escape_text(o) for o in objectives],
        prerequisites=[escape_text(p) for p in prerequisites], sections=list(sections),
        language=language, toc=toc, short_title=escape_text(short_title or title)[:80],
    )
    return _with_flags(src, caps)


def render_fragment(section: str, *, language: str = "en", cheatsheet: bool = False,
                    caps: TexCapabilities | None = None, columns: int = 3, fontsize: str = "footnotesize") -> str:
    src = _env.get_template("fragment.tex.j2").render(section=section, language=language, cheatsheet=cheatsheet,
                                                      columns=columns, fontsize=fontsize)
    return _with_flags(src, caps)


def render_cheatsheet(*, title: str, meta: str = "", language: str = "en", columns: int = 3,
                      fontsize: str = "footnotesize", margin: str = "0.7cm",
                      caps: TexCapabilities | None = None) -> str:
    src = _env.get_template("cheatsheet.tex.j2").render(
        title=escape_text(title), meta=escape_text(meta), language=language, columns=columns,
        fontsize=fontsize, margin=margin)
    return _with_flags(src, caps)


def render_tikz_standalone(body: str, *, language: str = "en", caps: TexCapabilities | None = None) -> str:
    return _env.get_template("tikz_standalone.tex.j2").render(
        body=body, language=language,
        has_standalone=caps.has("standalone") if caps else True)


@dataclass
class BuildDir:
    """A self-contained LaTeX workspace: main file + style + sections/ + figures/."""

    root: Path

    @classmethod
    def create(cls, root: str | Path | None = None, prefix: str = "lnbuild-") -> "BuildDir":
        path = Path(root) if root else Path(tempfile.mkdtemp(prefix=prefix))
        (path / "sections").mkdir(parents=True, exist_ok=True)
        (path / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(STY_FILE, path / STY_FILE.name)
        return cls(path)

    @property
    def sections_dir(self) -> Path:
        return self.root / "sections"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    def write_section(self, sid: str, tex: str) -> Path:
        p = self.sections_dir / f"{sid}.tex"
        p.write_text(tex, encoding="utf-8")
        return p

    def write_main(self, tex: str, name: str = "main.tex") -> Path:
        p = self.root / name
        p.write_text(tex, encoding="utf-8")
        return p

    def link_figures(self, src_dir: Path, names: Iterable[str]) -> None:
        for n in names:
            s = Path(src_dir) / n
            if s.exists():
                shutil.copy2(s, self.figures_dir / n)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


_INPUT_RE = re.compile(r"^\\input\{sections/([^}]+)\}\s*$", re.M)


def flatten(main_tex: str, sections_dir: Path, sty_inline: bool = False) -> str:
    """Single-file export: inline every ``\\input{sections/x}``."""
    def repl(m: re.Match) -> str:
        p = Path(sections_dir) / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        body = p.read_text(encoding="utf-8") if p.exists() else f"% missing {p.name}"
        return f"% ---- {p.name}\n{body}\n"
    out = _INPUT_RE.sub(repl, main_tex)
    if sty_inline:
        sty = STY_FILE.read_text(encoding="utf-8")
        sty = sty.replace("\\NeedsTeXFormat{LaTeX2e}", "").replace("\\endinput", "")
        sty = re.sub(r"\\ProvidesPackage\{[^}]*\}\[[^\]]*\]", "", sty)
        sty = sty.replace("\\DeclareOption{cheatsheet}{\\ln@cheatsheettrue}\n\\ProcessOptions\\relax", "")
        def inline(m: re.Match) -> str:
            body = sty
            if m.group(1) and "cheatsheet" in m.group(1):
                body = body.replace("\\ln@cheatsheetfalse", "\\ln@cheatsheettrue", 1)
            return "\\makeatletter\n" + body + "\n\\makeatother"
        out = re.sub(r"\\usepackage(\[[^\]]*\])?\{lecturenotes\}", inline, out, count=1)
    return out
