"""Probe which LaTeX features *actually work* on this machine.

"Package file exists" is not enough: a stale MiKTeX can ship a new siunitx with
an old LaTeX3 kernel, a Debian TeX Live may lack Cyrillic fonts, etc. Each
feature is probed by compiling a tiny document. The result is cached per TeX
installation and used to (a) set ``\\lnno<feature>`` flags for the house style
and (b) tell the model which tools it may use.
"""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..utils import stable_hash
from .compiler import LatexCompiler, default_compiler

_DOC = "\\documentclass{article}\n%s\n\\begin{document}\n%s\n\\end{document}\n"

PROBES: dict[str, str] = {
    "mathtools": _DOC % ("\\usepackage{mathtools}", "$\\coloneqq$"),
    "booktabs": _DOC % ("\\usepackage{booktabs}", "\\begin{tabular}{c}\\toprule a\\\\\\bottomrule\\end{tabular}"),
    "tikz": _DOC % ("\\usepackage{tikz}\\usetikzlibrary{arrows.meta,positioning,calc}",
                    "\\begin{tikzpicture}\\node (a) {A}; \\node[right=of a] (b) {B}; \\draw[-Stealth] (a)--(b);\\end{tikzpicture}"),
    "pgfplots": _DOC % ("\\usepackage{pgfplots}\\pgfplotsset{compat=newest}",
                        "\\begin{tikzpicture}\\begin{axis}\\addplot[domain=0:1]{x^2};\\end{axis}\\end{tikzpicture}"),
    "tcolorbox": _DOC % ("\\usepackage{tcolorbox}\\tcbuselibrary{skins,breakable}",
                         "\\begin{tcolorbox}[enhanced,breakable,borderline west={2pt}{0pt}{red}]x\\end{tcolorbox}"),
    "siunitx": _DOC % ("\\usepackage{siunitx}", "\\SI{9.81}{\\metre\\per\\second\\squared} \\num{12345.6}"),
    "enumitem": _DOC % ("\\usepackage{enumitem}", "\\begin{enumerate}[label=(\\alph*)]\\item a\\end{enumerate}"),
    "titlesec": _DOC % ("\\usepackage{titlesec}", "\\section{A}"),
    "fancyhdr": _DOC % ("\\usepackage{fancyhdr}\\pagestyle{fancy}", "x"),
    "caption": _DOC % ("\\usepackage{caption}", "x"),
    "microtype": _DOC % ("\\usepackage{microtype}", "x"),
    "lmodern": _DOC % ("\\usepackage[T1]{fontenc}\\usepackage{lmodern}", "x"),
    "hyperref": _DOC % ("\\usepackage{hyperref}", "\\href{https://example.org}{x}"),
    "standalone": "\\documentclass[border=2pt]{standalone}\n\\begin{document}x\\end{document}\n",
    "cyrillic": _DOC % ("\\usepackage[T2A,T1]{fontenc}\\usepackage[utf8]{inputenc}\\usepackage[russian,english]{babel}",
                        "\\selectlanguage{russian}Привет, мир"),
}

# features the house style can switch off with \lnno<name>
STYLE_FLAGS = ("tikz", "pgfplots", "tcolorbox", "siunitx")


@dataclass
class TexCapabilities:
    engine: str
    distribution: str
    features: dict[str, bool] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return self.features.get(name, False)

    def flags_tex(self) -> str:
        """``\\def\\lnno<x>{}`` lines for every unavailable optional feature."""
        return "".join(f"\\def\\lnno{n}{{}}" for n in STYLE_FLAGS if not self.has(n))

    def prompt_notes(self) -> str:
        notes = []
        if not self.has("tikz"):
            notes.append("TikZ is unavailable: use kind=\"matplotlib\" for every figure.")
        elif not self.has("pgfplots"):
            notes.append("pgfplots is unavailable: draw plots with matplotlib, use TikZ only for diagrams.")
        if not self.has("siunitx"):
            notes.append("siunitx is unavailable: write units as \\,\\mathrm{m/s}.")
        if not self.has("enumitem"):
            notes.append("enumitem is unavailable: do not pass options to itemize/enumerate.")
        if not self.has("booktabs"):
            notes.append("booktabs is unavailable: use \\hline in tables.")
        return " ".join(notes)

    @classmethod
    def all_enabled(cls) -> "TexCapabilities":
        return cls("pdflatex", "assumed", {k: True for k in PROBES})


def _cache_path(compiler: LatexCompiler) -> Path:
    root = Path(os.environ.get("LN_CACHE_DIR", Path.home() / ".cache" / "lecturenotes"))
    key = stable_hash(compiler.exe, compiler.engine, sorted(PROBES))
    return root / f"texcaps-{key}.json"


def probe_feature(compiler: LatexCompiler, name: str, source: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="lnprobe-") as d:
        Path(d, "probe.tex").write_text(source, encoding="utf-8")
        res = compiler.compile(d, "probe.tex", passes=1)
        return res.ok


def detect_capabilities(compiler: LatexCompiler | None = None, *, refresh: bool = False,
                        workers: int = 6) -> TexCapabilities:
    compiler = compiler or default_compiler()
    if not compiler.exe:
        return TexCapabilities(compiler.engine, "missing", {k: False for k in PROBES})
    path = _cache_path(compiler)
    if path.exists() and not refresh:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if set(data.get("features", {})) == set(PROBES):
                return TexCapabilities(**data)
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {k: ex.submit(probe_feature, compiler, k, src) for k, src in PROBES.items()}
        feats = {k: f.result() for k, f in futs.items()}
    caps = TexCapabilities(compiler.engine, compiler.distribution, feats)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(caps), indent=1), encoding="utf-8")
    except OSError:
        pass
    return caps
