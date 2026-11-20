"""Sandboxed LaTeX compilation with structured results."""

from __future__ import annotations

import functools
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import IS_WINDOWS, ProcResult, run_process, which
from .logparse import LogSummary, parse_log

TEMPLATE_DIR = Path(__file__).parent / "templates"
STY_FILE = TEMPLATE_DIR / "lecturenotes.sty"


@dataclass
class CompileResult:
    ok: bool
    pdf_path: Path | None
    log: LogSummary
    passes: int
    duration: float
    timed_out: bool = False
    returncode: int = 0
    log_tail: str = ""
    engine_output: str = ""

    @property
    def errors(self):
        return self.log.errors


@dataclass
class TexInfo:
    engine: str
    path: str | None
    distribution: str  # "texlive" | "miktex" | "unknown" | "missing"
    version: str = ""
    packages: dict[str, bool] = field(default_factory=dict)


REQUIRED_PACKAGES = [
    "amsmath.sty", "mathtools.sty", "xcolor.sty", "graphicx.sty", "booktabs.sty", "listings.sty",
    "multicol.sty", "geometry.sty", "hyperref.sty",
]
RECOMMENDED_PACKAGES = [
    "tikz.sty", "pgfplots.sty", "tcolorbox.sty", "standalone.cls", "siunitx.sty", "enumitem.sty",
    "titlesec.sty", "fancyhdr.sty", "caption.sty", "microtype.sty", "lmodern.sty",
]


class LatexCompiler:
    def __init__(self, engine: str = "pdflatex", timeout: float = 90.0, max_errors: int = 20):
        self.engine = engine
        self.timeout = timeout
        self.max_errors = max_errors
        self.exe = which(engine)

    # ------------------------------------------------------------------ environment
    @staticmethod
    def _env() -> dict[str, str]:
        env = dict(os.environ)
        # Disable log line wrapping (TeX Live honours these texmf.cnf variables from env).
        env.update({
            "max_print_line": "100000", "error_line": "254", "half_error_line": "238",
            "openout_any": "p", "shell_escape": "f",
        })
        # Do not leak API keys/tokens into a TeX run.
        for k in list(env):
            if any(s in k.upper() for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
                env.pop(k, None)
        return env

    @functools.cached_property
    def distribution(self) -> str:
        if not self.exe:
            return "missing"
        r = run_process([self.exe, "--version"], timeout=20)
        out = (r.stdout + r.stderr).lower()
        if "miktex" in out:
            return "miktex"
        if "tex live" in out or "texlive" in out:
            return "texlive"
        return "unknown"

    def info(self, probe_packages: bool = True) -> TexInfo:
        ver = ""
        if self.exe:
            r = run_process([self.exe, "--version"], timeout=20)
            ver = (r.stdout.splitlines() or [""])[0]
        info = TexInfo(self.engine, self.exe, self.distribution, ver)
        if probe_packages and self.exe:
            for p in REQUIRED_PACKAGES + RECOMMENDED_PACKAGES:
                info.packages[p] = self.has_file(p)
        return info

    @functools.lru_cache(maxsize=256)
    def has_file(self, name: str) -> bool:
        kpse = which("kpsewhich")
        if not kpse:
            return False
        # MiKTeX may try to auto-install on lookup; keep a short timeout.
        r = run_process([kpse, name], timeout=15)
        return r.returncode == 0 and bool(r.stdout.strip())

    # ------------------------------------------------------------------ compile
    def _cmd(self, main: str, draft: bool) -> list[str]:
        cmd = [self.exe or self.engine, "-interaction=nonstopmode", "-file-line-error", "-halt-on-error"]
        if self.distribution == "miktex":
            # never pop up the package-installer dialog (it blocks until timeout)
            cmd += ["-disable-write18", "-disable-installer"]
        else:
            cmd += ["-no-shell-escape"]
        if draft and self.engine == "pdflatex":
            cmd.append("-draftmode")
        cmd.append(main)
        return cmd

    def compile(self, workdir: str | Path, main: str = "main.tex", *, passes: int = 1,
                draft: bool = False, max_passes: int = 3, halt_on_error: bool = True) -> CompileResult:
        """Compile ``workdir/main``. ``passes`` is the minimum number of runs; reruns are
        added automatically (up to ``max_passes``) when LaTeX asks for them."""
        workdir = Path(workdir)
        if not self.exe:
            return CompileResult(False, None, LogSummary(fatal=True), 0, 0.0,
                                 log_tail=f"{self.engine} not found on PATH")
        stem = Path(main).stem
        log_path = workdir / f"{stem}.log"
        pdf_path = workdir / f"{stem}.pdf"
        total = 0.0
        last: ProcResult | None = None
        summary = LogSummary()
        run_idx = 0
        cmd = self._cmd(main, draft)
        if not halt_on_error:
            cmd = [c for c in cmd if c != "-halt-on-error"]
        while run_idx < max(passes, 1) or (summary.needs_rerun and run_idx < max_passes):
            if log_path.exists():
                log_path.unlink()
            last = run_process(cmd, cwd=workdir, timeout=self.timeout, env=self._env())
            run_idx += 1
            total += last.duration
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            summary = parse_log(text, max_errors=self.max_errors)
            if last.timed_out:
                summary.fatal = True
                from .logparse import TexMessage
                summary.errors.append(TexMessage(kind="error", message=f"Compilation timed out after {self.timeout:.0f}s (infinite loop?)", category="timeout"))
                break
            if summary.errors or summary.fatal:
                break
            if not summary.needs_rerun and run_idx >= passes:
                break
        assert last is not None
        has_pdf = draft or (pdf_path.exists() and pdf_path.stat().st_size > 0)
        ok = summary.ok and has_pdf and not last.timed_out and (last.returncode == 0 or not summary.errors)
        if last.returncode != 0 and not summary.errors and not last.timed_out:
            # engine failed without a parsable error (e.g. missing package auto-install refused)
            from .logparse import TexMessage
            tail = (last.stdout + last.stderr).strip().splitlines()[-5:]
            summary.errors.append(TexMessage(kind="error", message="LaTeX run failed: " + " | ".join(tail)[:300], category="other"))
            ok = False
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        return CompileResult(
            ok=ok, pdf_path=pdf_path if (has_pdf and not draft) else None, log=summary, passes=run_idx,
            duration=total, timed_out=last.timed_out, returncode=last.returncode,
            log_tail=log_text[-3000:], engine_output=(last.stdout[-2000:] if last else ""),
        )


def install_style(workdir: str | Path) -> None:
    """Copy the house style next to a document."""
    shutil.copy2(STY_FILE, Path(workdir) / STY_FILE.name)


@functools.lru_cache(maxsize=1)
def default_compiler() -> LatexCompiler:
    engine = os.environ.get("LN_LATEX_ENGINE", "pdflatex")
    return LatexCompiler(engine=engine, timeout=float(os.environ.get("LN_LATEX_TIMEOUT", "90")))


__all__ = ["LatexCompiler", "CompileResult", "TexInfo", "install_style", "default_compiler", "TEMPLATE_DIR",
           "IS_WINDOWS"]
