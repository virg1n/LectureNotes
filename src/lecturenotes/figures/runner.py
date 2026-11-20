"""Render figure blocks (matplotlib / TikZ / slide images) in a sandbox, with caching."""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..latex.assembler import render_tikz_standalone
from ..latex.capabilities import TexCapabilities
from ..latex.compiler import LatexCompiler, default_compiler
from ..schemas import Issue
from ..utils import IS_WINDOWS, run_process, stable_hash, which
from .blocks import FigureBlock
from .style import RCPARAMS, STYLE_VERSION, TEXTWIDTH_PT
from .validate import image_issues, text_overlap_issues

SANDBOX_ENTRY = Path(__file__).with_name("sandbox_entry.py")
DEFAULT_ALLOWED = ["numpy", "matplotlib", "math", "random", "itertools", "functools", "collections",
                   "statistics", "scipy", "cmath", "fractions", "decimal", "string", "textwrap", "datetime",
                   "mpl_toolkits", "cycler", "operator", "re", "heapq", "bisect", "copy", "dataclasses",
                   "typing", "enum", "warnings", "numbers", "networkx", "sympy", "pandas", "seaborn"]
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input",
                   "breakpoint", "memoryview", "setattr", "delattr"}
# module attributes that lead out of the sandbox (e.g. matplotlib.os, numpy.ctypeslib, np.lib.npyio)
BLOCKED_ATTRS = {"os", "sys", "subprocess", "builtins", "importlib", "ctypes", "ctypeslib", "shutil", "socket",
                 "pathlib", "io", "pickle", "marshal", "system", "popen", "spawn", "fork", "unlink",
                 "rmdir", "rmtree", "chmod", "chown", "environ", "load_library", "_ctypes", "fromfile", "tofile",
                 "savetxt", "loadtxt", "genfromtxt", "save", "savez", "load", "memmap", "f_globals", "f_locals",
                 "gi_frame", "cr_frame", "tb_frame"}


@dataclass
class FigureResult:
    id: str
    kind: str
    ok: bool
    pdf_path: Path | None = None
    png_path: Path | None = None
    issues: list[Issue] = field(default_factory=list)
    error: str | None = None
    error_line: int | None = None
    traceback: str = ""
    width_pt: float | None = None
    height_pt: float | None = None
    duration: float = 0.0
    cached: bool = False
    credit: str | None = None  # attribution for web images
    relevance: float | None = None  # VLM judge score / 5 (figures/relevance.py), when enabled

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def passed(self) -> bool:
        """Rendered and no error-level lint findings."""
        return self.ok and not self.errors

    def feedback(self) -> str:
        lines = []
        if self.error:
            loc = f" (line {self.error_line} of the figure code)" if self.error_line else ""
            lines.append(f"Execution failed{loc}: {self.error}")
            if self.traceback:
                lines.append(self.traceback.strip()[-700:])
        for i in self.issues:
            if i.code != "runtime_error":
                lines.append(i.render())
        return "\n".join(lines)

    def include_options(self, block: FigureBlock) -> str:
        if block.width:
            return f"width={block.width:.2f}\\linewidth"
        if self.width_pt and self.width_pt > 0.97 * TEXTWIDTH_PT:
            return "width=\\linewidth"
        if self.height_pt and self.height_pt > 560:
            return "height=0.6\\textheight,keepaspectratio"
        return ""


def static_check(code: str, allowed: set[str]) -> list[Issue]:
    """AST screening before running anything (fast feedback + defence in depth)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Issue(source="figure", code="syntax_error", line=e.lineno,
                      message=f"SyntaxError: {e.msg}: {(e.text or '').strip()[:80]}")]
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for nm in names:
                if nm and nm.split(".")[0] not in allowed:
                    issues.append(Issue(source="figure", code="forbidden_import", line=node.lineno,
                                        message=f"import of '{nm}' is not allowed",
                                        hint="use numpy / matplotlib / scipy / math only"))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            issues.append(Issue(source="figure", code="forbidden_call", line=node.lineno,
                                message=f"'{node.id}' is not allowed in figure code"))
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__") \
                and node.attr not in ("__init__", "__name__", "__doc__"):
            issues.append(Issue(source="figure", code="forbidden_attr", line=node.lineno,
                                message=f"dunder attribute access '{node.attr}' is not allowed"))
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRS:
            issues.append(Issue(source="figure", code="forbidden_attr", line=node.lineno,
                                message=f"attribute '{node.attr}' is not allowed in figure code"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr") \
                and len(node.args) >= 2 and not (isinstance(node.args[1], ast.Constant)
                                                 and isinstance(node.args[1].value, str)
                                                 and not node.args[1].value.startswith("_")
                                                 and node.args[1].value not in BLOCKED_ATTRS):
            issues.append(Issue(source="figure", code="forbidden_getattr", line=node.lineno,
                                message="getattr() is only allowed with a literal, public attribute name"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "savefig":
            issues.append(Issue(source="figure", severity="info", code="savefig", line=node.lineno,
                                message="savefig() is ignored: the harness saves the figure itself"))
    return issues


def _limits(timeout: float):  # pragma: no cover - POSIX only
    if IS_WINDOWS:
        return None

    def apply():
        import resource
        mem = 4 * 1024 ** 3
        for lim, val in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, int(timeout) + 5),
                         (resource.RLIMIT_FSIZE, 128 * 1024 ** 2), (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
    return apply


@functools.lru_cache(maxsize=1)
def _netns_prefix() -> tuple[str, ...]:
    """Use an unprivileged network namespace when the kernel allows it (no network for figure code)."""
    if IS_WINDOWS or os.environ.get("LN_SANDBOX_NET", "auto") == "off":
        return ()
    exe = which("unshare")
    if not exe:
        return ()
    r = run_process([exe, "-rn", "true"], timeout=5)
    return (exe, "-rn") if r.returncode == 0 else ()


def _clean_env(workdir: Path, mplconfig: Path) -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL", "COMSPEC", "PATHEXT",
            "LD_LIBRARY_PATH")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"HOME": str(workdir), "USERPROFILE": str(workdir), "MPLBACKEND": "Agg",
                "TMPDIR": str(workdir), "TEMP": str(workdir), "TMP": str(workdir),
                "MPLCONFIGDIR": str(mplconfig), "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


class FigureRenderer:
    def __init__(self, cache_dir: str | Path | None = None, *, timeout: float = 45.0,
                 allowed_modules: list[str] | None = None, compiler: LatexCompiler | None = None,
                 caps: TexCapabilities | None = None, language: str = "en", python: str | None = None):
        root = Path(cache_dir or os.environ.get("LN_CACHE_DIR", Path.home() / ".cache" / "lecturenotes"))
        self.cache_dir = root / "figures"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mplconfig = root / "mplconfig"
        self.mplconfig.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.allowed = set(allowed_modules or DEFAULT_ALLOWED)
        self.compiler = compiler or default_compiler()
        self.caps = caps
        self.language = language
        self.python = python or sys.executable

    # ------------------------------------------------------------------ public
    def render(self, block: FigureBlock, out_dir: str | Path, slide_images: dict[int, Path] | None = None) -> FigureResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if block.kind == "slide":
            return self._render_slide(block, out_dir, slide_images or {})
        if block.kind == "web":
            return self._render_web(block, out_dir)
        key = stable_hash(block.kind, block.code, STYLE_VERSION, self.language)
        cached = self.cache_dir / key
        meta_path = cached / "result.json"
        if meta_path.exists():
            res = self._load_cached(block, meta_path, out_dir)
            if res is not None:
                return res
        if block.kind == "tikz":
            res = self._render_tikz(block, cached)
        else:
            res = self._render_matplotlib(block, cached)
        self._store(res, meta_path)
        return self._publish(res, block, out_dir)

    # ------------------------------------------------------------------ cache
    def _store(self, res: FigureResult, meta_path: Path) -> None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"ok": res.ok, "error": res.error, "error_line": res.error_line, "traceback": res.traceback,
                "issues": [i.model_dump() for i in res.issues], "width_pt": res.width_pt,
                "height_pt": res.height_pt, "duration": res.duration}
        meta_path.write_text(json.dumps(data), encoding="utf-8")

    def _load_cached(self, block: FigureBlock, meta_path: Path, out_dir: Path) -> FigureResult | None:
        try:
            d = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        res = FigureResult(id=block.id, kind=block.kind, ok=d["ok"], error=d["error"], error_line=d["error_line"],
                           traceback=d.get("traceback", ""), issues=[Issue(**i) for i in d["issues"]],
                           width_pt=d["width_pt"], height_pt=d["height_pt"], duration=0.0, cached=True)
        res.pdf_path = meta_path.parent / "figure.pdf" if d["ok"] else None
        res.png_path = meta_path.parent / "figure.png" if d["ok"] else None
        if d["ok"] and not (res.pdf_path and res.pdf_path.exists()):
            return None
        for i in res.issues:
            i.figure_id = block.id
        return self._publish(res, block, out_dir)

    def _publish(self, res: FigureResult, block: FigureBlock, out_dir: Path) -> FigureResult:
        res.id = block.id
        for i in res.issues:
            i.figure_id = block.id
        if res.ok and res.pdf_path and res.pdf_path.exists():
            # content-addressed name: concurrent best-of-n drafts reuse the same ids, and a
            # shared "<id>.pdf" would let one draft's figure overwrite another's
            stem = f"{block.id}-{stable_hash(block.kind, block.code, STYLE_VERSION, self.language, n=8)}"
            dst = out_dir / f"{stem}.pdf"
            _atomic_copy(res.pdf_path, dst)
            if res.png_path and res.png_path.exists():
                png = out_dir / f"{stem}.png"
                _atomic_copy(res.png_path, png)
                res.png_path = png
            res.pdf_path = dst
        return res

    # ------------------------------------------------------------------ matplotlib
    def _render_matplotlib(self, block: FigureBlock, work: Path) -> FigureResult:
        res = FigureResult(id=block.id, kind="matplotlib", ok=False)
        static = static_check(block.code, self.allowed)
        blocking = [i for i in static if i.severity == "error"]
        if blocking:
            res.issues = static
            res.error = blocking[0].message
            res.error_line = blocking[0].line
            return res
        work.mkdir(parents=True, exist_ok=True)
        pdf, png = work / "figure.pdf", work / "figure.png"
        job = {"code": block.code, "out_pdf": str(pdf), "out_png": str(png), "rcparams": RCPARAMS,
               "allowed_modules": sorted(self.allowed)}
        with tempfile.TemporaryDirectory(prefix="lnfig-") as tmp:
            cmd = [*_netns_prefix(), self.python, "-I", str(SANDBOX_ENTRY)]
            pr = run_process(cmd, cwd=tmp, timeout=self.timeout, env=_clean_env(Path(tmp), self.mplconfig),
                             input_text=json.dumps(job), preexec_fn=_limits(self.timeout))
        res.duration = pr.duration
        if pr.timed_out:
            res.error = f"figure code timed out after {self.timeout:.0f}s (too much computation or an infinite loop)"
            res.issues = [Issue(source="figure", code="timeout", message=res.error)]
            return res
        payload = None
        for line in pr.stdout.splitlines():
            if line.startswith("@@LNRESULT@@"):
                payload = json.loads(line[len("@@LNRESULT@@"):])
        if payload is None:
            tail = (pr.stderr or pr.stdout).strip()[-600:]
            res.error = f"sandbox crashed (exit code {pr.returncode}): {tail or 'no output (memory limit?)'}"
            res.issues = [Issue(source="figure", code="sandbox_crash", message=res.error[:300])]
            return res
        res.error, res.error_line, res.traceback = payload["error"], payload["error_line"], payload.get("traceback", "")
        res.issues = [i for i in static if i.severity != "error"]
        if res.error:
            res.issues.append(Issue(source="figure", code="runtime_error", line=res.error_line, message=res.error))
            return res
        for code, sev, msg in payload["lint"]:
            res.issues.append(Issue(source="figure", severity=sev if sev in ("error", "warning", "info") else "warning",
                                    code=code, message=msg))
        res.issues += image_issues(png)
        res.ok = pdf.exists() and pdf.stat().st_size > 0
        res.pdf_path, res.png_path = pdf, png
        if payload.get("size_in"):
            res.width_pt, res.height_pt = payload["size_in"][0] * 72, payload["size_in"][1] * 72
        return res

    # ------------------------------------------------------------------ tikz
    @staticmethod
    def prepare_tikz(code: str) -> str:
        body = code.strip()
        if "\\begin{document}" in body:
            body = body.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0].strip()
        body = re.sub(r"^\s*\\(?:usepackage|usetikzlibrary|usepgfplotslibrary|pgfplotsset|documentclass)\b.*$", "",
                      body, flags=re.M).strip()
        if "\\begin{tikzpicture}" not in body:
            body = "\\begin{tikzpicture}\n" + body + "\n\\end{tikzpicture}"
        return body

    def _render_tikz(self, block: FigureBlock, work: Path) -> FigureResult:
        from ..latex.lint import lint_latex

        res = FigureResult(id=block.id, kind="tikz", ok=False)
        if self.caps is not None and not self.caps.has("tikz"):
            res.error = "TikZ is not available on this system; use kind=\"matplotlib\""
            res.issues = [Issue(source="figure", code="tikz_unavailable", message=res.error)]
            return res
        body = self.prepare_tikz(block.code)
        lint = lint_latex(body, mode="figure", language=self.language)
        hard = [f for f in lint.findings if f.severity == "error" and f.code in (
            "forbidden", "unclosed_env", "unmatched_end", "env_mismatch", "unclosed_brace", "extra_brace")]
        if hard:
            res.error = "; ".join(f.message for f in hard[:3])
            res.error_line = hard[0].line
            res.issues = [f.to_issue() for f in hard]
            for i in res.issues:
                i.source = "figure"
            return res
        src = render_tikz_standalone(body, language=self.language, caps=self.caps)
        offset = src[: src.index(body)].count("\n") if body in src else 0
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lntikz-") as tmp:
            Path(tmp, "fig.tex").write_text(src, encoding="utf-8")
            cr = self.compiler.compile(tmp, "fig.tex", passes=1)
            res.duration = cr.duration
            if not cr.ok:
                e = cr.errors[0] if cr.errors else None
                line = (e.line - offset) if (e and e.line and e.line > offset) else None
                res.error = (e.message + (f" `{e.culprit}`" if e and e.culprit and e.category == "undefined_cs" else "")) if e else "TikZ compilation failed"
                if e and e.context:
                    res.traceback = f"near: {e.context[:200]}"
                res.error_line = line
                res.issues = [Issue(source="figure", code=f"tikz_{e.category if e else 'error'}", line=line,
                                    message=res.error)]
                return res
            pdf = work / "figure.pdf"
            shutil.copy2(cr.pdf_path, pdf)
        png = work / "figure.png"
        try:
            import pymupdf

            doc = pymupdf.open(str(pdf))
            if self.caps is not None and not self.caps.has("standalone"):
                _crop_pdf_inplace(doc, pdf)
                doc = pymupdf.open(str(pdf))
            page = doc[0]
            res.width_pt, res.height_pt = page.rect.width, page.rect.height
            page.get_pixmap(dpi=72).save(str(png))
            if doc.page_count > 1:
                res.issues.append(Issue(source="figure", severity="warning", code="multi_page",
                                        message="TikZ figure produced several pages"))
            doc.close()
        except Exception as e:  # rendering previews is best-effort
            res.issues.append(Issue(source="figure", severity="info", code="preview_failed", message=str(e)[:200]))
        if png.exists():
            res.issues += image_issues(png)
        res.issues += text_overlap_issues(pdf)
        if res.width_pt and res.height_pt:
            ar = res.height_pt / max(res.width_pt, 1)
            if ar > 2.8 or ar < 0.12:
                res.issues.append(Issue(source="figure", severity="warning", code="aspect_ratio",
                                        message=f"unusual aspect ratio {res.width_pt:.0f}x{res.height_pt:.0f}pt"))
        res.ok = True
        res.pdf_path, res.png_path = pdf, png if png.exists() else None
        return res

    # ------------------------------------------------------------------ web images
    def _render_web(self, block: FigureBlock, out_dir: Path) -> FigureResult:
        from .web import fetch_image

        res = FigureResult(id=block.id, kind="web", ok=False)
        query = block.code.strip() or block.caption
        if not query:
            res.error = "web figure needs a query attribute"
            res.issues = [Issue(source="figure", code="no_query", message=res.error)]
            return res
        pinned = (block.attrs or {}).get("file")  # a model-chosen Commons file (task pick_image)
        if pinned:
            from .web import fetch_file
            img = fetch_file(pinned, out_dir, self.cache_dir.parent / "web")
        else:
            img = fetch_image(query, out_dir, self.cache_dir.parent / "web")
        if img is None:
            res.error = f"no freely licensed image found for query '{query}' (try a simpler, more common query)"
            res.issues = [Issue(source="figure", code="web_not_found", message=res.error)]
            return res
        res.ok, res.pdf_path, res.png_path, res.credit = True, img.path, img.path, img.attribution()
        res.issues = image_issues(img.path)
        return res

    # ------------------------------------------------------------------ slides
    def _render_slide(self, block: FigureBlock, out_dir: Path, slide_images: dict[int, Path]) -> FigureResult:
        res = FigureResult(id=block.id, kind="slide", ok=False)
        img = slide_images.get(block.slide_no or -1)
        if img is None or not Path(img).exists():
            res.error = f"slide {block.slide_no} is not available"
            res.issues = [Issue(source="figure", code="no_slide", message=res.error)]
            return res
        dst = out_dir / f"{block.id}{Path(img).suffix}"
        shutil.copy2(img, dst)
        res.ok, res.pdf_path, res.png_path = True, dst, dst
        return res


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy via a temp file + rename so concurrent readers never see a half-written figure."""
    if dst.exists():
        return
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{id(src)}.tmp")
    shutil.copy2(src, tmp)
    try:
        os.replace(tmp, dst)
    except OSError:  # another thread won the race (Windows cannot replace an open file)
        tmp.unlink(missing_ok=True)


def _crop_pdf_inplace(doc, path: Path) -> None:
    """Crop an article-class page to the drawn content (fallback when 'standalone' is missing)."""
    page = doc[0]
    rects = [d["rect"] for d in page.get_drawings()] + [b[:4] for b in page.get_text("blocks")]
    if not rects:
        return
    import pymupdf

    r = pymupdf.Rect(rects[0])
    for x in rects[1:]:
        r |= pymupdf.Rect(x)
    r = (r + (-4, -4, 4, 4)) & page.rect
    page.set_cropbox(r)
    tmp = path.with_suffix(".crop.pdf")
    doc.save(str(tmp))
    doc.close()
    os.replace(tmp, path)
