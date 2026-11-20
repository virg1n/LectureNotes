"""Entry point executed in a *separate, isolated* interpreter to render one figure.

    python -I sandbox_entry.py  < job.json   ->   stdout: "@@LNRESULT@@{json}"

It must not import anything from the lecturenotes package (the child runs with
-I and a scrubbed environment). Besides executing the code it performs a
"figure lint": empty axes, missing labels, overlapping text, tiny fonts, NaN
series -- problems a VLM would notice but which are cheap to detect from the
matplotlib object model directly.
"""

import builtins
import contextlib
import io
import json
import math
import sys
import traceback

MARK = "@@LNRESULT@@"
_BLOCKED_BUILTINS = ("open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit", "help",
                     "globals", "locals", "vars", "memoryview", "__loader__", "__spec__")


def _install_audit_hook(write_dirs, read_dirs):
    """Runtime guard (defence in depth behind the AST check): no processes, sockets, ctypes;
    writes only below `write_dirs`; reads only below `read_dirs` (Python, matplotlib data, fonts,
    sandbox cwd). Blocks e.g. `matplotlib.os.system(...)` or drawing ~/.ssh/id_rsa into a plot."""
    import os
    import sys as _sys

    norm = lambda p: os.path.normcase(os.path.realpath(p))  # noqa: E731
    wdirs = tuple(norm(d) + os.sep for d in write_dirs)
    rdirs = tuple(norm(d) + os.sep for d in read_dirs) + wdirs
    banned = ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "subprocess.Popen",
              "socket.", "ctypes.", "os.startfile", "winreg.", "os.kill", "os.putenv", "os.unsetenv",
              "shutil.rmtree", "os.remove", "os.rmdir", "os.rename", "os.symlink", "os.link", "os.chmod",
              "os.chown", "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect",
              "webbrowser.open", "import_module_from_path")

    def hook(event, args):
        if event.startswith(banned):
            raise PermissionError(f"blocked in figure sandbox: {event}")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = args[0].decode() if isinstance(args[0], bytes) else args[0]
            if path in ("/dev/null", "nul", "NUL"):
                return
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            p = norm(path) + os.sep  # trailing sep: "/a/b" must not match allowed dir "/a/bc/"
            if not p.startswith(wdirs if writing else rdirs):
                raise PermissionError(f"blocked in figure sandbox: {'write' if writing else 'read'} {path}")

    _sys.addaudithook(hook)


def _emit(result):
    sys.stdout.write(MARK + json.dumps(result) + "\n")
    sys.stdout.flush()


def _figure_lint(fig, plt, mtext):
    issues = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [a for a in fig.axes if a.get_visible()]
    if not axes:
        issues.append(["no_axes", "warning", "figure contains no axes"])
    labeled_series = 0
    for ax in axes:
        is_cbar = getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>"
        if is_cbar:
            continue
        has_data = bool(ax.lines or ax.collections or ax.patches or ax.images or ax.texts or ax.tables
                        or getattr(ax, "containers", None) or ax.artists)
        title = ax.get_title() or "untitled"
        if not has_data:
            issues.append(["empty_axes", "error", f"axes '{title}' contains no data"])
        if ax.axison and ax.name == "rectilinear" and has_data:
            no_x = not ax.get_xlabel().strip() and len(ax.get_xticks()) > 0 and ax.xaxis.get_visible()
            no_y = not ax.get_ylabel().strip() and len(ax.get_yticks()) > 0 and ax.yaxis.get_visible()
            if no_x and no_y:
                issues.append(["no_axis_labels", "warning", f"axes '{title}' has no axis labels"])
        for ln in ax.lines:
            try:
                ys = [float(v) for v in ln.get_ydata()]
            except Exception:
                continue
            if ys and all(not math.isfinite(v) for v in ys):
                issues.append(["nan_series", "error", f"a plotted line in '{title}' has only NaN/inf values"])
        handles, labels = ax.get_legend_handles_labels()
        labeled = [l for l in labels if l and not l.startswith("_")]
        labeled_series += len(labeled)
        if len(labeled) >= 2 and ax.get_legend() is None and fig.legends == []:
            issues.append(["missing_legend", "warning", f"axes '{title}' has labeled series but no legend()"])

    texts = []
    for t in fig.findobj(mtext.Text):
        try:
            if not t.get_visible() or not t.get_text().strip() or (t.get_alpha() is not None and t.get_alpha() == 0):
                continue
            bb = t.get_window_extent(renderer)
            if bb.width > 0 and bb.height > 0 and math.isfinite(bb.x0) and math.isfinite(bb.y0):
                texts.append((t, bb))
        except Exception:
            continue
    overlaps = []
    for i in range(len(texts)):
        ti, bi = texts[i]
        for j in range(i + 1, len(texts)):
            tj, bj = texts[j]
            w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if w > 0 and h > 0:
                inter = w * h
                small = min(bi.width * bi.height, bj.width * bj.height)
                if small > 0 and inter > 0.2 * small:
                    overlaps.append((ti.get_text()[:30], tj.get_text()[:30]))
    if overlaps:
        a, b = overlaps[0]
        sev = "error" if len(overlaps) > 3 else "warning"
        issues.append(["text_overlap", sev,
                       f"{len(overlaps)} pairs of overlapping text labels, e.g. '{a}' and '{b}' "
                       "(rotate tick labels, reduce ticks, or move annotations)"])
    tiny = [t.get_text()[:20] for t, _ in texts if t.get_fontsize() < 6.5]
    if tiny:
        issues.append(["tiny_font", "warning", f"{len(tiny)} text elements use font size < 6.5pt, e.g. '{tiny[0]}'"])
    if len([a for a in axes if a.get_label() != "<colorbar>"]) > 6:
        issues.append(["too_many_subplots", "warning", f"{len(axes)} subplots; more than 6 become unreadable"])
    w, h = fig.get_size_inches()
    if w > 7.5:
        issues.append(["too_wide", "warning", f"figure is {w:.1f}in wide; text width is ~6.4in, fonts will shrink"])
    if h > 7.5:
        issues.append(["too_tall", "warning", f"figure is {h:.1f}in tall"])
    return issues, labeled_series


def main():
    cfg = json.loads(sys.stdin.read())
    result = {"ok": False, "error": None, "error_line": None, "traceback": "", "lint": [], "n_figures": 0,
              "stdout": "", "size_in": None}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.text as mtext
        from cycler import cycler
        from matplotlib.figure import Figure
    except Exception as e:  # pragma: no cover
        result["error"] = f"matplotlib is not available in the sandbox: {e}"
        _emit(result)
        return

    rc = dict(cfg.get("rcparams", {}))
    cycle = rc.pop("axes.prop_cycle_colors", None)
    for k, v in rc.items():
        try:
            plt.rcParams[k] = v
        except Exception:
            pass
    if cycle:
        plt.rcParams["axes.prop_cycle"] = cycler(color=cycle)

    allowed = set(cfg.get("allowed_modules", []))
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level == 0 and root not in allowed:
            raise ImportError(f"import of '{name}' is not allowed in figure code; allowed modules: "
                              + ", ".join(sorted(allowed)))
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = {k: v for k, v in builtins.__dict__.items() if k not in _BLOCKED_BUILTINS}
    safe_builtins["__import__"] = guarded_import

    orig_savefig = Figure.savefig
    Figure.savefig = lambda self, *a, **k: None
    plt.savefig = lambda *a, **k: None
    plt.show = lambda *a, **k: None
    plt.close = lambda *a, **k: None  # keep figures alive so the harness can save them

    import os
    import tempfile

    import matplotlib.font_manager as fm

    out_dir = os.path.dirname(os.path.abspath(cfg["out_pdf"]))
    read_dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, matplotlib.get_data_path(),
                 matplotlib.get_configdir(), matplotlib.get_cachedir()}
    read_dirs |= {p for p in sys.path if p and os.path.isdir(p)}
    read_dirs |= {os.path.dirname(f.fname) for f in fm.fontManager.ttflist}
    try:
        import numpy

        read_dirs.add(os.path.dirname(os.path.dirname(numpy.__file__)))
    except Exception:
        pass
    _install_audit_hook(write_dirs=[os.getcwd(), out_dir, matplotlib.get_cachedir(), tempfile.gettempdir()],
                        read_dirs=sorted(read_dirs))

    g = {"__name__": "__main__", "__builtins__": safe_builtins}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(cfg["code"], "<figure>", "exec"), g)
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e.msg}"
        result["error_line"] = e.lineno
        result["traceback"] = f"line {e.lineno}: {e.text or ''}".strip()
        _emit(result)
        return
    except BaseException as e:  # noqa: BLE001 -- report everything, including SystemExit
        tb = traceback.extract_tb(e.__traceback__)
        frames = [fr for fr in tb if fr.filename == "<figure>"]
        result["error"] = f"{type(e).__name__}: {e}"[:600]
        result["error_line"] = frames[-1].lineno if frames else None
        # Only the user's own failing line + the library function that raised (no file paths).
        src_lines = cfg["code"].splitlines()
        ctx = []
        if frames and 0 < frames[-1].lineno <= len(src_lines):
            ctx.append(f"line {frames[-1].lineno}: {src_lines[frames[-1].lineno - 1].strip()}")
        lib = [fr for fr in tb if fr.filename != "<figure>"]
        if lib:
            ctx.append(f"raised inside {lib[-1].name}()")
        result["traceback"] = "\n".join(ctx)
        result["stdout"] = buf.getvalue()[-500:]
        _emit(result)
        return

    result["stdout"] = buf.getvalue()[-500:]
    nums = plt.get_fignums()
    if not nums:
        result["error"] = "the code did not create a matplotlib figure (use plt.subplots())"
        _emit(result)
        return
    figs = [plt.figure(n) for n in nums]
    result["n_figures"] = len(figs)

    def weight(f):
        return sum(len(a.lines) + len(a.collections) + len(a.patches) + len(a.images) for a in f.axes)

    fig = max(figs, key=weight)
    try:
        issues, _ = _figure_lint(fig, plt, mtext)
    except Exception as e:  # lint must never kill rendering
        issues = [["lint_failed", "info", f"figure lint failed: {e}"]]
    if len(figs) > 1:
        issues.append(["multiple_figures", "warning", f"{len(figs)} figures were created; only the richest one is used"])
    try:
        orig_savefig(fig, cfg["out_pdf"], format="pdf", bbox_inches="tight")
        orig_savefig(fig, cfg["out_png"], format="png", dpi=72, bbox_inches="tight")
    except Exception as e:
        result["error"] = f"saving the figure failed: {type(e).__name__}: {e}"[:600]
        _emit(result)
        return
    result.update(ok=True, lint=issues, size_in=list(map(float, fig.get_size_inches())))
    _emit(result)


if __name__ == "__main__":
    main()
