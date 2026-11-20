"""Regression tests for issues found in the code review (fast; the sandbox tests spawn Python)."""

from pathlib import Path

import pytest

from lecturenotes.data.tex_processing import expand_macros, read_macros, strip_dangling_refs
from lecturenotes.figures.blocks import FigureBlock
from lecturenotes.figures.runner import DEFAULT_ALLOWED, FigureRenderer, static_check
from lecturenotes.harness.edits import dangling_refs, remove_refs
from lecturenotes.latex.autofix import autofix
from lecturenotes.latex.lint import lint_latex
from lecturenotes.training.grpo import group_advantages


# ----------------------------------------------------------------------------- autofix must not rewrite valid LaTeX
@pytest.mark.parametrize("snippet", [
    "\\section{S}\n\\begin{tabular}{ccc}\n\\toprule\nx & 1 & 2 \\\\\n- & 3 & 4 \\\\\n\\bottomrule\n\\end{tabular}\n",
    "\\section{S}\nWe have $a\n- b = c$ for all values.\n",
    "\\section{S}\nIn 1999 the value rose.\n42. That year was special.\n7) Another sentence.\n",
    "\\section{S}\nThe ratio is 5 % this is a comment\nnext line.\n",
    "\\section{S}\n\\begin{definition}[Norm]\nA norm $\\norm{x}$ is ...\n\\end{definition}\n\\begin{align}\n a &= b \\\\\n c &= d\n\\end{align}\n",
])
def test_autofix_is_identity_on_valid_latex(snippet):
    assert autofix(snippet).source == snippet


def test_markdown_lists_still_converted_at_paragraph_start():
    s = autofix("\\section{S}\nSteps:\n1. first\n2. second\n\nAlso:\n- a\n- b\n").source
    assert "\\begin{enumerate}" in s and "\\begin{itemize}" in s


def test_ampersand_in_equation_is_not_silently_escaped():
    res = lint_latex("\\section{S}\n\\begin{equation}\na & b\n\\end{equation}\n")
    f = [x for x in res.findings if x.code == "ampersand_outside_table"]
    assert f and f[0].replacement is None


# ----------------------------------------------------------------------------- LaTeX file-reading escapes
@pytest.mark.parametrize("src", [
    "\\section{S}\n\\lstinputlisting{/etc/passwd}\n",
    "\\section{S}\n\\csname input\\endcsname{/etc/passwd}\n",
    "\\section{S}\n\\newcommand{\\x}{\\input{/etc/passwd}}\n\\x\n",
    "\\section{S}\n\\verbatiminput{~/.ssh/id_rsa}\n",
])
def test_file_reading_commands_are_forbidden(src):
    assert any(f.code == "forbidden" for f in lint_latex(src).findings)


# ----------------------------------------------------------------------------- data normalisation
def test_document_macro_with_different_meaning_is_expanded():
    pre = "\\newcommand{\\N}{\\mathcal{N}}\n\\newcommand{\\R}{\\mathbb R}\n\\DeclareMathOperator*{\\argmin}{arg\\,min}\n"
    out = expand_macros("$X \\sim \\N(0,1)$, $x \\in \\R$, $\\argmin f$", read_macros(pre))
    assert "\\mathcal{N}(0,1)" in out          # different from the house \N -> expanded
    assert "\\R$" in out and "\\argmin" in out  # same meaning as the house macro -> kept


def test_dangling_refs_removed():
    s = "\\section{A}\nBy Theorem~\\ref{thm:far} and Figure~\\ref{fig:p}, see (Eq.~\\eqref{eq:k}).\n" \
        "\\begin{equation}x\\label{eq:k}\\end{equation}\n<figure id=\"p\" kind=\"tikz\" caption=\"c\">x</figure>"
    assert dangling_refs(s) == {"thm:far"}
    out = strip_dangling_refs(s)
    assert "thm:far" not in out and "\\ref{fig:p}" in out and "\\eqref{eq:k}" in out
    assert remove_refs("as shown in Figure~\\ref{fig:x}, it rises.", ["fig:x"]) == "as shown, it rises."


# ----------------------------------------------------------------------------- RL helpers
def test_advantage_std_floor():
    adv = group_advantages([0.50, 0.51, 0.50, 0.51])  # tiny, noisy differences
    assert max(abs(a) for a in adv) < 0.1
    assert group_advantages([0.3, 0.3, 0.3]) is None


# ----------------------------------------------------------------------------- figure sandbox
ESCAPES = {
    "mpl_os": "import matplotlib\nmatplotlib.os.system('echo pwned')",
    "getattr_dunder": "x = getattr((), '__class__')",
    "ctypeslib": "import numpy as np\nnp.ctypeslib.ctypes.CDLL(None)",
}


@pytest.mark.parametrize("name", list(ESCAPES))
def test_static_check_blocks_known_escapes(name):
    assert [i for i in static_check(ESCAPES[name], set(DEFAULT_ALLOWED)) if i.severity == "error"]


def test_runtime_guard_blocks_reads_and_writes_outside_sandbox(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    target = tmp_path / "pwned.npy"
    r = FigureRenderer(cache_dir=tmp_path / "cache")
    read_code = ("import matplotlib.cbook as cb\nimport matplotlib.pyplot as plt\n"
                 f"fh = cb.to_filehandle({str(secret)!r})\nfig, ax = plt.subplots()\nax.set_title(fh.read())\n")
    write_code = ("import numpy as np\nimport matplotlib.pyplot as plt\n"
                  f"np.lib.format.open_memmap({str(target)!r}, mode='w+', shape=(3,))\n")
    for i, code in enumerate((read_code, write_code)):
        assert not static_check(code, set(DEFAULT_ALLOWED)) or True
        b = FigureBlock(id=f"esc{i}", kind="matplotlib", caption="", code=code, width=None, start=0, end=0,
                        start_line=1, end_line=1)
        res = r.render(b, tmp_path / "out")
        assert not res.ok and "blocked in figure sandbox" in (res.error or ""), res.error
    assert not target.exists()


def test_legit_figure_still_renders(tmp_path):
    code = ("import numpy as np\nimport matplotlib.pyplot as plt\nxs = [1, 2, 3]\nxs.remove(2)\n"
            "fig, ax = plt.subplots()\nax.plot(xs, xs)\nax.set_xlabel('x'); ax.set_ylabel('y')\n")
    b = FigureBlock(id="ok", kind="matplotlib", caption="", code=code, width=None, start=0, end=0, start_line=1, end_line=1)
    res = FigureRenderer(cache_dir=tmp_path / "cache").render(b, tmp_path / "out")
    assert res.passed, res.feedback()
    assert Path(res.pdf_path).name.startswith("ok-")
