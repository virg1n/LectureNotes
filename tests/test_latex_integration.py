"""Integration tests that run pdflatex / the figure sandbox (marked `latex`)."""

import asyncio
import re
import shutil

import pytest

from lecturenotes.harness.rewards import section_reward
from lecturenotes.harness.verifier import SectionContext, Verifier
from lecturenotes.latex.capabilities import detect_capabilities

pytestmark = pytest.mark.latex
if shutil.which("pdflatex") is None:
    pytest.skip("pdflatex not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def verifier(tmp_path_factory):
    caps = detect_capabilities()
    return Verifier(caps=caps)


GOOD = r"""\section{Gradient descent}
We minimise a differentiable function $f\colon \R^n \to \R$ by repeatedly stepping against the gradient.
\begin{definition}[Gradient step]\label{def:step}
Given a step size $\eta > 0$, the update is $x_{k+1} = x_k - \eta \nabla f(x_k)$.
\end{definition}
<figure id="gd-path" kind="matplotlib" caption="Iterates of gradient descent on a quadratic.">
import numpy as np
import matplotlib.pyplot as plt
x = [2.0]
for _ in range(12):
    x.append(x[-1] - 0.3 * 2 * x[-1])
xs = np.linspace(-2.2, 2.2, 200)
fig, ax = plt.subplots()
ax.plot(xs, xs**2, label="$f(x)=x^2$")
ax.plot(x, np.array(x)**2, "o-", label="iterates")
ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$"); ax.legend()
</figure>
As Figure~\ref{fig:gd-path} shows, the iterates approach the minimiser.
\begin{keyidea}
Small steps follow the steepest descent direction.
\end{keyidea}
"""


def test_good_section_compiles_with_figure(verifier, tmp_path):
    rep = verifier.verify_section(GOOD, SectionContext(sid="s1", title="Gradient descent", target_words=60,
                                                       key_points=["step size", "gradient"]),
                                  figures_dir=tmp_path)
    assert rep.compile_ok, rep.latex_feedback()
    assert rep.figures_ok == 1 and not rep.failed_figures
    assert list((tmp_path / "s1").glob("gd-path-*.pdf"))  # content-addressed file name
    r, info = section_reward(rep)
    assert r > 0.5, info


def test_reward_ordering(verifier):
    """clean > local error > thin answers; deleting broken math is not a better 'repair' than fixing it."""
    from lecturenotes.harness.rewards import repair_reward, score_section

    ctx = SectionContext(sid="s1", title="Gradient descent", target_words=60, key_points=["step size", "gradient"])

    def score(src):
        return score_section(verifier, verifier.verify_section(src, ctx), ctx, expected_figures=1)[0]

    clean = score(GOOD)
    broken = score(GOOD.replace("Small steps", "Small \\qedx steps"))
    empty = score("\\section{Gradient descent}\n")
    bullets = score("\\section{Gradient descent}\n\\begin{itemize}\\item step size \\item gradient\\end{itemize}\n")
    assert clean > broken > max(empty, bullets), (clean, broken, empty, bullets)

    nctx = SectionContext(sid="s1")
    formula = "$x_{k+1} = x_k - \\eta \\nabla f(x_k)$"
    assert formula in GOOD
    bad_src = GOOD.replace(formula, formula.replace("\\eta", "\\etta"))  # undefined macro inside the formula
    bad = verifier.verify_section(bad_src, nctx, render_figures=False)
    assert not bad.compile_ok
    fixed = verifier.verify_section(GOOD, nctx, render_figures=False)
    deleted = verifier.verify_section(GOOD.replace(formula, ""), nctx, render_figures=False)  # "repair" by deletion
    assert deleted.compile_ok
    r_fix = repair_reward(bad, fixed, bad.source, fixed.source, gold_src=GOOD)
    r_del = repair_reward(bad, deleted, bad.source, deleted.source, gold_src=GOOD)
    assert r_fix > r_del, (r_fix, r_del)


def test_concurrent_drafts_do_not_share_figure_files(verifier, tmp_path):
    """Best-of-n drafts reuse figure ids; each must reference its own rendered file."""
    from concurrent.futures import ThreadPoolExecutor

    other = GOOD.replace('label="iterates")', 'label="iterates", color="red")')
    with ThreadPoolExecutor(2) as ex:
        a, b = ex.map(lambda s: verifier.verify_section(s, SectionContext(sid="s1"), figures_dir=tmp_path), [GOOD, other])
    fa = re.search(r"\\lnfigure\{[^}]*\}\{([^}]+)\}", a.processed).group(1)
    fb = re.search(r"\\lnfigure\{[^}]*\}\{([^}]+)\}", b.processed).group(1)
    assert fa != fb and (tmp_path / fa).exists() and (tmp_path / fb).exists()


def test_undefined_macro_reported_with_line(verifier):
    bad = GOOD.replace("Small steps", "Small \\qedx steps")
    rep = verifier.verify_section(bad, SectionContext(sid="s1"), render_figures=False)
    assert not rep.compile_ok
    err = rep.compile_issues[0]
    assert err.code == "undefined_cs" and "\\qedx" in err.message
    assert bad.splitlines()[err.line - 1].count("\\qedx") == 1  # line maps back to the model's text


def test_failing_figure_detected(verifier):
    bad = GOOD.replace("np.array(x)**2", "np.array(x[:3])**2")
    rep = verifier.verify_section(bad, SectionContext(sid="s1"))
    assert rep.compile_ok  # placeholder keeps the document valid
    assert [b.id for b in rep.failed_figures] == ["gd-path"]
    assert "ValueError" in rep.figure_results["gd-path"].error


def test_figure_inside_box_compiles(verifier):
    src = GOOD.replace("\\begin{keyidea}\nSmall steps", "\\begin{example}\n<figure id=\"inbox\" kind=\"tikz\" "
                       "caption=\"Inside a box.\">\n\\begin{tikzpicture}\\draw (0,0) circle (1);\\end{tikzpicture}\n"
                       "</figure>\nSee Figure~\\ref{fig:inbox}.\n\\end{example}\n\\begin{keyidea}\nSmall steps")
    rep = verifier.verify_section(src, SectionContext(sid="s1"))
    assert rep.compile_ok, rep.latex_feedback()
    assert "\\lnfigurehere" in rep.processed


def test_mock_agent_end_to_end(tmp_path):
    from lecturenotes.agent.orchestrator import LectureAgent
    from lecturenotes.config import AgentConfig
    from lecturenotes.schemas import LectureRequest

    cfg = AgentConfig.model_validate({"backends": {"m": {"type": "mock"}}, "routes": {"*": "m"}})
    res = asyncio.run(LectureAgent(cfg).run(LectureRequest(topic="Metric spaces", expected_length="3 pages"), tmp_path))
    assert res.notes_pdf is not None and res.notes_pdf.exists()
    assert res.cheatsheet_pdf is not None
    rep = res.report
    assert rep["document"]["ok"] and rep["totals"]["figures_ok"] == rep["totals"]["figures"]
    steps = [s for sec in rep["sections"] for s in sec["steps"]]
    assert "repair_latex" in steps and "repair_figure" in steps  # the mock's injected errors were repaired
    assert (tmp_path / "trace.jsonl").exists()
