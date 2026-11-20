"""Fast unit tests (no TeX / no GPU needed)."""

import random

import pytest

from lecturenotes.agent.budget import parse_length
from lecturenotes.agent.planner import _normalize_plan, fallback_outline
from lecturenotes.data.dedup import MinHashDeduper
from lecturenotes.data.synth.corrupt import corrupt_latex, make_edits
from lecturenotes.figures.blocks import format_block, parse_blocks, remove_figure_refs, replace_blocks
from lecturenotes.harness.edits import apply_edit, apply_model_repair, parse_edits, quarantine, top_level_units
from lecturenotes.ingest.selection import parse_selection
from lecturenotes.latex.autofix import autofix
from lecturenotes.latex.lint import lint_latex
from lecturenotes.latex.logparse import parse_log
from lecturenotes.latex.textstats import coverage, repetition_rate, word_count
from lecturenotes.schemas import LectureRequest, Outline
from lecturenotes.training.rl_common import parse_repair_prompt, parse_section_prompt
from lecturenotes.utils import extract_json_object


# ----------------------------------------------------------------------------- selection / budget
@pytest.mark.parametrize("spec,total,expected", [
    ("1-3, 5", None, [1, 2, 3, 5]), ("8-", 10, [8, 9, 10]), ("-2", 10, [1, 2]), ("all", 3, [1, 2, 3]),
    ("3..5", None, [3, 4, 5]), ("5-3", None, [3, 4, 5]), ("2 4 6", None, [2, 4, 6]),
])
def test_parse_selection(spec, total, expected):
    assert parse_selection(spec, total) == expected


def test_parse_length():
    assert parse_length("6 pages").pages == 6
    assert parse_length("2500 words").words == 2500
    assert parse_length("short").pages == 4
    assert parse_length("подробный конспект").pages == 14
    assert parse_length("10 стр").pages == 10
    b = parse_length(None, n_slides=30)
    assert b.source == "slides" and b.pages == 15
    assert parse_length("4-6 pages").pages == 5


# ----------------------------------------------------------------------------- JSON / plan
def test_extract_json_with_latex_escapes_and_fences():
    text = 'Sure!\n```json\n{"title": "Norms \\alpha", "sections": [{"title": "A",},],}\n```'
    data = extract_json_object(text)
    assert data["title"].startswith("Norms") and data["sections"][0]["title"] == "A"


def test_normalize_plan_variants():
    data = _normalize_plan({"lecture_title": "X", "chapters": ["Intro", {"name": "Core", "keypoints": "a\nb"}]})
    o = Outline.model_validate(data)
    assert [s.id for s in o.sections] == ["s1", "s2"]
    assert o.sections[1].key_points == ["a", "b"]


def test_fallback_outline_with_budget():
    o = fallback_outline(LectureRequest(topic="Graphs"), parse_length("6 pages"), None)
    assert 2 <= len(o.sections) <= 8


# ----------------------------------------------------------------------------- figure blocks
SRC = r"""\section{A}
Text before.
<figure id="p1" kind="matplotlib" caption="A plot.">
```python
import numpy as np
x = 1
```
</figure>
See Figure~\ref{fig:p1}.
<slide n="3" caption="Original slide"/>
<figure kind="tikz" caption="d">
\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}
</figure>
End.
"""


def test_parse_blocks_and_line_preservation():
    blocks, issues = parse_blocks(SRC)
    assert [b.kind for b in blocks] == ["matplotlib", "slide", "tikz"]
    assert blocks[0].code.startswith("import numpy")  # fences stripped
    assert blocks[1].slide_no == 3
    out = replace_blocks(SRC, blocks, lambda b: f"\\X{{{b.id}}}")
    assert out.count("\n") == SRC.count("\n")  # line numbers preserved
    assert out.splitlines()[2] == "\\X{p1}"


def test_format_block_roundtrip_and_ref_removal():
    blocks, _ = parse_blocks(SRC)
    again, _ = parse_blocks(format_block(blocks[0]))
    assert again[0].id == "p1" and again[0].code == blocks[0].code
    assert "\\ref{fig:p1}" not in remove_figure_refs(SRC, ["p1"])


# ----------------------------------------------------------------------------- lint
def test_lint_text_in_math_and_aliases():
    src = "\\section{S}\n\\begin{align}\n a &= b \\text{ if $x>0$ }\n\\end{align}\n\\begin{thm}x\\end{thm}\n"
    res = lint_latex(src)
    codes = [f.code for f in res.findings]
    assert "dollar_in_math_env" not in codes  # $ inside \text{} is legal
    assert codes.count("env_alias") == 2


def test_lint_structural_errors():
    src = "\\section{S}\n\\begin{definition}\nunclosed {brace\n\n$x\n\nok\n\\end{example}\n"
    codes = {f.code for f in lint_latex(src).findings}
    assert {"unclosed_brace", "unclosed_inline_math", "unmatched_end", "unclosed_env"} <= codes


def test_lint_newcommand_hash_and_verbatim():
    src = "\\section{S}\n\\newcommand{\\foo}[1]{#1}\n\\begin{lstlisting}\nx_1 = a & b # c\n\\end{lstlisting}\n"
    assert not [f for f in lint_latex(src).findings if f.code in ("hash", "script_outside_math", "ampersand_outside_table")]


def test_lint_forbidden_preamble():
    src = "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n\\section{S} x\n\\end{document}\n"
    assert sum(f.code == "forbidden" for f in lint_latex(src).findings) == 4


# ----------------------------------------------------------------------------- autofix
def test_autofix_markdown_unicode_specials():
    src = ("## Title\nThe **rate** α is 50% of x_i & more.\n- a\n- b\n\\begin{thm}\nθ ≤ 1\n\\end{thm}\n"
           "\\begin{definition}\nopen")
    r = autofix(src, section_title="Title")
    s = r.source
    assert s.startswith("\\section{Title}")
    assert "\\textbf{rate}" in s and "$\\alpha$" in s and "50\\%" in s and "$x_i$" in s and "\\&" in s
    assert "\\begin{itemize}" in s and "\\begin{theorem}" in s and "\\leq" in s
    assert s.rstrip().endswith("\\end{definition}")
    assert not r.lint.errors


def test_unclosed_box_closed_before_next_box():
    src = "\\section{S}\n\\begin{definition}\nA def.\n\\begin{example}\nAn ex.\n\\end{example}\nTail.\n"
    s = autofix(src).source
    assert s.index("\\end{definition}") < s.index("\\begin{example}")


def test_autofix_snake_case_is_escaped_not_mathified():
    r = autofix("\\section{S}\nset learning_rate to 3.\n")
    assert "learning\\_rate" in r.source


# ----------------------------------------------------------------------------- edits / quarantine
def test_search_replace_exact_and_fuzzy():
    src = "a\n  b = 1 \nc\n"
    out = apply_model_repair(src, "<<<<<<< SEARCH\nb = 1\n=======\nb = 2\n>>>>>>> REPLACE", expect_prefix=None)
    assert out.applied == 1 and "b = 2" in out.source
    e = parse_edits("<<<<<<< SEARCH\nthe quick brown fox jumps\n=======\nX\n>>>>>>> REPLACE")[0]
    assert apply_edit("intro\nthe quick brown fox jumped\nend", e) == "intro\nX\nend"


def test_edits_are_located_in_the_original_source():
    # the second SEARCH also matches text that the first REPLACE inserts; atomic application must not care
    src = "A\n\\end{x}\nB\nC\n"
    out = ("<<<<<<< SEARCH\nA\n=======\nA\n\\end{x}\n>>>>>>> REPLACE\n"
           "<<<<<<< SEARCH\nB\n\\end{x}\n=======\nB\n>>>>>>> REPLACE")
    r = apply_model_repair("A\nB\n\\end{x}\nC\n", out, expect_prefix=None)
    assert r.source == src.replace("\\end{x}\nB", "\\end{x}\nB") and r.applied == 2


def test_inside_box_detection():
    from lecturenotes.harness.verifier import inside_box

    s = "\\begin{example}\nX <figure> \\end{example}\n<figure>"
    assert inside_box(s, s.index("<figure>")) and not inside_box(s, s.rindex("<figure>"))


def test_full_rewrite_guard():
    src = "\\section{A}\n" + "long text\n" * 50
    assert apply_model_repair(src, "\\section{A}\nshort").mode == "none"  # truncated rewrite rejected


def test_quarantine_removes_failing_unit_but_keeps_heading():
    src = "\\section{A}\nGood para.\n\nBad \\foo para.\n\n\\begin{example}\nx\n\\end{example}\n"
    units = top_level_units(src)
    assert units == [(0, 0), (1, 1), (3, 3), (5, 7)]  # heading, paragraph, paragraph, environment
    out, removed = quarantine(src, [4])
    assert "\\foo" not in out and "\\section{A}" in out and "\\begin{example}" in out and len(removed) == 1


# ----------------------------------------------------------------------------- log parsing
LOG = r"""(./main.tex (./sections/s2.tex
./sections/s2.tex:12: Undefined control sequence.
l.12 This uses \foo
                    bar.
./sections/s2.tex:20: LaTeX Error: Environment defintion undefined.

See the LaTeX manual or LaTeX Companion for explanation.
Type  H <return>  for immediate help.
 ...

l.20 \begin{defintion}

Overfull \hbox (23.5pt too wide) in paragraph at lines 30--31
LaTeX Warning: Reference `fig:x' on page 1 undefined on input line 33.
LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.
))
Output written on main.pdf (3 pages, 12345 bytes).
"""


def test_parse_log():
    s = parse_log(LOG)
    assert len(s.errors) == 2
    e0, e1 = s.errors
    assert (e0.file, e0.line, e0.category, e0.culprit) == ("sections/s2.tex", 12, "undefined_cs", "\\foo")
    assert e1.category == "env_mismatch" and e1.line == 20
    assert s.pages == 3 and s.needs_rerun and s.undefined_refs == ["fig:x"]
    assert s.overfull(5)[0].amount_pt == 23.5


# ----------------------------------------------------------------------------- stats / rewards helpers
def test_word_count_and_coverage():
    src = "\\section{Eigen}\nAn \\term{eigenvector} satisfies $Av = \\lambda v$.\n\\[ x^2 \\]\n% comment words\n"
    assert word_count(src) == 5  # Eigen, An, eigenvector, satisfies + inline math counted as one token "X"
    cov, missing = coverage(["eigenvectors", "spectral theorem"], src)
    assert cov == 0.5 and missing == ["spectral theorem"]
    assert repetition_rate("\\section{A} " + "same words again and again " * 20) > 0.5


# ----------------------------------------------------------------------------- data synthesis
def test_corruption_inverse_edit_roundtrip():
    clean = ("\\section{S}\nLet $\\frac{a}{b}$ and \\textbf{bold}.\n\\begin{definition}\nx\n\\end{definition}\n"
             "\\begin{align}\n f &= \\left( x \\right)\n\\end{align}\n")
    rng = random.Random(3)
    for _ in range(20):
        c = corrupt_latex(clean, rng)
        if c is None:
            continue
        edits = make_edits(c.source, clean)
        assert apply_model_repair(c.source, edits, expect_prefix=None).source == clean


def test_minhash_dedup():
    d = MinHashDeduper()
    base = "the quick brown fox jumps over the lazy dog " * 20
    assert d.add("a", base)
    assert not d.add("b", base + " extra")
    assert d.add("c", "completely different text about eigenvalues and matrices " * 10)


def test_prompt_parsers_roundtrip():
    from lecturenotes.prompts.templates import repair_latex_messages, section_messages
    from lecturenotes.schemas import SectionPlan

    o = Outline(title="T", sections=[SectionPlan(title="Intro", key_points=["a", "b"], target_words=420)])
    msgs = section_messages(LectureRequest(topic="T"), o, o.sections[0])
    meta = parse_section_prompt(msgs[-1]["content"])
    assert meta["target_words"] == 420 and meta["key_points"] == ["a", "b"] and meta["title"] == "Intro"
    src = "\\section{A}\nline two"
    assert parse_repair_prompt(repair_latex_messages(src, "- err")[-1]["content"])["source"] == src
