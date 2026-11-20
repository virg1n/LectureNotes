"""Checks added after reading the first SFT model's notes: pseudo-environment tags, filler phrases,
overlapping TikZ labels."""

import pytest

from lecturenotes.latex.autofix import autofix
from lecturenotes.latex.textstats import count_cliches, tex_stats

BS = chr(92)


def test_pseudo_env_tag_becomes_box():
    src = (f"{BS}section{{A}}\nIntro.\n\n[keyidea] Redundancy is added.\nIt helps.\n\n"
           f"[definition: Code] A code is X.\n\nSee [1] and [Note] inline.\n")
    r = autofix(src)
    assert f"{BS}begin{{keyidea}}\nRedundancy is added.\nIt helps.\n{BS}end{{keyidea}}" in r.source
    assert f"{BS}begin{{definition}}[Code]" in r.source and "[1]" in r.source and "[Note] inline" in r.source
    assert [f.code for f in r.fixes].count("md_pseudo_env") == 2


def test_cliches_counted_and_penalised():
    assert count_cliches("The parity bit is the XOR of the covered bits.") == 0
    bad = ("In summary, the Kalman filter is a powerful tool. It plays a crucial role in robotics. "
           "In summary, it is a powerful tool.")
    assert count_cliches(bad) == 5
    assert tex_stats(bad).cliches == 5


@pytest.mark.latex
def test_overlapping_tikz_labels_fail(tmp_path):
    from lecturenotes.figures.blocks import FigureBlock
    from lecturenotes.figures.runner import FigureRenderer

    r = FigureRenderer(cache_dir=tmp_path / "c")
    good = (f"{BS}node at (0,0) {{Kalman Filter}};\n{BS}node at (0,-1.5) {{Measurement Noise}};")
    bad = (f"{BS}node at (0,0) {{Kalman Filter}};\n{BS}node at (0.6,0) {{Measurement Noise}};")
    res_good = r.render(FigureBlock(id="g", kind="tikz", caption="c", code=good, width=None, start=0, end=0, start_line=1,
                                    end_line=1), tmp_path / "o")
    res_bad = r.render(FigureBlock(id="b", kind="tikz", caption="c", code=bad, width=None, start=0, end=0, start_line=1,
                                   end_line=1), tmp_path / "o")
    assert res_good.passed, res_good.feedback()
    assert not res_bad.passed and any(i.code == "text_overlap" for i in res_bad.issues)


def test_vlm_judge_marks_irrelevant_figure(tmp_path, monkeypatch):
    """A figure the VLM scores <= 2 becomes an error (repair / drop), score 3 a warning."""
    from lecturenotes.figures.relevance import FigureJudge, figures_in_notes
    from lecturenotes.harness import verifier as ver
    from lecturenotes.figures.runner import FigureResult

    png = tmp_path / "f.png"
    from PIL import Image
    Image.new("RGB", (50, 50), "white").save(png)
    calls = []
    j = FigureJudge("http://x/v1", cache_dir=tmp_path / "j")
    monkeypatch.setattr(j, "judge", lambda img, cap, sec="", topic="": calls.append(cap) or
                        {"score": 1 if "decoding" in cap else 5, "reason": "sine wave"})
    r1 = FigureResult(id="a", kind="matplotlib", ok=True, png_path=png)
    ver._apply_relevance(r1, j.judge(png, "Hamming decoding"))
    r2 = FigureResult(id="b", kind="matplotlib", ok=True, png_path=png)
    ver._apply_relevance(r2, j.judge(png, "Parity bits"))
    assert not r1.passed and r1.relevance == 0.2 and r2.passed and r2.relevance == 1.0
    tex = (BS + "section{Codes}\n" + BS + "lnfigure{width=0.6" + BS + "linewidth}{s1/a.pdf}{Parity bits. {"
           + BS + "footnotesize(Image: X, CC0, Wikimedia Commons)}}{fig:a}\n")
    assert figures_in_notes(tex) == [("s1/a.pdf", "Parity bits.", "Codes")]
