"""Web figures, guaranteed figures and on-policy repair plumbing (offline)."""

import os

from lecturenotes.agent.budget import parse_length
from lecturenotes.agent.planner import postprocess
from lecturenotes.figures.blocks import format_block, parse_blocks, replace_blocks
from lecturenotes.figures.runner import FigureRenderer
from lecturenotes.schemas import Outline, SectionPlan

SRC = """\\section{Cells}
<figure id="mito" kind="web" query="mitochondrion structure" caption="A mitochondrion."/>
Text.
<figure id="p" kind="matplotlib" caption="A plot.">
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
</figure>
"""


def test_self_closing_web_block_does_not_swallow_next_block():
    blocks, issues = parse_blocks(SRC)
    assert [(b.id, b.kind) for b in blocks] == [("mito", "web"), ("p", "matplotlib")]
    assert blocks[0].code == "mitochondrion structure" and not issues
    assert replace_blocks(SRC, blocks, lambda b: "X").count("\n") == SRC.count("\n")


def test_web_block_roundtrip():
    b = parse_blocks(SRC)[0][0]
    again = parse_blocks(format_block(b))[0][0]
    assert (again.id, again.kind, again.code, again.caption) == ("mito", "web", "mitochondrion structure", "A mitochondrion.")


def test_web_offline_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("LN_OFFLINE", "1")
    b = parse_blocks(SRC)[0][0]
    b.code = "some query never cached " + str(os.getpid())
    res = FigureRenderer(cache_dir=tmp_path / "c").render(b, tmp_path / "o")
    assert not res.ok and "no freely licensed image" in res.error


def test_planner_gives_every_section_a_figure():
    o = Outline(title="T", sections=[SectionPlan(title="Overview of the architecture", key_points=["a"]),
                                     SectionPlan(title="Error analysis", key_points=["b"])])
    o = postprocess(o, parse_length("4 pages"), None, None)
    assert all(s.figures for s in o.sections)
    assert o.sections[0].figures[0].kind == "tikz" and o.sections[1].figures[0].kind == "matplotlib"


def test_web_query_prompt_clean_and_parse():
    from lecturenotes.prompts.templates import clean_query, web_query_messages
    from lecturenotes.training.rl_common import parse_web_query_prompt

    msgs = web_query_messages("Cells", "Organelles", "a mitochondrion under the microscope", failed=["mito"])
    assert parse_web_query_prompt(msgs[-1]["content"])["description"] == "a mitochondrion under the microscope"
    assert clean_query('Query: "mitochondrion electron micrograph".\nextra') == "mitochondrion electron micrograph"


def test_teacher_requests_web_figures_for_subject_fields():
    import random

    from lecturenotes.data.synth import teacher

    assert "chemistry & biology" in teacher.WEB_FIELDS and 'kind=\\"web\\"' in teacher.WEB_INSTRUCTION.replace('"', '\\"')


def test_parse_pick_and_reward():
    from lecturenotes.prompts.templates import parse_pick
    from lecturenotes.training.rl_common import compute_reward

    assert parse_pick("choice: 2\nreject: 1,3\nreason: x", 4) == (2, True)
    assert parse_pick("Choice: none\nreason: nothing fits", 4) == (None, True)
    assert parse_pick("choice: 9", 4) == (None, False) and parse_pick("I like the second", 4) == (None, False)
    meta = {"n_candidates": 4, "gold_output": "choice: 2\nreject: -\nreason: y"}
    assert compute_reward("pick_image", meta, "choice: 2")[0] == 1.0
    assert compute_reward("pick_image", meta, "choice: 3")[0] == -0.3
    assert compute_reward("pick_image", meta, "choice: none")[0] == -0.5
    assert compute_reward("pick_image", meta, "whatever")[0] == -1.0


def test_agent_picks_and_pins_web_image(tmp_path, monkeypatch):
    """web figure in a draft -> candidates -> pick_image -> block pinned to the chosen Commons file."""
    import asyncio

    from PIL import Image

    from lecturenotes.agent.section import SectionWorker
    from lecturenotes.config import HarnessConfig
    from lecturenotes.figures import web
    from lecturenotes.harness.verifier import SectionContext, Verifier
    from lecturenotes.llm.client import LLMClient
    from lecturenotes.llm.mock import MockBackend
    from lecturenotes.schemas import LectureRequest

    png = tmp_path / "img.png"
    Image.new("RGB", (400, 300), "white").save(png)
    from PIL import ImageDraw
    im = Image.open(png); ImageDraw.Draw(im).ellipse((50, 50, 350, 250), fill="blue"); im.save(png)
    cands = [{"title": f"File:Cand{i}.png", "url": "x", "mime": "image/png", "width": 400, "height": 300,
              "license": "CC0", "author": "A", "page": "p", "description": f"candidate {i}", "categories": []}
             for i in (1, 2)]
    monkeypatch.setattr(web, "candidates", lambda q, k=6, cache_dir=None: cands)
    monkeypatch.setattr(web, "fetch_file", lambda title, out_dir, cache_dir: (
        out_dir.mkdir(parents=True, exist_ok=True), (out_dir / "c.png").write_bytes(png.read_bytes()),
        web.WebImage(out_dir / "c.png", title, "A", "CC0", "p"))[2])
    req = LectureRequest(topic="Cells")
    from lecturenotes.schemas import Outline, SectionPlan
    sec = SectionPlan(title="Organelles", key_points=["mitochondrion"], target_words=50)
    outline = Outline(title="Cells", sections=[sec])
    v = Verifier(renderer=None) if False else Verifier()
    client = LLMClient({"m": MockBackend()}, {"*": "m"})
    w = SectionWorker(client=client, verifier=v, cfg=HarnessConfig(), req=req, outline=outline, deck=None,
                      caps=None, figures_dir=tmp_path / "figs")
    src = ("\\section{Organelles}\nThe mitochondrion produces energy for the cell.\n"
           '<figure id="mito" kind="web" query="mitochondrion" caption="A mitochondrion."/>\n')
    rep = v.verify_section(src, SectionContext(sid=sec.id, title=sec.title), render_figures=False, figures_dir=tmp_path / "figs")
    out = asyncio.run(w._select_web_images(sec, rep, []))
    assert 'file="File:Cand1.png"' in out.source and out.compile_ok
    picks = [r for r in client.tracer.records.values() if r["task"] == "pick_image"]
    assert picks and picks[0]["outcome"].get("accepted") is True


def test_pinned_web_image_does_not_reuse_unpinned_render(tmp_path, monkeypatch):
    """Regression: the draft's unpinned render (first search hit) was reused after the model pinned a
    different file, because prior results were keyed by the query only (a tank in the Hamming notes)."""
    import asyncio

    from PIL import Image

    from lecturenotes.agent.section import SectionWorker
    from lecturenotes.config import HarnessConfig
    from lecturenotes.figures import web
    from lecturenotes.harness.verifier import SectionContext, Verifier
    from lecturenotes.llm.client import LLMClient
    from lecturenotes.llm.mock import MockBackend
    from lecturenotes.schemas import LectureRequest

    def img(name, color):
        p = tmp_path / f"{name}.png"
        im = Image.new("RGB", (400, 300), "white")
        from PIL import ImageDraw
        ImageDraw.Draw(im).rectangle((40, 40, 360, 260), fill=color)
        im.save(p)
        return p

    tank, good = img("tank", "green"), img("good", "blue")

    def fake(src, author):
        def f(*a, **k):
            out_dir = k.get("out_dir") or a[1]
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"{author}.png"
            dst.write_bytes(src.read_bytes())
            return web.WebImage(dst, f"File:{author}.png", author, "CC0", "p")
        return f

    cands = [{"title": "File:Good.png", "url": "x", "mime": "image/png", "width": 400, "height": 300,
              "license": "CC0", "author": "Good", "page": "p", "description": "the right one", "categories": []}]
    monkeypatch.setattr(web, "candidates", lambda q, k=6, cache_dir=None: cands)
    monkeypatch.setattr(web, "fetch_image", fake(tank, "Tank"))
    monkeypatch.setattr(web, "fetch_file", fake(good, "Good"))
    from lecturenotes.schemas import Outline, SectionPlan
    sec = SectionPlan(title="Codes", key_points=["parity"], target_words=50)
    v = Verifier()
    w = SectionWorker(client=LLMClient({"m": MockBackend()}, {"*": "m"}), verifier=v, cfg=HarnessConfig(),
                      req=LectureRequest(topic="Codes"), outline=Outline(title="Codes", sections=[sec]), deck=None,
                      caps=None, figures_dir=tmp_path / "figs")
    src = ("\\section{Codes}\nParity bits protect data against single errors in transmission.\n"
           '<figure id="ecc" kind="web" query="error correcting code" caption="Error correction."/>\n')
    rep = v.verify_section(src, SectionContext(sid=sec.id, title=sec.title), figures_dir=tmp_path / "figs")
    assert "Tank" in (rep.figure_results["ecc"].credit or "")
    out = asyncio.run(w._select_web_images(sec, rep, []))
    assert 'file="File:Good.png"' in out.source
    assert "Good" in (out.figure_results["ecc"].credit or "") and "Tank" not in (out.figure_results["ecc"].credit or "")
