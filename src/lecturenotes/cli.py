"""Command line interface.

    lecturenotes generate --topic "Fourier series" [--slides deck.pdf --pages 1-12] [--length "6 pages"]
    lecturenotes verify section.tex                 # run the harness on a LaTeX fragment
    lecturenotes render-figure fig.py --kind matplotlib
    lecturenotes doctor                             # toolchain / capability report
    lecturenotes data ...  / train ... / eval ...   # see the respective modules
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def cmd_generate(a: argparse.Namespace) -> int:
    from .agent.orchestrator import LectureAgent
    from .config import AgentConfig, BackendConfig
    from .schemas import LectureRequest

    cfg = AgentConfig.load(a.config)
    if a.backend:
        cfg.backends = {"cli": BackendConfig(type=a.backend, model=a.model or BackendConfig().model,
                                             base_url=a.base_url or BackendConfig().base_url)}
        cfg.routes = {"*": "cli"}
    if a.samples:
        cfg.harness.samples_per_section = a.samples
    req = LectureRequest(topic=a.topic, presentation=Path(a.slides) if a.slides else None, slides=a.pages,
                         expected_length=a.length, language=a.lang, make_cheatsheet=not a.no_cheatsheet,
                         cheatsheet_pages=a.cheatsheet_pages, instructions=a.instructions, course=a.course,
                         audience=a.audience)
    res = asyncio.run(LectureAgent(cfg).run(req, a.out))
    t = res.report["totals"]
    print(f"notes:      {res.notes_pdf}  ({res.report['document']['pages']} pages, {t['words']} words)")
    print(f"source:     {res.notes_tex}")
    if res.cheatsheet_pdf:
        print(f"cheatsheet: {res.cheatsheet_pdf}")
    print(f"report:     {Path(a.out) / 'report.md'}")
    return 0 if res.notes_pdf else 1


def cmd_verify(a: argparse.Namespace) -> int:
    from .harness.rewards import score_section
    from .harness.verifier import SectionContext, Verifier
    from .latex.capabilities import detect_capabilities

    src = Path(a.file).read_text(encoding="utf-8")
    caps = detect_capabilities()
    v = Verifier(caps=caps, language=a.lang, keep_dirs=False)
    ctx = SectionContext(sid="s1", title=a.title, target_words=a.target_words, language=a.lang,
                         key_points=a.key_points.split(";") if a.key_points else [],
                         mode="cheatsheet" if a.cheatsheet else "section")
    rep = v.verify_section(src, ctx, figures_dir=Path(a.figures_dir) if a.figures_dir else None)
    reward, info = score_section(v, rep, ctx)
    print(json.dumps({"summary": rep.summary(), "reward": reward, "reward_info": info}, indent=2, default=str))
    if rep.latex_errors:
        print("\nLaTeX errors:\n" + rep.latex_feedback())
    for b in rep.figure_blocks:
        r = rep.figure_results.get(b.id)
        if r is not None and not r.passed:
            print(f"\nfigure {b.id}:\n{r.feedback()}")
    if rep.fixes:
        print("\nautofixes: " + ", ".join(sorted({f.code for f in rep.fixes})))
    if a.write_fixed:
        Path(a.write_fixed).write_text(rep.source, encoding="utf-8")
    return 0 if rep.ok else 2


def cmd_render_figure(a: argparse.Namespace) -> int:
    from .figures.blocks import FigureBlock
    from .figures.runner import FigureRenderer
    from .latex.capabilities import detect_capabilities

    code = Path(a.file).read_text(encoding="utf-8")
    b = FigureBlock(id=Path(a.file).stem, kind=a.kind, caption="", code=code, width=None, start=0, end=0,
                    start_line=1, end_line=1)
    res = FigureRenderer(caps=detect_capabilities()).render(b, Path(a.out))
    print(f"ok={res.ok} passed={res.passed} file={res.pdf_path}")
    if res.feedback():
        print(res.feedback())
    return 0 if res.passed else 2


def cmd_doctor(a: argparse.Namespace) -> int:
    import importlib.util

    from .latex.capabilities import detect_capabilities
    from .latex.compiler import default_compiler

    comp = default_compiler()
    caps = detect_capabilities(comp, refresh=a.refresh)
    print(f"LaTeX engine: {comp.engine} at {comp.exe} ({caps.distribution})")
    for k, v in sorted(caps.features.items()):
        print(f"  {'OK ' if v else '-- '} {k}")
    missing = [k for k, v in caps.features.items() if not v]
    if missing:
        print(f"Missing features: {', '.join(missing)} (see scripts/setup_tex.sh)")
    print("Python packages:")
    for m in ("numpy", "matplotlib", "pymupdf", "pptx", "httpx", "torch", "transformers", "peft", "vllm",
              "anthropic", "datasets"):
        print(f"  {'OK ' if importlib.util.find_spec(m) else '-- '} {m}")
    if a.base_url:
        import httpx

        try:
            r = httpx.get(a.base_url.rstrip("/") + "/models", timeout=10)
            print(f"LLM endpoint {a.base_url}: {r.status_code} {[m['id'] for m in r.json().get('data', [])]}")
        except Exception as e:
            print(f"LLM endpoint {a.base_url}: unreachable ({e})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("lecturenotes", description="Harness-guided small-LM lecture notes agent")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate lecture notes (+ cheat sheet)")
    g.add_argument("--topic", required=True)
    g.add_argument("--slides", help="presentation file (pdf, pptx, beamer .tex, md)")
    g.add_argument("--pages", help='slide selection, e.g. "1-12,15"')
    g.add_argument("--length", help='expected length, e.g. "6 pages", "2500 words", "short"')
    g.add_argument("--lang", default="en")
    g.add_argument("--audience", default="university students")
    g.add_argument("--course")
    g.add_argument("--instructions")
    g.add_argument("--no-cheatsheet", action="store_true")
    g.add_argument("--cheatsheet-pages", type=int, default=2)
    g.add_argument("--config", help="agent YAML config (configs/agent/*.yaml)")
    g.add_argument("--backend", choices=["openai", "hf", "anthropic", "mock"])
    g.add_argument("--model")
    g.add_argument("--base-url")
    g.add_argument("--samples", type=int, help="best-of-n drafts per section")
    g.add_argument("--out", default="out")
    g.set_defaults(fn=cmd_generate)

    v = sub.add_parser("verify", help="run the harness on a LaTeX fragment")
    v.add_argument("file")
    v.add_argument("--title")
    v.add_argument("--target-words", type=int)
    v.add_argument("--key-points", help="semicolon-separated")
    v.add_argument("--lang", default="en")
    v.add_argument("--cheatsheet", action="store_true")
    v.add_argument("--figures-dir")
    v.add_argument("--write-fixed", help="write the autofixed source here")
    v.set_defaults(fn=cmd_verify)

    f = sub.add_parser("render-figure", help="render one figure file in the sandbox")
    f.add_argument("file")
    f.add_argument("--kind", choices=["matplotlib", "tikz"], default="matplotlib")
    f.add_argument("--out", default="figures_out")
    f.set_defaults(fn=cmd_render_figure)

    d = sub.add_parser("doctor", help="check LaTeX/Python toolchain and an optional LLM endpoint")
    d.add_argument("--refresh", action="store_true")
    d.add_argument("--base-url")
    d.set_defaults(fn=cmd_doctor)

    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
