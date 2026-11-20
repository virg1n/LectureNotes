"""Pairwise LLM-as-judge on *rendered pages* (vision): which lecture notes are better?

    python -m lecturenotes.eval.judge --a eval_runs/base --b eval_runs/sft-v1 --pages 4

The judge sees PNG renders of the first pages of both PDFs plus the LaTeX source
excerpts, scores a rubric, and states a preference. Each pair is judged twice with
swapped positions; inconsistent verdicts count as ties.
Requires ANTHROPIC_API_KEY (or an `ant auth login` profile) and `pip install anthropic`.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from ..utils import extract_json_object

RUBRIC = """You are an experienced university lecturer reviewing two versions (A and B) of lecture notes on the
same topic. Judge them as a student would use them for learning. Criteria (1-10 each):
- correctness: mathematical/factual accuracy, no hallucinated claims
- clarity: explanations, intuition before formalism, worked examples
- structure: logical order, sections, boxes (definitions/theorems/examples) used well
- figures: figures are relevant, correct, readable and referenced in the text
- typography: visual quality of the rendered pages
Return only JSON: {"a": {"correctness":..,"clarity":..,"structure":..,"figures":..,"typography":..},
"b": {...}, "preference": "A" | "B" | "tie", "reason": "<one sentence>"}"""


def _pages_png(pdf: Path, n: int, dpi: int = 110) -> list[str]:  # 70 dpi is too coarse to judge typography
    import pymupdf

    doc = pymupdf.open(str(pdf))
    out = []
    for i in range(min(n, doc.page_count)):
        out.append(base64.b64encode(doc[i].get_pixmap(dpi=dpi).tobytes("png")).decode())
    return out


def _content(label: str, run: Path, pages: int) -> list[dict]:
    blocks: list[dict] = [{"type": "text", "text": f"=== Version {label}: rendered pages ==="}]
    for b64 in _pages_png(run / "notes.pdf", pages):
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
    tex = (run / "notes.tex").read_text(encoding="utf-8", errors="replace") if (run / "notes.tex").exists() else ""
    body = tex.split("\\begin{document}", 1)[-1][:12000]
    blocks.append({"type": "text", "text": f"=== Version {label}: LaTeX source excerpt ===\n{body}"})
    return blocks


def judge_pair(client, model: str, a: Path, b: Path, pages: int) -> dict:
    verdicts = []
    for first, second, names in ((a, b, ("A", "B")), (b, a, ("B", "A"))):
        content = _content("A", first, pages) + _content("B", second, pages) + [{"type": "text", "text": RUBRIC}]
        # server-side refusal fallback keeps long judge runs from dying on a single declined request
        with client.beta.messages.stream(model=model, max_tokens=4000, messages=[{"role": "user", "content": content}],
                                         output_config={"effort": "medium"},
                                         betas=["server-side-fallback-2026-07-01"], fallbacks="default") as s:
            msg = s.get_final_message()
        if msg.stop_reason == "refusal":
            verdicts.append(None)
            continue
        data = extract_json_object("".join(x.text for x in msg.content if x.type == "text"))
        pref = data.get("preference", "tie")
        # map back to the original identities
        mapped = {"A": names[0], "B": names[1]}.get(pref, "tie")
        verdicts.append({"pref": mapped, "scores_first": data.get("a"), "scores_second": data.get("b"),
                         "reason": data.get("reason")})
    prefs = [v["pref"] for v in verdicts if v]
    final = prefs[0] if len(prefs) == 2 and prefs[0] == prefs[1] else "tie"
    return {"final": final, "verdicts": verdicts}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    import anthropic

    client = anthropic.Anthropic()
    results = {}
    for run_a in sorted(p for p in a.a.iterdir() if (p / "notes.pdf").exists()):
        run_b = a.b / run_a.name
        if (run_b / "notes.pdf").exists():
            results[run_a.name] = judge_pair(client, a.model, run_a, run_b, a.pages)
            print(run_a.name, results[run_a.name]["final"])
    wins = {k: sum(1 for r in results.values() if r["final"] == k) for k in ("A", "B", "tie")}
    print(json.dumps({"A": str(a.a), "B": str(a.b), **wins}, indent=1))
    if a.out:
        a.out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
