"""Does the figure show what its caption says?  A vision-language judge for rendered figures.

Pixel and layout checks cannot tell a sine wave captioned "Hamming code decoding" from a real decoding
diagram. This module asks a VLM served behind an OpenAI-compatible endpoint (e.g. vLLM with
Qwen2.5-VL-7B-Instruct) to score each rendered figure against its caption and section:

    LN_VLM_URL=http://127.0.0.1:8200/v1 LN_VLM_MODEL=vlm   # enables the check inside the verifier

* inside the harness (opt-in, env above): score <= 2 is an error-level issue "irrelevant_figure", so the
  figure is repaired or dropped and the figure reward drops; score 3 is a warning
* for evaluation, over finished runs (no agent needed):

    python -m lecturenotes.figures.relevance --runs eval_runs/sft-merged --out eval_runs/sft-merged/relevance.json

Verdicts are cached by (image bytes, caption, section, model), so repeated verification is free.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import statistics
from pathlib import Path

from ..utils import stable_hash

PROMPT = """You check figures in university lecture notes.
Lecture topic: {topic}
Section: {section}
Caption: {caption}

Look at the image. Does it correctly and clearly show what the caption describes, in a way that helps a
student understand this section? Penalise: unrelated content, generic decoration (e.g. a random curve
that does not illustrate the stated concept), wrong or unreadable labels, overlapping text.
Score 1-5: 1 = unrelated or wrong, 2 = barely related, 3 = related but weak, 4 = good, 5 = excellent.
Answer exactly:
score: <1-5>
reason: <one short sentence>"""

_SCORE = re.compile(r"score\s*[:=]\s*([1-5])", re.I)


def _png_b64(path: Path, max_side: int = 896) -> str:
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(str(path)) as doc:
            data = doc[0].get_pixmap(dpi=144).tobytes("png")
        im = Image.open(io.BytesIO(data))
    else:
        im = Image.open(path)
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FigureJudge:
    def __init__(self, url: str, model: str = "vlm", cache_dir: str | Path | None = None, timeout: float = 120):
        self.url = url.rstrip("/")
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout

    @classmethod
    def from_env(cls, cache_dir: str | Path | None = None) -> "FigureJudge | None":
        url = os.environ.get("LN_VLM_URL")
        return cls(url, os.environ.get("LN_VLM_MODEL", "vlm"), cache_dir) if url else None

    def judge(self, image: str | Path, caption: str, section: str = "", topic: str = "") -> dict | None:
        """{"score": 1-5, "reason": str} or None when the judge is unreachable / answers off-format."""
        image = Path(image)
        if not image.exists():
            return None
        key = stable_hash("figjudge", image.read_bytes(), caption, section, topic, self.model)
        cp = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cp is not None and cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))
        import httpx

        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64(image)}"}},
                   {"type": "text", "text": PROMPT.format(topic=topic or "-", section=section or "-",
                                                          caption=caption or "-")}]
        try:
            r = httpx.post(f"{self.url}/chat/completions", timeout=self.timeout,
                           json={"model": self.model, "temperature": 0, "max_tokens": 80,
                                 "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except Exception:
            return None
        m = _SCORE.search(text)
        if not m:
            return None
        reason = re.search(r"reason\s*[:=]\s*(.+)", text, re.I)
        out = {"score": int(m.group(1)), "reason": (reason.group(1).strip() if reason else "")[:200]}
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(out), encoding="utf-8")
        return out


# ----------------------------------------------------------------------------- offline evaluation
_LNFIG = re.compile(r"\\lnfigure(?:here)?\{[^{}]*\}\{([^{}]+)\}\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
_SECTION = re.compile(r"\\section\*?\{([^{}]*)\}")


def figures_in_notes(tex: str) -> list[tuple[str, str, str]]:
    """(relative image path, caption, section title) for every included figure of an assembled notes.tex."""
    out = []
    tex = re.sub(r"(?m)^\s*%.*$", "", tex)  # the preamble documents \lnfigure in a comment
    for m in _LNFIG.finditer(tex):
        secs = _SECTION.findall(tex, 0, m.start())
        cap = re.sub(r"\{\\footnotesize\(.*?\)\}", "", m.group(2)).strip()  # drop the web-image credit
        out.append((m.group(1), cap, secs[-1] if secs else ""))
    return out


def judge_run(judge: FigureJudge, run: Path) -> list[dict]:
    tex_p = run / "notes.tex"
    if not tex_p.exists():
        return []
    tex = tex_p.read_text(encoding="utf-8", errors="replace")
    topic = ""
    rp = run / "report.json"
    if rp.exists():
        topic = json.loads(rp.read_text(encoding="utf-8")).get("request", {}).get("topic", "")
    rows = []
    for rel, cap, sec in figures_in_notes(tex):
        img = next((p for p in (run / "figures" / rel, run / "build" / rel, run / rel) if p.exists()), None)
        v = judge.judge(img, cap, sec, topic) if img else None
        rows.append({"figure": rel, "caption": cap, "section": sec, **(v or {"score": None})})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="benchmark output dir (one sub-dir per topic)")
    ap.add_argument("--url", default=os.environ.get("LN_VLM_URL", "http://127.0.0.1:8200/v1"))
    ap.add_argument("--model", default=os.environ.get("LN_VLM_MODEL", "vlm"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    judge = FigureJudge(a.url, a.model, cache_dir=a.runs / ".figjudge")
    results = {}
    for run in sorted(p for p in a.runs.iterdir() if p.is_dir() and not p.name.startswith(".")):
        rows = judge_run(judge, run)
        if rows:
            results[run.name] = rows
            sc = [r["score"] for r in rows if r["score"]]
            print(f"{run.name}: {len(rows)} figures, mean score {statistics.mean(sc):.2f}" if sc else run.name)
    scores = [r["score"] for rows in results.values() for r in rows if r["score"]]
    summary = {"figures": len(scores), "mean_score": round(statistics.mean(scores), 3) if scores else None,
               "relevant_share": round(sum(s >= 4 for s in scores) / len(scores), 3) if scores else None,
               "irrelevant_share": round(sum(s <= 2 for s in scores) / len(scores), 3) if scores else None}
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "runs": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
