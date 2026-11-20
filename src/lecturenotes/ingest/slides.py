"""Load lecture presentations (PDF, PPTX, Beamer .tex, Markdown) into a common SlideDeck."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .selection import parse_selection


@dataclass
class Slide:
    index: int  # 1-based
    title: str = ""
    text: str = ""
    notes: str = ""
    n_images: int = 0
    n_drawings: int = 0
    image_path: Path | None = None

    @property
    def has_visual(self) -> bool:
        return self.n_images > 0 or self.n_drawings > 25

    def render(self, max_chars: int = 1200) -> str:
        body = self.text.strip()
        if len(body) > max_chars:
            body = body[:max_chars] + " ..."
        extra = f"\n  (speaker notes: {self.notes.strip()[:400]})" if self.notes.strip() else ""
        vis = " [slide contains a diagram/image]" if self.has_visual else ""
        return f"[slide {self.index}] {self.title}{vis}\n{body}{extra}"


@dataclass
class SlideDeck:
    path: Path | None
    slides: list[Slide] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.slides)

    def select(self, spec: str | None) -> "SlideDeck":
        keep = set(parse_selection(spec, len(self.slides)))
        return SlideDeck(self.path, [s for s in self.slides if s.index in keep])

    @property
    def indices(self) -> list[int]:
        return [s.index for s in self.slides]

    def get(self, idx: int) -> Slide | None:
        return next((s for s in self.slides if s.index == idx), None)

    def digest(self, per_slide: int = 160, max_chars: int = 9000) -> str:
        """Compact overview for the planner."""
        lines = []
        for s in self.slides:
            body = re.sub(r"\s+", " ", s.text).strip()
            lines.append(f"{s.index}. {s.title or '(untitled)'} -- {body[:per_slide]}")
        text = "\n".join(lines)
        if len(text) > max_chars:  # very long decks: shorten excerpts uniformly
            k = max(40, per_slide * max_chars // len(text))
            return self.digest(per_slide=k, max_chars=10 ** 9)
        return text

    def text_for(self, indices: list[int], max_chars: int = 5000) -> str:
        chunks = [s.render() for s in self.slides if s.index in set(indices)]
        out = "\n\n".join(chunks)
        return out if len(out) <= max_chars else out[:max_chars] + "\n[...]"

    def images(self) -> dict[int, Path]:
        return {s.index: s.image_path for s in self.slides if s.image_path}


# ----------------------------------------------------------------------------- loaders

def _clean(t: str) -> str:
    t = t.replace("\u00a0", " ").replace("\x0c", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def load_pdf(path: Path, render_dir: Path | None = None, dpi: int = 110) -> SlideDeck:
    import pymupdf

    doc = pymupdf.open(str(path))
    slides = []
    for i, page in enumerate(doc, 1):
        d = page.get_text("dict")
        spans = []
        for b in d.get("blocks", []):
            for line in b.get("lines", []):
                for sp in line.get("spans", []):
                    if sp.get("text", "").strip():
                        spans.append((sp["size"], sp["bbox"][1], sp["text"].strip()))
        title = ""
        if spans:
            top_size = max(s[0] for s in spans)
            title_spans = [s for s in spans if s[0] >= top_size - 0.5 and s[1] < page.rect.height * 0.35]
            title = " ".join(s[2] for s in sorted(title_spans, key=lambda s: s[1]))[:160]
        text = _clean(page.get_text("text"))
        if title and text.startswith(title[:30]):
            text = text[len(title):].strip()
        n_draw = len(page.get_drawings())
        slide = Slide(index=i, title=title, text=text, n_images=len(page.get_images()), n_drawings=n_draw)
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
            out = render_dir / f"slide-{i:03d}.png"
            page.get_pixmap(dpi=dpi).save(str(out))
            slide.image_path = out
        slides.append(slide)
    return SlideDeck(path, slides)


def load_pptx(path: Path, render_dir: Path | None = None) -> SlideDeck:
    from pptx import Presentation
    from pptx.util import Emu  # noqa: F401  (ensures python-pptx is functional)

    prs = Presentation(str(path))
    slides = []
    for i, s in enumerate(prs.slides, 1):
        title, texts, n_img = "", [], 0
        for shape in s.shapes:
            if shape.shape_type == 13:  # picture
                n_img += 1
            if getattr(shape, "has_chart", False) and shape.has_chart:
                n_img += 1
            if shape.has_text_frame:
                t = "\n".join(p.text for p in shape.text_frame.paragraphs if p.text.strip())
                if not t:
                    continue
                if s.shapes.title is not None and shape.shape_id == s.shapes.title.shape_id:
                    title = t
                else:
                    texts.append(t)
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [" | ".join(c.text for c in r.cells) for r in shape.table.rows]
                texts.append("\n".join(rows))
        notes = ""
        if s.has_notes_slide and s.notes_slide.notes_text_frame is not None:
            notes = s.notes_slide.notes_text_frame.text
        slides.append(Slide(index=i, title=title.strip(), text=_clean("\n".join(texts)), notes=notes, n_images=n_img))
    deck = SlideDeck(path, slides)
    if render_dir is not None:  # optional: render through LibreOffice when available
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice:
            with tempfile.TemporaryDirectory() as tmp:
                r = subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(path)],
                                   capture_output=True, timeout=180)
                pdf = Path(tmp) / (path.stem + ".pdf")
                if r.returncode == 0 and pdf.exists():
                    rendered = load_pdf(pdf, render_dir)
                    for s, rs in zip(deck.slides, rendered.slides):
                        s.image_path = rs.image_path
    return deck


def load_beamer(path: Path) -> SlideDeck:
    src = path.read_text(encoding="utf-8", errors="replace")
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", src, re.S)
    slides = []
    for i, f in enumerate(frames, 1):
        t = re.search(r"\\frametitle\{([^}]*)\}", f) or re.match(r"\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", f)
        title = t.group(1) if t else ""
        body = re.sub(r"\\frametitle\{[^}]*\}", "", f)
        slides.append(Slide(index=i, title=title, text=_clean(body), n_drawings=body.count("tikzpicture") * 30))
    return SlideDeck(path, slides)


def load_markdown(path: Path) -> SlideDeck:
    src = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"^\s*---\s*$", src, flags=re.M) if re.search(r"^\s*---\s*$", src, re.M) else \
        re.split(r"(?=^#{1,2} )", src, flags=re.M)
    slides = []
    for p in parts:
        if not p.strip():
            continue
        m = re.search(r"^#{1,3} (.+)$", p, re.M)
        title = m.group(1).strip() if m else ""
        body = p.replace(m.group(0), "", 1) if m else p
        slides.append(Slide(index=len(slides) + 1, title=title, text=_clean(body),
                            n_images=len(re.findall(r"!\[", body))))
    return SlideDeck(path, slides)


def load_deck(path: str | Path, render_dir: str | Path | None = None) -> SlideDeck:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    rd = Path(render_dir) if render_dir else None
    suf = p.suffix.lower()
    if suf == ".pdf":
        return load_pdf(p, rd)
    if suf in (".pptx", ".ppt"):
        return load_pptx(p, rd)
    if suf == ".tex":
        return load_beamer(p)
    if suf in (".md", ".markdown", ".txt"):
        return load_markdown(p)
    raise ValueError(f"unsupported presentation format: {suf}")
