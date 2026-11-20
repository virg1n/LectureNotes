"""Pixel-level sanity checks on rendered figures."""

from __future__ import annotations

from pathlib import Path

from ..schemas import Issue


def ink_stats(png: str | Path) -> dict:
    import numpy as np
    from PIL import Image

    im = Image.open(png).convert("L")
    a = np.asarray(im, dtype=np.uint8)
    ink = a < 235
    frac = float(ink.mean()) if a.size else 0.0
    if ink.any():
        ys, xs = np.nonzero(ink)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    else:
        bbox = None
    return {"ink": frac, "bbox": bbox, "size": im.size}


def image_issues(png: str | Path) -> list[Issue]:
    p = Path(png)
    if not p.exists():
        return [Issue(source="figure", severity="error", code="no_image", message="no image was produced")]
    try:
        st = ink_stats(p)
    except Exception as e:  # pragma: no cover
        return [Issue(source="figure", severity="info", code="image_check_failed", message=str(e)[:200])]
    out = []
    w, h = st["size"]
    if st["ink"] < 0.002:
        out.append(Issue(source="figure", severity="error", code="blank",
                         message="the rendered figure is blank (nothing visible was drawn)"))
    elif st["ink"] < 0.01:
        out.append(Issue(source="figure", severity="warning", code="nearly_blank",
                         message="the rendered figure is almost empty"))
    if w < 40 or h < 25:
        out.append(Issue(source="figure", severity="warning", code="tiny",
                         message=f"the rendered figure is tiny ({w}x{h}px at 72dpi)"))
    return out


def text_overlap_issues(pdf: str | Path, min_frac: float = 0.25, max_report: int = 3) -> list[Issue]:
    """Labels drawn on top of each other (typical TikZ failure: nodes placed at the same coordinate).

    Words of different text lines whose boxes overlap by more than `min_frac` of the smaller box.
    On the benchmark figures this flags exactly the unreadable diagrams and nothing else."""
    try:
        import pymupdf

        with pymupdf.open(str(pdf)) as doc:
            words = doc[0].get_text("words") if doc.page_count else []
    except Exception:
        return []
    pairs = []
    for i, a in enumerate(words):
        for b in words[i + 1:]:
            if (a[5], a[6]) == (b[5], b[6]):  # same text line (glyph kerning, sub/superscripts)
                continue
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix <= 0 or iy <= 0:
                continue
            small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if small > 0 and ix * iy / small > min_frac:
                pairs.append((a[4], b[4]))
    if not pairs:
        return []
    shown = ", ".join(f"'{x}' / '{y}'" for x, y in pairs[:max_report])
    return [Issue(source="figure", severity="error", code="text_overlap",
                  message=f"{len(pairs)} text label(s) overlap and are unreadable ({shown}); move nodes apart "
                          f"(e.g. larger node distance, anchor=..., or positioning with below=of/right=of)")]
