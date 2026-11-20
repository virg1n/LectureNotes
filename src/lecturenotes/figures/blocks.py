"""Figure blocks: the model embeds executable figures inside LaTeX as

    <figure id="sigmoid" kind="matplotlib" caption="The logistic function." width="0.7">
    import numpy as np
    ...
    </figure>

    <slide n="5" caption="Original slide 5."/>

The harness extracts them, renders them in a sandbox and replaces each block
with an ``\\lnfigure`` line followed by ``%`` filler lines, so line numbers of
the surrounding LaTeX are preserved exactly (compiler errors map 1:1 back to
what the model wrote).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Issue, normalize_kind
from ..utils import slugify

FIG_RE = re.compile(r"<figure\b(?P<attrs>[^>]*?)(?<!/)>(?P<body>.*?)</figure\s*>", re.S | re.I)
SLIDE_RE = re.compile(r"<slide\b(?P<attrs>[^>]*?)/?>", re.I)
OPEN_RE = re.compile(r"<figure\b", re.I)
SELF_FIG_RE = re.compile(r"<figure\b(?P<attrs>[^>]*?)/>", re.I)
ATTR_RE = re.compile(r"""([A-Za-z_][\w-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")
FENCE_RE = re.compile(r"^\s*```[\w+-]*\s*\n(.*?)\n\s*```\s*$", re.S)


@dataclass
class FigureBlock:
    id: str
    kind: str  # "matplotlib" | "tikz" | "slide"
    caption: str
    code: str
    width: float | None
    start: int  # char offsets in the source
    end: int
    start_line: int
    end_line: int
    attrs: dict = field(default_factory=dict)
    slide_no: int | None = None

    @property
    def label(self) -> str:
        return f"fig:{self.id}"

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1


def parse_attrs(s: str) -> dict[str, str]:
    out = {}
    for m in ATTR_RE.finditer(s or ""):
        out[m.group(1).lower()] = next(g for g in m.groups()[1:] if g is not None)
    return out


def strip_fences(code: str) -> str:
    m = FENCE_RE.match(code.strip("\n"))
    if m:
        return m.group(1)
    code = re.sub(r"^\s*```[\w+-]*\s*$", "", code, flags=re.M)
    return code


def _dedent(code: str) -> str:
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    k = min(indents) if indents else 0
    return "\n".join(l[k:] for l in lines)


def _width(v: str | None) -> float | None:
    if not v:
        return None
    v = v.strip().replace("\\linewidth", "").replace("\\textwidth", "")
    try:
        if v.endswith("%"):
            w = float(v[:-1]) / 100
        else:
            w = float(v)
        return w if 0.1 <= w <= 1.0 else None
    except ValueError:
        return None


def parse_blocks(src: str) -> tuple[list[FigureBlock], list[Issue]]:
    blocks: list[FigureBlock] = []
    issues: list[Issue] = []
    seen: set[str] = set()
    spans: list[tuple[int, int]] = []

    def line_of(pos: int) -> int:
        return src.count("\n", 0, pos) + 1

    for m in FIG_RE.finditer(src):
        attrs = parse_attrs(m.group("attrs"))
        code = _dedent(strip_fences(m.group("body")))
        kind = normalize_kind(attrs.get("kind") or attrs.get("type") or ("tikz" if "\\begin{tikzpicture}" in code or "\\draw" in code else "matplotlib"))
        fid = slugify(attrs.get("id") or attrs.get("label", "").replace("fig:", "") or f"fig{len(blocks) + 1}", 32, "fig")
        base, k = fid, 2
        while fid in seen:
            fid = f"{base}-{k}"
            k += 1
        if fid != base:
            issues.append(Issue(source="figure", severity="warning", code="dup_figure_id", figure_id=fid,
                                line=line_of(m.start()), message=f"duplicate figure id '{base}' renamed to '{fid}'"))
        seen.add(fid)
        caption = (attrs.get("caption") or attrs.get("title") or "").strip()
        if kind == "web" and not code.strip():
            code = (attrs.get("query") or caption).strip()
        b = FigureBlock(
            id=fid, kind=kind, caption=caption, code=code, width=_width(attrs.get("width")),
            start=m.start(), end=m.end(), start_line=line_of(m.start()), end_line=line_of(m.end()),
            attrs=attrs,
        )
        if not code.strip():
            issues.append(Issue(source="figure", code="empty_figure", figure_id=fid, line=b.start_line,
                                message="figure block has no code"))
        if not caption:
            issues.append(Issue(source="figure", severity="warning", code="no_caption", figure_id=fid,
                                line=b.start_line, message="figure has no caption attribute"))
        blocks.append(b)
        spans.append((m.start(), m.end()))

    for m in SELF_FIG_RE.finditer(src):  # <figure kind="web" query="..." caption="..."/>
        if any(s <= m.start() < e for s, e in spans):
            continue
        attrs = parse_attrs(m.group("attrs"))
        fid = slugify(attrs.get("id") or f"img{len(blocks) + 1}", 32, "img")
        base, k = fid, 2
        while fid in seen:
            fid = f"{base}-{k}"
            k += 1
        seen.add(fid)
        q = (attrs.get("query") or attrs.get("search") or attrs.get("caption") or "").strip()
        blocks.append(FigureBlock(id=fid, kind="web", caption=(attrs.get("caption") or "").strip(), code=q,
                                  width=_width(attrs.get("width")) or 0.6, start=m.start(), end=m.end(),
                                  start_line=line_of(m.start()), end_line=line_of(m.end()), attrs=attrs))
        spans.append((m.start(), m.end()))

    for m in SLIDE_RE.finditer(src):
        if any(s <= m.start() < e for s, e in spans):
            continue
        attrs = parse_attrs(m.group("attrs"))
        try:
            n = int(attrs.get("n") or attrs.get("number") or attrs.get("page") or "0")
        except ValueError:
            n = 0
        fid = slugify(attrs.get("id") or f"slide-{n}", 32, "slide")
        base, k = fid, 2
        while fid in seen:
            fid = f"{base}-{k}"
            k += 1
        seen.add(fid)
        blocks.append(FigureBlock(
            id=fid, kind="slide", caption=(attrs.get("caption") or "").strip(), code="",
            width=_width(attrs.get("width")) or 0.8, start=m.start(), end=m.end(),
            start_line=line_of(m.start()), end_line=line_of(m.end()), attrs=attrs, slide_no=n,
        ))
        spans.append((m.start(), m.end()))

    # unterminated <figure ...> tags
    for m in OPEN_RE.finditer(src):
        if not any(s <= m.start() < e for s, e in spans):
            issues.append(Issue(source="figure", code="unclosed_figure", line=line_of(m.start()),
                                message="<figure> block is not closed with </figure>",
                                hint="close every figure block with </figure>"))
    blocks.sort(key=lambda b: b.start)
    return blocks, issues


def replace_blocks(src: str, blocks: list[FigureBlock], render) -> str:
    """Replace each block by ``render(block)`` (a single line) + ``%`` filler lines,
    preserving the total number of lines."""
    out = []
    pos = 0
    for b in sorted(blocks, key=lambda b: b.start):
        out.append(src[pos : b.start])
        first = render(b).replace("\n", " ")
        filler = "\n%" * (b.end_line - b.start_line)
        out.append(first + filler)
        pos = b.end
    out.append(src[pos:])
    return "".join(out)


def format_block(b: FigureBlock, code: str | None = None, caption: str | None = None) -> str:
    """Serialize a block back to the model-facing <figure> syntax."""
    if b.kind == "web":
        cap = (caption if caption is not None else b.caption).replace('"', "''")
        pin = (b.attrs or {}).get("file")
        pin = f' file="{pin}"' if pin else ""
        return f'<figure id="{b.id}" kind="web" query="{(code or b.code).strip()}"{pin} caption="{cap}"/>'
    if b.kind == "slide":
        return f'<slide n="{b.slide_no}" caption="{(caption or b.caption).replace(chr(34), chr(39))}"/>'
    cap = (caption if caption is not None else b.caption).replace('"', "''")
    width = f' width="{b.width:.2f}"' if b.width else ""
    return f'<figure id="{b.id}" kind="{b.kind}" caption="{cap}"{width}>\n{(code if code is not None else b.code).strip()}\n</figure>'


def replace_block_source(src: str, b: FigureBlock, new_text: str) -> str:
    return src[: b.start] + new_text + src[b.end :]


def remove_block(src: str, b: FigureBlock) -> str:
    return remove_figure_refs(src[: b.start] + src[b.end :], [b.id])


def mask_blocks(src: str, blocks: list[FigureBlock] | None = None) -> str:
    """Replace blocks with comment lines (used before linting prose)."""
    if blocks is None:
        blocks, _ = parse_blocks(src)
    return replace_blocks(src, blocks, lambda b: f"%LNFIGURE {b.id}")


def remove_figure_refs(src: str, fig_ids: list[str]) -> str:
    """Drop references to figures that could not be rendered."""
    for fid in fig_ids:
        lab = re.escape(f"fig:{fid}")
        src = re.sub(rf"\s*\((?:see |cf\. )?(?:Figure|Fig\.|figure)~?\s*\\(?:auto)?ref\{{{lab}\}}\)", "", src)
        src = re.sub(rf"\s*(?:Figure|Fig\.|figure)~?\s*\\ref\{{{lab}\}}", "", src)
        src = re.sub(rf"\\(?:auto)?ref\{{{lab}\}}", "", src)
    return src
