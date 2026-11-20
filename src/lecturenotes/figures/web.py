"""Web images for figure blocks of kind="web":

    <figure id="mito" kind="web" query="mitochondrion structure" caption="Structure of a mitochondrion."/>
    <figure id="mito" kind="web" query="..." file="File:Mitochondrion structure.svg" caption="..."/>   (chosen)

Images come from Wikimedia Commons (freely licensed; the API returns license, author, description and
categories, so the harness can attach attribution and a *text* model can judge relevance).

Flow: `candidates(query)` returns the top-k licence-clean images with metadata -> a model picks one or
answers "none" (task pick_image, see agent/section.py) -> the chosen file is pinned in the block
(`file=`) and `fetch_file` downloads exactly that file. Without a choice, `fetch_image` takes the first
acceptable candidate.

Downloads are size-capped, MIME-checked, decoded with Pillow (anything that is not a real image is
rejected) and re-encoded to PNG, so no untrusted bytes reach LaTeX.
"""

from __future__ import annotations

import html
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..utils import stable_hash

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": os.environ.get("LN_HTTP_UA", "LectureNotesAgent/0.1 (https://www.mediawiki.org/wiki/API:Etiquette; educational use) python-httpx")}
OK_LICENSES = re.compile(r"^(cc0|cc[- ]by(-sa)?([- ]\d(\.\d)?)?|public domain|pd[-\w]*|gfdl.*)$", re.I)
MAX_BYTES = 8 << 20
_IIPROP = {"prop": "imageinfo", "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 1200}


@dataclass
class WebImage:
    path: Path
    title: str
    author: str
    license: str
    source_url: str

    def attribution(self) -> str:
        author = re.sub(r"<[^>]+>", "", self.author).strip()[:60] or "unknown author"
        return f"Image: {author}, {self.license}, Wikimedia Commons"


def _text(v: str, limit: int) -> str:
    v = html.unescape(re.sub(r"<[^>]+>", " ", v or ""))
    v = re.sub(r"\s+", " ", v).strip()
    return v[:limit]


def _parse_pages(pages: dict) -> list[dict]:
    out = []
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata") or {}
        get = lambda k: (meta.get(k) or {}).get("value", "")  # noqa: E731
        out.append({
            "title": p.get("title", ""), "url": ii.get("thumburl") or ii.get("url"), "mime": ii.get("mime", ""),
            "width": ii.get("width", 0), "height": ii.get("height", 0), "license": _text(get("LicenseShortName"), 40),
            "author": get("Artist"), "page": ii.get("descriptionurl", ""),
            "description": _text(get("ImageDescription"), 300),
            "categories": [c for c in _text(get("Categories"), 400).split("|") if c][:8],
        })
    return out


def search_commons(query: str, limit: int = 12, timeout: float = 20) -> list[dict]:
    params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrsearch": query,
              "gsrlimit": limit, **_IIPROP}
    r = httpx.get(API, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    return _parse_pages((r.json().get("query") or {}).get("pages") or {})


def acceptable(c: dict, min_width: int = 300) -> bool:
    return bool(c["url"]) and bool(OK_LICENSES.match(c["license"].strip())) and c["width"] >= min_width \
        and c["mime"].startswith("image/") and c["mime"] != "image/gif"


def candidates(query: str, k: int = 6, cache_dir: Path | None = None) -> list[dict]:
    """Top-k licence-clean candidates with metadata (cached per query)."""
    cp = cache_dir / f"cand-{stable_hash('commons-cand', query.lower().strip())}.json" if cache_dir else None
    if cp is not None and cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))[:k]
    if os.environ.get("LN_OFFLINE"):
        return []
    try:
        cands = [c for c in search_commons(query) if acceptable(c)][:max(k, 8)]
    except Exception:
        return []
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(cands), encoding="utf-8")
    return cands[:k]


def describe_candidates(cands: list[dict]) -> str:
    """Text rendering of candidates for a (text-only) judge model."""
    lines = []
    for i, c in enumerate(cands, 1):
        title = re.sub(r"^File:", "", c["title"])
        kind = "vector drawing" if "svg" in c["mime"] else ("photo/raster" if "jpeg" in c["mime"] else "raster image")
        cats = "; ".join(c.get("categories") or []) or "-"
        lines.append(f"[{i}] {title} ({kind}, {c['width']}x{c.get('height', 0)}px)\n"
                     f"    description: {c.get('description') or '-'}\n    categories: {cats}")
    return "\n".join(lines)


def _download(c: dict, png: Path) -> bool:
    from PIL import Image

    try:
        with httpx.stream("GET", c["url"], headers=UA, timeout=30, follow_redirects=True) as r:
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
                return False
            buf = io.BytesIO()
            for chunk in r.iter_bytes():
                buf.write(chunk)
                if buf.tell() > MAX_BYTES:
                    return False
        im = Image.open(io.BytesIO(buf.getvalue()))
        im.load()
        if im.mode not in ("RGB", "L"):
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
            im = bg
        im.thumbnail((1600, 1600))
        png.parent.mkdir(parents=True, exist_ok=True)
        im.save(png, "PNG", optimize=True)
        return True
    except Exception:
        return False


def _publish(cache_png: Path, meta: dict, out_dir: Path) -> WebImage:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / cache_png.name
    if not dst.exists():
        dst.write_bytes(cache_png.read_bytes())
    return WebImage(dst, meta["title"], meta["author"], meta["license"], meta["page"])


def download_candidate(c: dict, out_dir: Path, cache_dir: Path) -> WebImage | None:
    key = stable_hash("commons-file", c["title"])
    png = cache_dir / f"{key}.png"
    if not png.exists() and not _download(c, png):
        return None
    (cache_dir / f"{key}.json").write_text(json.dumps(c), encoding="utf-8")  # lets fetch_file reuse it
    return _publish(png, c, out_dir)


def fetch_file(title: str, out_dir: Path, cache_dir: Path) -> WebImage | None:
    """Download one specific Commons file (a pinned choice), with metadata (cached by title)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = stable_hash("commons-file", title)
    png, meta_p = cache_dir / f"{key}.png", cache_dir / f"{key}.json"
    if png.exists() and meta_p.exists():
        return _publish(png, json.loads(meta_p.read_text(encoding="utf-8")), out_dir)
    if os.environ.get("LN_OFFLINE"):
        return None
    try:
        r = httpx.get(API, params={"action": "query", "format": "json", "titles": title, **_IIPROP}, headers=UA,
                      timeout=20)
        r.raise_for_status()
        cands = _parse_pages((r.json().get("query") or {}).get("pages") or {})
    except Exception:
        return None
    if not cands or not acceptable(cands[0], min_width=1) or not _download(cands[0], png):
        return None
    meta_p.write_text(json.dumps(cands[0]), encoding="utf-8")
    return _publish(png, cands[0], out_dir)


def fetch_image(query: str, out_dir: Path, cache_dir: Path, *, min_width: int = 300) -> WebImage | None:
    """First acceptable candidate for `query` (no relevance judgement)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for c in candidates(query, k=8, cache_dir=cache_dir):
        if c["width"] < min_width:
            continue
        img = download_candidate(c, out_dir, cache_dir)
        if img is not None:
            return img
    return None
