"""Turn a free-form 'expected length' into a word budget and a section-count range."""

from __future__ import annotations

import re
from dataclasses import dataclass

WORDS_PER_PAGE = 330  # prose words per page in our layout (boxes, display math and figures take space)
NAMED = {
    "short": 4, "brief": 3, "concise": 4, "medium": 8, "normal": 8, "standard": 8, "long": 14, "detailed": 14,
    "comprehensive": 20, "кратк": 4, "коротк": 4, "средн": 8, "подробн": 14, "длинн": 14, "развернут": 14,
}


@dataclass
class LengthBudget:
    pages: float
    words: int
    n_sections: tuple[int, int]
    source: str

    @property
    def mid_sections(self) -> int:
        return (self.n_sections[0] + self.n_sections[1]) // 2


def _sections_for(pages: float) -> tuple[int, int]:
    if pages <= 3:
        return (2, 4)
    if pages <= 6:
        return (3, 5)
    if pages <= 12:
        return (4, 7)
    if pages <= 20:
        return (6, 9)
    return (7, 12)


def parse_length(spec: str | None, n_slides: int | None = None) -> LengthBudget:
    s = (spec or "").strip().lower()
    pages: float | None = None
    words: int | None = None
    src = "default"
    if s:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:-|to|–)?\s*(\d+(?:[.,]\d+)?)?\s*(pages?|pp?\b|p\b|стр\w*|лист\w*)", s)
        if m:
            a = float(m.group(1).replace(",", "."))
            b = float(m.group(2).replace(",", ".")) if m.group(2) else a
            pages, src = (a + b) / 2, "pages"
        m = re.search(r"(\d[\d ,]*)\s*(?:words?|слов\w*)", s)
        if m and pages is None:
            words, src = int(re.sub(r"[ ,]", "", m.group(1))), "words"
        if pages is None and words is None:
            for k, v in NAMED.items():
                if k in s:
                    pages, src = float(v), f"named:{k}"
                    break
        if pages is None and words is None and re.fullmatch(r"\d+(?:\.\d+)?", s):
            pages, src = float(s), "pages"
    if pages is None and words is None:
        if n_slides:
            pages, src = max(3.0, min(25.0, 0.5 * n_slides)), "slides"
        else:
            pages = 8.0
    if words is None:
        words = int(pages * WORDS_PER_PAGE)
    else:
        pages = words / WORDS_PER_PAGE
    pages = max(1.0, min(pages, 60.0))
    words = max(300, min(words, 60 * WORDS_PER_PAGE))
    return LengthBudget(pages=round(pages, 1), words=words, n_sections=_sections_for(pages), source=src)
