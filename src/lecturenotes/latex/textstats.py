"""Measure LaTeX content: prose word count, structure, density (no TeX run needed)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

_COMMENT = re.compile(r"(?<!\\)%.*$", re.M)
_VERB = re.compile(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", re.S)
_FIG = re.compile(r"<figure\b.*?</figure>|<slide\b[^>]*/?>", re.S | re.I)
_DISPLAY = re.compile(r"\\begin\{(equation|align|gather|multline|flalign|alignat|eqnarray)\*?\}.*?\\end\{\1\*?\}|\\\[.*?\\\]|\$\$.*?\$\$", re.S)
_INLINE = re.compile(r"(?<!\\)\$[^$]+?(?<!\\)\$|\\\(.*?\\\)", re.S)
_DROP_ARG = re.compile(r"\\(?:label|ref|eqref|autoref|pageref|cite|url|includegraphics|hspace|vspace|color|lnfigure)\*?(?:\[[^\]]*\])?\{[^{}]*\}")
_KEEP_ARG = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}")
_CMD = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?|\\.")
_WORD = re.compile(r"[^\W\d_](?:[\w'-]*\w)?", re.U)


def strip_latex(src: str) -> str:
    """Approximate plain-text rendering of a LaTeX fragment (for counting / coverage)."""
    s = _COMMENT.sub("", src)
    s = _VERB.sub(" ", s)
    s = _FIG.sub(" ", s)
    s = _DISPLAY.sub(" ", s)
    s = _INLINE.sub(" X ", s)
    s = _DROP_ARG.sub(" ", s)
    s = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\[([^\]]*)\])?", lambda m: f" {m.group(1) or ''} ", s)
    for _ in range(4):  # unwrap nested \textbf{\emph{x}}
        s2 = _KEEP_ARG.sub(r" \1 ", s)
        if s2 == s:
            break
        s = s2
    s = _CMD.sub(" ", s)
    s = s.replace("{", " ").replace("}", " ").replace("~", " ").replace("&", " ")
    return re.sub(r"[ \t]+", " ", s)


def word_count(src: str) -> int:
    return len(_WORD.findall(strip_latex(src)))


@dataclass
class TexStats:
    words: int
    display_math: int
    inline_math: int
    figures: int
    subsections: int
    lists: int
    tables: int
    code_listings: int
    boxes: dict
    chars: int
    cliches: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# filler a small model leans on ("In summary, X is a powerful tool ..."); counted, then penalised in the reward
CLICHES = re.compile(
    r"\b(?:is an? (?:powerful|essential|crucial|fundamental|indispensable|invaluable) (?:tool|concept|technique|algorithm|method)"
    r"|plays? an? (?:crucial|vital|key|pivotal|important) role|in the realm of|delves? into|it is (?:important|worth) (?:to note|noting)"
    r"|in today's|a cornerstone of|is crucial for (?:understanding|designing|ensuring)"
    r"|in (?:summary|conclusion),)", re.I)


def count_cliches(src: str) -> int:
    return len(CLICHES.findall(strip_latex(src)))


_BOX = ("definition", "theorem", "lemma", "proposition", "corollary", "example", "remark", "keyidea",
        "warning", "summary", "exercise", "solution", "proof")


def tex_stats(src: str) -> TexStats:
    s = _COMMENT.sub("", src)
    envs = Counter(re.findall(r"\\begin\{([a-zA-Z*]+)\}", s))
    return TexStats(
        words=word_count(src),
        display_math=len(_DISPLAY.findall(s)),
        inline_math=len(_INLINE.findall(s)),
        figures=len(_FIG.findall(s)) + envs.get("tikzpicture", 0),
        subsections=len(re.findall(r"\\subsection\*?\{", s)),
        lists=envs.get("itemize", 0) + envs.get("enumerate", 0) + envs.get("description", 0),
        tables=envs.get("tabular", 0),
        code_listings=envs.get("lstlisting", 0),
        boxes={k: envs[k] for k in _BOX if envs.get(k)},
        chars=len(src),
        cliches=count_cliches(src),
    )


def repetition_rate(src: str, n: int = 4) -> float:
    """Fraction of repeated word n-grams; a degenerate-loop / reward-hacking detector."""
    words = [w.lower() for w in _WORD.findall(strip_latex(src))]
    if len(words) < n + 5:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    c = Counter(grams)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return repeated / max(len(grams), 1)


def normalize_term(t: str) -> str:
    t = strip_latex(t).lower()
    t = re.sub(r"[^\w\s-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _stem(w: str) -> str:
    for suf in ("ization", "isation", "ations", "ation", "ments", "ment", "ness", "ities", "ity", "ies", "ing",
                "ed", "es", "s", "al", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def term_present(term: str, text_norm_stems: set[str], text_norm: str) -> bool:
    t = normalize_term(term)
    if not t:
        return True
    if t in text_norm:
        return True
    toks = [_stem(w) for w in t.split() if len(w) > 2]
    if not toks:
        return True
    hit = sum(1 for w in toks if w in text_norm_stems)
    return hit / len(toks) >= 0.7


def coverage(terms: list[str], src: str) -> tuple[float, list[str]]:
    """Share of key terms that appear (fuzzily) in the LaTeX text."""
    if not terms:
        return 1.0, []
    text = normalize_term(src)
    stems = {_stem(w) for w in text.split()}
    missing = [t for t in terms if not term_present(t, stems, text)]
    return 1 - len(missing) / len(terms), missing
