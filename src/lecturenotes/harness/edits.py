"""Edit formats for repairs: SEARCH/REPLACE blocks (preferred) with robust application,
plus 'quarantine' -- deterministic removal of the smallest top-level unit that keeps failing.

Repair output format the model is trained to produce:

    <<<<<<< SEARCH
    exact lines from the current source
    =======
    corrected lines
    >>>>>>> REPLACE
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

EDIT_RE = re.compile(r"<{5,9} ?SEARCH[^\n]*\n(.*?)\n?={5,9}[^\n]*\n(.*?)\n?>{5,9} ?REPLACE", re.S)


@dataclass
class Edit:
    search: str
    replace: str


@dataclass
class EditOutcome:
    source: str
    applied: int
    failed: int
    mode: str  # "edits" | "full" | "none"


def parse_edits(text: str) -> list[Edit]:
    return [Edit(m.group(1), m.group(2)) for m in EDIT_RE.finditer(text)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _find_fuzzy(src_lines: list[str], search_lines: list[str], cutoff: float = 0.86) -> tuple[int, int] | None:
    n = len(search_lines)
    if n == 0 or n > len(src_lines):
        return None
    target = _norm("\n".join(search_lines))
    best, best_i = 0.0, -1
    for i in range(len(src_lines) - n + 1):
        cand = _norm("\n".join(src_lines[i : i + n]))
        r = difflib.SequenceMatcher(None, cand, target, autojunk=False).ratio()
        if r > best:
            best, best_i = r, i
    return (best_i, best_i + n) if best >= cutoff else None


def apply_edit(src: str, e: Edit) -> str | None:
    if e.search and e.search in src:
        return src.replace(e.search, e.replace, 1)
    s_lines = e.search.strip("\n").split("\n")
    src_lines = src.split("\n")
    # whitespace-insensitive line match
    ns = [_norm(l) for l in s_lines]
    for i in range(len(src_lines) - len(s_lines) + 1):
        if [_norm(l) for l in src_lines[i : i + len(s_lines)]] == ns:
            return "\n".join(src_lines[:i] + e.replace.split("\n") + src_lines[i + len(s_lines):])
    span = _find_fuzzy(src_lines, s_lines)
    if span:
        i, j = span
        return "\n".join(src_lines[:i] + e.replace.split("\n") + src_lines[j:])
    return None


def locate_edit(src: str, e: Edit) -> tuple[int, int] | None:
    """Character span of an edit's SEARCH text in `src` (exact, then whitespace-insensitive line match)."""
    if e.search and src.count(e.search) >= 1:
        i = src.find(e.search)
        return i, i + len(e.search)
    s_lines = e.search.strip("\n").split("\n")
    lines = src.split("\n")
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l) + 1)
    ns = [_norm(l) for l in s_lines]
    for i in range(len(lines) - len(s_lines) + 1):
        if [_norm(l) for l in lines[i : i + len(s_lines)]] == ns:
            return offs[i], offs[i + len(s_lines)] - 1
    return None


def apply_model_repair(src: str, output: str, *, expect_prefix: str | None = "\\section") -> EditOutcome:
    """Apply a repair answer: SEARCH/REPLACE blocks, or a complete rewritten fragment.

    All edits are located in the *original* source and applied atomically (edits are
    written against the text the model saw, not against intermediate states).
    Edits that cannot be located exactly fall back to fuzzy sequential application."""
    edits = parse_edits(output)
    if edits:
        spans, rest = [], []
        for e in edits:
            sp = locate_edit(src, e)
            (spans.append((sp[0], sp[1], e.replace)) if sp else rest.append(e))
        spans.sort()
        cur, pos, ok, parts = src, 0, 0, []
        for s, t, rep in spans:
            if s < pos:  # overlapping edits: keep the first
                rest.append(None)
                continue
            parts.append(src[pos:s])
            parts.append(rep)
            pos, ok = t, ok + 1
        parts.append(src[pos:])
        cur = "".join(parts)
        bad = sum(1 for e in rest if e is None)
        for e in rest:
            if e is None:
                continue
            new = apply_edit(cur, e)  # fuzzy fallback
            if new is None:
                bad += 1
            else:
                cur, ok = new, ok + 1
        return EditOutcome(cur, ok, bad, "edits")
    body = strip_wrapping(output)
    if expect_prefix is None or body.lstrip().startswith(expect_prefix):
        if len(body) > 0.4 * len(src):  # guard against truncated rewrites
            return EditOutcome(body, 1, 0, "full")
    return EditOutcome(src, 0, 1, "none")


def strip_wrapping(text: str) -> str:
    """Remove ```latex fences / prose around a LaTeX answer."""
    t = text.strip()
    m = re.search(r"```(?:latex|tex)?\s*\n(.*?)\n```", t, re.S)
    if m:
        return m.group(1).strip()
    return t


# ---------------------------------------------------------------------------- references

_REF_PHRASE_TMPL = (r"\s*\(?(?:see |cf\. |in |of )?(?:Figure|Fig\.|Table|Section|Sec\.|Chapter|Eq\.|Equation|Lemma|"
                    r"Theorem|Definition|Proposition|Corollary|Example|Exercise|Remark|Appendix)?~?\s*"
                    r"\\(?:eq|auto|c|C|page|name)?ref\{{{lab}\}}\)?")
_ANY_REF = re.compile(r"\\(?:eq|auto|c|C|page|name)?ref\{([^}]+)\}")


def remove_refs(src: str, labels) -> str:
    """Drop references (with their 'see Figure~' / '(Eq.~...)' phrase) to the given labels."""
    for lab in set(labels):
        src = re.sub(_REF_PHRASE_TMPL.format(lab=re.escape(lab)), "", src)
    return src


def dangling_refs(src: str, known: set[str] | None = None) -> set[str]:
    labels = set(re.findall(r"\\label\{([^}]+)\}", src)) | {f"fig:{m}" for m in re.findall(r'<figure\b[^>]*\bid="([^"]+)"', src)}
    return {lab for lab in _ANY_REF.findall(src) if lab not in labels and lab not in (known or set())}


# ---------------------------------------------------------------------------- quarantine

_BEGIN = re.compile(r"\\begin\{([^}]+)\}")
_END = re.compile(r"\\end\{([^}]+)\}")


_HEADING = ("\\section", "\\subsection", "\\subsubsection", "\\paragraph")


def top_level_units(src: str) -> list[tuple[int, int]]:
    """Split into top-level units as 0-based inclusive line ranges: headings, whole top-level
    environments, figure blocks, and prose paragraphs. Boundaries are blank lines *and* environment /
    figure / heading edges -- models often write sections without any blank line, and a section that is
    one giant unit would be quarantined as a whole."""
    lines = src.split("\n")
    units: list[tuple[int, int]] = []
    start: int | None = None
    depth, in_fig = 0, False

    def close(end: int) -> None:
        nonlocal start
        if start is not None and end >= start:
            units.append((start, end))
        start = None

    for i, line in enumerate(lines):
        code = re.sub(r"(?<!\\)%.*$", "", line)
        stripped = code.lstrip()
        top = depth == 0 and not in_fig
        if top and not code.strip():
            close(i - 1)
            continue
        opens_fig = top and re.search(r"<figure\b", code, re.I) is not None
        starts_env = top and stripped.startswith("\\begin")
        heading = top and stripped.startswith(_HEADING)
        if (opens_fig or starts_env or heading) and start is not None:
            close(i - 1)  # a block or heading never shares a unit with the preceding prose
        if start is None:
            start = i
        prev_depth = depth
        if opens_fig or (in_fig is False and re.search(r"<figure\b", code, re.I)):
            in_fig = True
        closes_fig = in_fig and re.search(r"</figure\s*>", code, re.I) is not None
        if closes_fig:
            in_fig = False
        depth = max(0, depth + len(_BEGIN.findall(code)) - len(_END.findall(code)))
        if depth == 0 and not in_fig and (heading or closes_fig or starts_env or prev_depth > 0):
            close(i)
    close(len(lines) - 1)
    return units


def quarantine(src: str, bad_lines: list[int | None], *, keep_header: bool = True) -> tuple[str, list[str]]:
    """Remove the top-level units containing `bad_lines` (1-based). Lines that are None
    (unlocated errors) quarantine the last unit. Returns (new_source, removed_snippets)."""
    lines = src.split("\n")
    units = top_level_units(src)
    if not units:
        return src, []
    targets = set()
    for bl in bad_lines:
        if bl is None:
            targets.add(len(units) - 1)
            continue
        idx = next((k for k, (a, b) in enumerate(units) if a <= bl - 1 <= b), None)
        if idx is None:
            idx = max(0, min(len(units) - 1, next((k for k, (a, _) in enumerate(units) if a > bl - 1), len(units)) - 1))
        targets.add(idx)
    removed = []
    keep = [True] * len(lines)
    for k in sorted(targets):
        a, b = units[k]
        if keep_header and lines[a].lstrip().startswith("\\section"):
            a += 1  # never drop the section heading itself
        if a > b:
            continue
        removed.append("\n".join(lines[a : b + 1]))
        for i in range(a, b + 1):
            keep[i] = False
    return "\n".join(l for l, k in zip(lines, keep) if k), removed
