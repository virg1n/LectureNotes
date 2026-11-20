"""Deterministic, semantics-preserving repairs applied before any model-based repair.

Small models leak Markdown, Unicode math and unescaped specials constantly; all
of that is fixed here for free. Every applied fix is recorded (it is reported
to the user, and the counts are part of the RL reward so the model learns to
not rely on the safety net).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..figures.blocks import parse_blocks
from .lint import Finding, LintResult, lint_latex, mask_for_markdown

_LANG_MAP = {"py": "Python", "python": "Python", "python3": "Python", "c": "C", "cpp": "C++", "c++": "C++",
             "java": "Java", "js": "Java", "javascript": "Java", "bash": "bash", "sh": "bash", "shell": "bash",
             "sql": "SQL", "r": "R", "matlab": "Matlab", "octave": "Octave", "haskell": "Haskell", "go": "C",
             "rust": "C", "julia": "Python", "latex": "TeX", "tex": "TeX", "html": "HTML", "": ""}


@dataclass
class Fix:
    code: str
    line: int
    description: str


@dataclass
class AutofixResult:
    source: str
    fixes: list[Fix] = field(default_factory=list)
    lint: LintResult | None = None

    @property
    def n_fixes(self) -> int:
        return len(self.fixes)


def _line_at(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def _protected_spans(src: str) -> list[tuple[int, int]]:
    blocks, _ = parse_blocks(src)
    return [(b.start, b.end) for b in blocks]


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def apply_edits(src: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply (start, end, replacement) edits; later edits in the list win ties at the
    same insertion point by being placed after earlier ones. Overlaps are skipped."""
    indexed = sorted(enumerate(edits), key=lambda t: (t[1][0], t[0]))
    out, pos = [], 0
    for _, (s, e, r) in indexed:
        if s < pos:
            continue  # overlapping edit
        out.append(src[pos:s])
        out.append(r)
        pos = max(e, s)
    out.append(src[pos:])
    return "".join(out)


# ----------------------------------------------------------------------------- pseudo environments
# "[keyidea] Redundancy is added ..." -- the model names a box environment instead of opening it
_PSEUDO_ENV = re.compile(r"^[ \t]*\[(?P<env>[A-Za-z][A-Za-z ]{1,20}?)(?::[ \t]*(?P<title>[^\]\n]{1,80}))?\][ \t]*(?=\S)",
                         re.M)


def fix_pseudo_env(src: str) -> tuple[str, list[Fix]]:
    from .lint import BOX_ENVS, ENV_ALIASES

    spans = _protected_spans(src)
    fixes: list[Fix] = []
    edits = []
    for m in _PSEUDO_ENV.finditer(src):
        name = m.group("env").strip()
        key = name.lower().replace(" ", "")
        env = key if key in BOX_ENVS else ENV_ALIASES.get(name) or ENV_ALIASES.get(key)
        if env not in BOX_ENVS or env in ("csbox", "lnoverview") or _in_spans(m.start(), spans):
            continue
        end = src.find("\n\n", m.end())
        end = len(src.rstrip("\n")) if end < 0 else end
        title = f"[{m.group('title').strip()}]" if m.group("title") else ""
        edits.append((m.start(), m.end(), f"\\begin{{{env}}}{title}\n"))
        edits.append((end, end, f"\n\\end{{{env}}}"))
        fixes.append(Fix("md_pseudo_env", _line_at(src, m.start()), f"'[{name}] ...' paragraph -> {env} environment"))
    return (apply_edits(src, edits) if edits else src), fixes


# ----------------------------------------------------------------------------- markdown

_FENCE = re.compile(r"^[ \t]*```[ \t]*([\w+#-]*)[ \t]*\n(.*?)\n[ \t]*```[ \t]*$", re.M | re.S)
_HEADING = re.compile(r"^(#{1,4})[ \t]+(\S.*?)[ \t#]*$", re.M)
_BOLD = re.compile(r"(?<![\\*])\*\*(?=\S)([^*\n]+?)(?<=\S)\*\*(?!\*)")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_ICODE = re.compile(r"(?<![`\\])`([^`'\n]{1,80})`(?!`)")


def _escape_tt(s: str) -> str:
    return re.sub(r"([_&%#$])", r"\\\1", s).replace("{", "\\{").replace("}", "\\}")


# only "50%" directly after a digit: "5 % comment" is far more likely a real comment
_LIST_MASK_ENVS = re.compile(r"\\begin\{(tabular\*?|array|tikzpicture|axis|matrix|pmatrix|bmatrix|cases|aligned)\}"
                             r".*?\\end\{\1\}", re.S)
_INLINE_MATH = re.compile(r"(?<!\\)\$[^$]+?(?<!\\)\$|\\\(.*?\\\)", re.S)
_PERCENT = re.compile(r"(?<=\d)()%(?=[ \t]*[A-Za-z(),.;:\]\-])")


def fix_percent(src: str) -> tuple[str, list[Fix]]:
    """'50% of the data' silently comments out the rest of the line in LaTeX."""
    fixes: list[Fix] = []
    spans = _protected_spans(src)
    verb = [(m.start(), m.end()) for m in re.finditer(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", src, re.S)]

    def repl(m: re.Match) -> str:
        if _in_spans(m.start(), spans) or _in_spans(m.start(), verb):
            return m.group(0)
        fixes.append(Fix("percent", _line_at(src, m.start()), "'%' after a number escaped as \\%"))
        return m.group(1) + "\\%"

    return _PERCENT.sub(repl, src), fixes


def fix_markdown(src: str) -> tuple[str, list[Fix]]:
    fixes: list[Fix] = []
    spans = _protected_spans(src)

    def fence(m: re.Match) -> str:
        if _in_spans(m.start(), spans):
            return m.group(0)
        lang = _LANG_MAP.get(m.group(1).lower(), "")
        fixes.append(Fix("md_fence", _line_at(src, m.start()), "Markdown code fence -> lstlisting"))
        opt = f"[language={lang}]" if lang else ""
        return f"\\begin{{lstlisting}}{opt}\n{m.group(2)}\n\\end{{lstlisting}}"

    src = _FENCE.sub(fence, src)
    spans = _protected_spans(src)
    masked = mask_for_markdown(src)
    edits: list[tuple[int, int, str]] = []

    has_section = "\\section" in src
    for k, m in enumerate(_HEADING.finditer(masked)):
        level = len(m.group(1))
        cmd = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}[level]
        if level == 1 and "\\section" in src[: m.start()]:
            cmd = "subsection"
        if k == 0 and not has_section and not src[: m.start()].strip():
            cmd = "section"  # a leading heading is the section title
        title = src[m.start(2) : m.end(2)].strip()
        edits.append((m.start(), m.end(), f"\\{cmd}{{{title}}}"))
        fixes.append(Fix("md_heading", _line_at(src, m.start()), f"Markdown heading -> \\{cmd}"))
    for m in _BOLD.finditer(masked):
        edits.append((m.start(), m.end(), f"\\textbf{{{src[m.start(1):m.end(1)]}}}"))
        fixes.append(Fix("md_bold", _line_at(src, m.start()), "**bold** -> \\textbf"))
    for m in _LINK.finditer(masked):
        edits.append((m.start(), m.end(), f"\\href{{{m.group(2)}}}{{{src[m.start(1):m.end(1)]}}}"))
        fixes.append(Fix("md_link", _line_at(src, m.start()), "Markdown link -> \\href"))
    # inline code: only outside inline math
    no_math = re.sub(r"(?<!\\)\$[^$\n]*\$", lambda m: " " * len(m.group(0)), masked)
    for m in _ICODE.finditer(no_math):
        edits.append((m.start(), m.end(), f"\\texttt{{{_escape_tt(src[m.start(1):m.end(1)])}}}"))
        fixes.append(Fix("md_code", _line_at(src, m.start()), "`code` -> \\texttt"))
    src = apply_edits(src, edits)

    # bullet / numbered lists outside LaTeX lists. Tables, TikZ and (multi-line) math are masked: a
    # tabular row "- & 1 \\" or a continued formula "- b$" is valid LaTeX, not a Markdown list.
    lines = src.split("\n")
    masked = mask_for_markdown(src)
    for rx in (_LIST_MASK_ENVS, _INLINE_MATH):
        masked = rx.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), masked)
    masked_lines = masked.split("\n")
    out: list[str] = []
    i = 0
    bullet = re.compile(r"^[ \t]*([-*•]|\d{1,2}[.)])[ \t]+(\S.*)$")
    while i < len(lines):
        m = bullet.match(masked_lines[i]) if i < len(masked_lines) else None
        prev = lines[i - 1].strip() if i > 0 else ""
        list_start = i == 0 or not prev or prev.endswith(":") or prev.startswith(("\\section", "\\subsection",
                                                                                   "\\subsubsection", "\\paragraph"))
        # bullets ("- x") are converted anywhere outside tables/math (masked above); numbered lines only at a
        # paragraph/list start, because "42. That year ..." in running prose is not a list
        if m and (list_start or not m.group(1)[0].isdigit()) and not lines[i].lstrip().startswith("\\item"):
            j = i
            items = []
            numbered = m.group(1)[0].isdigit()
            nums = []
            while j < len(lines) and j < len(masked_lines):
                mm = bullet.match(masked_lines[j])
                if not mm or mm.group(1)[0].isdigit() != numbered:
                    break
                if numbered:
                    nums.append(int(re.match(r"\d+", mm.group(1)).group(0)))
                items.append(bullet.match(lines[j]).group(2) if bullet.match(lines[j]) else lines[j])
                j += 1
            if numbered and nums != list(range(1, len(nums) + 1)):
                items = []  # "42. ..." in prose is not a list
            if len(items) >= 2 or (items and not numbered):
                env = "enumerate" if numbered else "itemize"
                out.append(f"\\begin{{{env}}}")
                out.extend(f"  \\item {it}" for it in items)
                out.append(f"\\end{{{env}}}")
                fixes.append(Fix("md_list", i + 1, f"Markdown list -> {env}"))
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), fixes


# ----------------------------------------------------------------------------- structural fixes

_EXT_FIG_ENV = re.compile(r"\\begin\{figure\*?\}(?:(?!\\end\{figure).)*?\\includegraphics(?:(?!\\end\{figure).)*?\\end\{figure\*?\}", re.S)
_EXT_IMG = re.compile(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}")


def strip_external_images(src: str) -> tuple[str, list[Fix]]:
    fixes = []
    spans = _protected_spans(src)

    def drop(m: re.Match) -> str:
        if _in_spans(m.start(), spans):
            return m.group(0)
        fixes.append(Fix("external_image", _line_at(src, m.start()), "removed figure referencing a non-existent image"))
        return ""

    src = _EXT_FIG_ENV.sub(drop, src)
    spans = _protected_spans(src)
    src = _EXT_IMG.sub(lambda m: m.group(0) if _in_spans(m.start(), spans) else (
        fixes.append(Fix("external_image", _line_at(src, m.start()), "removed \\includegraphics of a missing file")) or ""), src)
    return src, fixes


def wrap_orphan_items(src: str, lint: LintResult) -> tuple[str, list[Fix]]:
    lines_with_orphans = sorted({f.line for f in lint.findings if f.code == "item_outside_list"})
    if not lines_with_orphans:
        return src, []
    lines = src.split("\n")
    groups: list[list[int]] = []
    for ln in lines_with_orphans:
        if groups and ln - groups[-1][-1] <= 2:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    fixes = []
    for g in reversed(groups):
        start, end = g[0] - 1, g[-1] - 1
        # extend the group to the continuation lines of the last item
        while end + 1 < len(lines) and lines[end + 1].strip() and not lines[end + 1].lstrip().startswith("\\"):
            end += 1
        lines.insert(end + 1, "\\end{itemize}")
        lines.insert(start, "\\begin{itemize}")
        fixes.append(Fix("item_outside_list", g[0], "wrapped orphan \\item lines in itemize"))
    return "\n".join(lines), fixes


_MATHY = re.compile(r"^(?:\\?[A-Za-z]{1,2}|\\[a-zA-Z]+|\d+)(?:[_^](?:\{[^{}\s]{1,12}\}|[A-Za-z0-9]{1,3}|\\[a-zA-Z]+))+$")


def _script_edit(src: str, pos: int) -> tuple[int, int, str]:
    """`x_i` in text -> `$x_i$`; `learning_rate` -> `learning\\_rate`."""
    s = pos
    while s > 0 and (src[s - 1].isalnum() or src[s - 1] in "\\_^"):
        s -= 1
    e = pos + 1
    depth = 0
    while e < len(src):
        ch = src[e]
        if ch == "{":
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
        elif not (ch.isalnum() or ch in "_^\\" or (depth > 0 and ch not in " \n$}")):
            break
        e += 1
    token = src[s:e]
    if _MATHY.match(token):
        return (s, e, f"${token}$")
    return (pos, pos + 1, "\\_" if src[pos] == "_" else "\\textasciicircum{}")


def autofix(src: str, *, language: str = "en", section_title: str | None = None, mode: str = "section",
            known_labels: set[str] | None = None, max_rounds: int = 4) -> AutofixResult:
    fixes: list[Fix] = []
    src = src.replace("\r\n", "\n").replace("\t", "    ")
    src, f = fix_markdown(src)
    fixes += f
    src, f = fix_pseudo_env(src)
    fixes += f
    src, f = fix_percent(src)
    fixes += f
    src, f = strip_external_images(src)
    fixes += f
    if mode == "section" and section_title and not re.match(r"\s*(%[^\n]*\n\s*)*\\section\*?\{", src):
        src = f"\\section{{{section_title}}}\n" + src.lstrip("\n")
        fixes.append(Fix("no_section", 1, "added missing \\section header"))

    lint = lint_masked(src, language=language, mode=mode, known_labels=known_labels)
    for _ in range(max_rounds):
        if any(fd.code == "item_outside_list" for fd in lint.findings):
            src, f = wrap_orphan_items(src, lint)
            if f:
                fixes += f
                lint = lint_masked(src, language=language, mode=mode, known_labels=known_labels)
                continue
        edits = []
        applied: list[Finding] = []
        for fd in lint.findings:
            if fd.replacement is None:
                continue
            if fd.code == "script_outside_math":
                edits.append(_script_edit(src, fd.pos))
            else:
                edits.append((fd.pos, fd.end, fd.replacement))
            applied.append(fd)
        if not edits:
            break
        src = apply_edits(src, edits)
        fixes += [Fix(fd.code, fd.line, fd.message) for fd in applied]
        lint = lint_masked(src, language=language, mode=mode, known_labels=known_labels)
    return AutofixResult(src, fixes, lint)


def lint_masked(src: str, **kw) -> LintResult:
    """Lint with figure blocks blanked out (same length and line structure)."""
    blocks, _ = parse_blocks(src)
    if not blocks:
        return lint_latex(src, **kw)
    masked = list(src)
    for b in blocks:
        for k in range(b.start, b.end):
            if masked[k] != "\n":
                masked[k] = " "
    return lint_latex("".join(masked), **kw)
