"""Parser for TeX engine logs (pdfTeX/XeTeX/LuaTeX, TeX Live and MiKTeX).

Turns a raw ``.log`` into structured messages with file/line attribution so the
harness can (1) map an error back to the section that caused it and (2) feed
the model a short, precise description instead of a wall of log text.
Assumes compilation with ``-file-line-error``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FILE_LINE_RE = re.compile(r"^(?P<file>(?:[A-Za-z]:)?[^:\n]*?\.(?:tex|sty|cls|ltx|aux|toc|out|def|cfg)):(?P<line>\d+): (?P<msg>.*)$")
BANG_RE = re.compile(r"^! (?P<msg>.*)$")
CONTEXT_RE = re.compile(r"^l\.(?P<line>\d+) ?(?P<text>.*)$")
WARN_RE = re.compile(r"^(?P<who>LaTeX|Package [\w.-]+|Class [\w.-]+|LaTeX Font|pdfTeX) [Ww]arning: (?P<msg>.*)$")
BADBOX_RE = re.compile(
    r"^(?P<kind>Overfull|Underfull) \\(?P<box>[hv])box \((?P<amount>[^)]*)\) "
    r"(?:in paragraph|in alignment|detected|has occurred while \\output is active)"
    r"(?: at lines? (?P<l1>\d+)(?:--(?P<l2>\d+))?)?"
)
OUTPUT_RE = re.compile(r"Output written on (?P<file>.+?) \((?P<pages>\d+) pages?")
MISSING_CHAR_RE = re.compile(r"Missing character: There is no (?P<char>.+?) (?:\(U\+[0-9A-F]+\) )?in font (?P<font>[^!]+)!")
ON_LINE_RE = re.compile(r"on input line (\d+)")
CS_RE = re.compile(r"\\[A-Za-z@]+\*?|\\.")
FILE_OPEN_RE = re.compile(r"\((?P<path>(?:[A-Za-z]:)?[./\\]?[^\s()]*?\.(?:tex|sty|cls|cfg|def|ldf|fd|aux|toc|clo))")

_CATEGORIES = [
    ("undefined_cs", re.compile(r"Undefined control sequence")),
    ("missing_dollar", re.compile(r"Missing \$ inserted|Display math should end with \$\$|Bad math environment delimiter")),
    ("env_mismatch", re.compile(r"\\begin\{[^}]*\} (?:on input line \d+ )?ended by \\end|Environment .* undefined|\\begin\{document\} ended by")),
    ("missing_brace", re.compile(r"Missing \} inserted|Missing \{ inserted|Extra \}, or forgotten|Too many \}'s|Extra alignment tab")),
    ("runaway", re.compile(r"Runaway argument|File ended while scanning|Paragraph ended before")),
    ("missing_file", re.compile(r"File `[^']*' not found|Unable to load picture|cannot find image file")),
    ("unicode", re.compile(r"Unicode character|Invalid UTF-8|Missing character")),
    ("tikz", re.compile(r"Package (?:tikz|pgf|pgfkeys|pgfplots) Error|I do not know the key|Giving up on this path")),
    ("misplaced", re.compile(r"Misplaced alignment tab|Misplaced \\noalign|Misplaced \\cr|Misplaced \\omit")),
    ("math_only", re.compile(r"allowed only in math mode|Command \\\w+ invalid in math mode")),
    ("no_line", re.compile(r"There's no line here to end")),
    ("capacity", re.compile(r"TeX capacity exceeded|Emergency stop")),
    ("illegal_param", re.compile(r"Illegal parameter number|You can't use `macro parameter character")),
    ("double_script", re.compile(r"Double (?:sub|super)script")),
]


def categorize(msg: str) -> str:
    for name, rx in _CATEGORIES:
        if rx.search(msg):
            return name
    return "other"


@dataclass
class TexMessage:
    kind: str  # "error" | "warning" | "badbox"
    message: str
    file: str | None = None
    line: int | None = None
    context: str = ""
    culprit: str | None = None
    category: str = "other"
    amount_pt: float | None = None

    def short(self) -> str:
        loc = f"{self.file}:{self.line}: " if self.file and self.line else (f"line {self.line}: " if self.line else "")
        ctx = f"  [near: {self.context.strip()[:120]}]" if self.context.strip() else ""
        cul = f" `{self.culprit}`" if self.culprit and self.category == "undefined_cs" else ""
        return f"{loc}{self.message}{cul}{ctx}"


@dataclass
class LogSummary:
    errors: list[TexMessage] = field(default_factory=list)
    warnings: list[TexMessage] = field(default_factory=list)
    badboxes: list[TexMessage] = field(default_factory=list)
    pages: int | None = None
    missing_chars: list[str] = field(default_factory=list)
    undefined_refs: list[str] = field(default_factory=list)
    multiply_defined_labels: list[str] = field(default_factory=list)
    needs_rerun: bool = False
    fatal: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors and not self.fatal

    def overfull(self, min_pt: float = 5.0) -> list[TexMessage]:
        return [b for b in self.badboxes if b.message.startswith("Overfull") and (b.amount_pt or 0) >= min_pt]


def _normalize_file(path: str | None) -> str | None:
    if not path:
        return None
    p = path.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def _culprit_from_context(text: str) -> str | None:
    toks = CS_RE.findall(text)
    return toks[-1] if toks else None


def parse_log(log: str, max_errors: int = 20) -> LogSummary:
    s = LogSummary()
    lines = log.splitlines()
    n = len(lines)
    file_stack: list[str] = []
    i = 0
    while i < n:
        line = lines[i]

        # ---- approximate file stack, used for attributing badboxes
        for m in FILE_OPEN_RE.finditer(line):
            file_stack.append(_normalize_file(m.group("path")) or "")
        closes = line.count(")") - line.count("(")
        if closes > 0 and file_stack and not line.startswith(("Overfull", "Underfull")):
            for _ in range(min(closes, len(file_stack))):
                file_stack.pop()

        m = FILE_LINE_RE.match(line)
        b = None if m else BANG_RE.match(line)
        if m or b:
            msg = (m or b).group("msg").strip()
            err = TexMessage(kind="error", message=msg,
                             file=_normalize_file(m.group("file")) if m else (file_stack[-1] if file_stack else None),
                             line=int(m.group("line")) if m else None)
            # Gather continuation lines up to the "l.<n>" context line.
            j = i + 1
            extra: list[str] = []
            while j < n and j < i + 14:
                cm = CONTEXT_RE.match(lines[j])
                if cm:
                    err.line = err.line or int(cm.group("line"))
                    after = lines[j + 1].strip() if j + 1 < n else ""
                    err.context = (cm.group("text") + " " + after).strip()
                    err.culprit = _culprit_from_context(cm.group("text"))
                    j += 1
                    break
                if FILE_LINE_RE.match(lines[j]) or BANG_RE.match(lines[j]):
                    break
                if lines[j].strip() and not lines[j].startswith(("See the", "Type  H", "You're in trouble", "Try typing")):
                    extra.append(lines[j].strip())
                j += 1
            if msg.startswith("LaTeX Error") or msg.startswith("Package") and extra:
                # first continuation line often completes the message
                if extra and not extra[0].startswith(("See ", "...", "\\")):
                    err.message = f"{msg} {extra[0]}".strip()
            err.category = categorize(err.message)
            if err.category == "undefined_cs" and not err.culprit and extra:
                err.culprit = _culprit_from_context(extra[-1])
            if "Emergency stop" in msg or "Fatal error" in msg:
                s.fatal = True
                if s.errors:  # emergency stop is a consequence of a previous error
                    i = j
                    continue
            if len(s.errors) < max_errors:
                s.errors.append(err)
            i = j
            continue

        wm = WARN_RE.match(line)
        if wm:
            msg = wm.group("msg").strip()
            j = i + 1
            # package warnings continue on lines starting with "(pkgname)"
            while j < n and lines[j].strip() and (lines[j].startswith("(") or lines[j].startswith(" ")) and j < i + 6:
                msg += " " + lines[j].strip().lstrip("(").split(")", 1)[-1].strip()
                j += 1
            lm = ON_LINE_RE.search(msg)
            w = TexMessage(kind="warning", message=f"{wm.group('who')} Warning: {msg}",
                           line=int(lm.group(1)) if lm else None,
                           file=file_stack[-1] if file_stack else None)
            s.warnings.append(w)
            if "Rerun to get" in msg or "Label(s) may have changed" in msg or "Rerun LaTeX" in msg:
                s.needs_rerun = True
            rm = re.search(r"Reference `([^']+)' on page", msg)
            if rm:
                s.undefined_refs.append(rm.group(1))
            lm2 = re.search(r"Label `([^']+)' multiply defined", msg)
            if lm2:
                s.multiply_defined_labels.append(lm2.group(1))
            i = j
            continue

        bm = BADBOX_RE.match(line)
        if bm:
            amount = None
            am = re.match(r"([\d.]+)pt too (?:wide|high)", bm.group("amount"))
            if am:
                amount = float(am.group(1))
            s.badboxes.append(TexMessage(
                kind="badbox", message=line.strip(), file=file_stack[-1] if file_stack else None,
                line=int(bm.group("l1")) if bm.group("l1") else None, amount_pt=amount,
                category="overfull" if bm.group("kind") == "Overfull" else "underfull",
            ))
            i += 1
            continue

        om = OUTPUT_RE.search(line)
        if om:
            s.pages = int(om.group("pages"))
        mc = MISSING_CHAR_RE.search(line)
        if mc:
            s.missing_chars.append(mc.group("char"))
        if "No pages of output" in line:
            s.pages = 0
        i += 1

    # de-duplicate cascaded errors (same file/line/category)
    seen = set()
    uniq = []
    for e in s.errors:
        key = (e.file, e.line, e.category, e.message[:40])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    s.errors = uniq
    return s
