"""Static LaTeX linter tuned for the mistakes small LMs make.

A single-pass scanner tracks braces, environments, math mode (including
``\\text{...}`` islands inside math), verbatim regions and raw command
arguments. Every finding carries a character span and, when the correction is
unambiguous, a replacement -- ``autofix`` applies those mechanically, so only
genuinely ambiguous problems reach the (expensive) model repair loop.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ..schemas import Issue

BOX_ENVS = {"definition", "theorem", "lemma", "proposition", "corollary", "example", "remark", "keyidea",
            "warning", "summary", "exercise", "solution", "intuition", "note", "lnoverview", "csbox"}
MATH_ENVS = {"equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
             "flalign", "flalign*", "alignat", "alignat*", "eqnarray", "eqnarray*", "displaymath", "math"}
INNER_MATH_ENVS = {"cases", "dcases", "rcases", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
                   "smallmatrix", "aligned", "alignedat", "gathered", "split", "array", "subarray", "multlined"}
ALIGN_ENVS = {"tabular", "tabular*", "tabularx", "array", "align", "align*", "flalign", "flalign*", "alignat",
              "alignat*", "eqnarray", "eqnarray*", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix",
              "Vmatrix", "smallmatrix", "cases", "dcases", "rcases", "aligned", "alignedat", "split",
              "tikzpicture", "longtable", "subarray"}
LIST_ENVS = {"itemize", "enumerate", "description"}
VERBATIM_ENVS = {"lstlisting", "verbatim", "verbatim*", "Verbatim", "comment"}
TEXT_ENVS = {"itemize", "enumerate", "description", "center", "flushleft", "flushright", "quote", "quotation",
             "verse", "tabular", "tabular*", "table", "table*", "figure", "figure*", "minipage", "proof",
             "tikzpicture", "scope", "axis", "semilogyaxis", "semilogxaxis", "loglogaxis", "groupplot",
             "multicols", "multicols*", "small", "footnotesize", "scriptsize", "tiny", "large", "Large",
             "subequations", "tcolorbox", "abstract"}
ALLOWED_ENVS = BOX_ENVS | MATH_ENVS | INNER_MATH_ENVS | LIST_ENVS | VERBATIM_ENVS | TEXT_ENVS

ENV_ALIASES = {
    "thm": "theorem", "Theorem": "theorem", "theo": "theorem", "defn": "definition", "defi": "definition",
    "Definition": "definition", "def": "definition", "dfn": "definition", "lem": "lemma", "Lemma": "lemma",
    "prop": "proposition", "Proposition": "proposition", "cor": "corollary", "Corollary": "corollary",
    "ex": "example", "exmp": "example", "Example": "example", "examples": "example", "rem": "remark",
    "Remark": "remark", "rmk": "remark", "key": "keyidea", "keypoint": "keyidea", "idea": "keyidea",
    "insight": "keyidea", "caution": "warning", "pitfall": "warning", "attention": "warning",
    "recap": "summary", "takeaways": "summary", "exercises": "exercise", "problem": "exercise",
    "sol": "solution", "answer": "solution", "observation": "remark", "notation": "remark",
    "claim": "proposition", "fact": "proposition", "conjecture": "proposition", "hypothesis": "proposition",
    "algorithm": "lstlisting", "algorithmic": "lstlisting", "code": "lstlisting", "minted": "lstlisting",
    "important": "keyidea", "highlight": "keyidea", "box": "keyidea", "tip": "remark", "hint": "remark",
}

FORBIDDEN_CMDS = {
    "documentclass": "the preamble is provided by the system",
    "usepackage": "packages are provided by the system", "RequirePackage": "packages are provided by the system",
    "input": "file inclusion is not allowed", "include": "file inclusion is not allowed",
    "includeonly": "file inclusion is not allowed", "write": "file writes are not allowed",
    "immediate": "file writes are not allowed", "openout": "file writes are not allowed",
    "openin": "file reads are not allowed", "newwrite": "file writes are not allowed",
    "newread": "file reads are not allowed", "read": "file reads are not allowed",
    "catcode": "catcode changes are not allowed", "makeatletter": "internal macros are not allowed",
    "makeatother": "internal macros are not allowed", "directlua": "Lua is not allowed",
    "ShellEscape": "shell escape is not allowed", "special": "\\special is not allowed",
    "maketitle": "the title is produced by the system", "title": "the title is produced by the system",
    "author": "the title is produced by the system", "date": "the title is produced by the system",
    "tableofcontents": "the table of contents is produced by the system",
    "bibliography": "no bibliography is available", "bibliographystyle": "no bibliography is available",
    "printbibliography": "no bibliography is available", "addbibresource": "no bibliography is available",
    "pagestyle": "page style is set by the system",
    # anything that reads files or builds control sequences by name (would bypass this list)
    "lstinputlisting": "file inclusion is not allowed", "verbatiminput": "file inclusion is not allowed",
    "VerbatimInput": "file inclusion is not allowed", "includepdf": "file inclusion is not allowed",
    "csname": "\\csname is not allowed", "pdffiledump": "file reads are not allowed",
    "pdfximage": "file reads are not allowed", "pdfobj": "low-level PDF commands are not allowed",
    "ReadFile": "file reads are not allowed", "lowercase": "not allowed", "uppercase": "not allowed",
    "scantokens": "not allowed", "endinput": "not allowed", "openany": "not allowed",
}
_FORBIDDEN_WITH_ARG = {"usepackage", "documentclass", "RequirePackage", "title", "author", "date", "bibliography",
                       "bibliographystyle", "addbibresource", "pagestyle", "input", "include", "includeonly"}
# commands whose first {argument} is a raw string (labels, urls, file names, colors)
RAW_ARG_CMDS = {"label", "ref", "eqref", "pageref", "autoref", "nameref", "cref", "Cref", "url", "href",
                "includegraphics", "hyperref", "definecolor", "usetikzlibrary", "color",
                "textcolor", "colorbox", "pgfmathsetmacro", "lnfigure", "lnmissingfigure"}
REF_CMDS = {"ref", "eqref", "pageref", "autoref", "nameref", "cref", "Cref"}
CITE_CMDS = {"cite", "citep", "citet", "citeauthor", "nocite"}
TEXT_IN_MATH = {"text", "textrm", "textit", "textbf", "textsf", "texttt", "mbox", "intertext",
                "shortintertext", "textnormal"}
DEF_CMDS = {"newcommand", "renewcommand", "providecommand", "DeclareMathOperator", "def", "newenvironment",
            "renewenvironment"}

# Unicode that pdflatex + utf8 inputenc + T1 typesets in text mode without extra packages.
TEXT_SAFE_PUNCT = set("–—‘’‚“”„†‡•…‰‹›«»€™©®°§¶·×÷¡¿")
UNICODE_MATH = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon", "ϵ": r"\epsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ϑ": r"\vartheta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma",
    "ς": r"\varsigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi", "ϕ": r"\phi", "χ": r"\chi",
    "ψ": r"\psi", "ω": r"\omega", "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "≡": r"\equiv", "≅": r"\cong", "∼": r"\sim",
    "≃": r"\simeq", "∝": r"\propto", "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "⟹": r"\implies", "⟺": r"\iff",
    "↦": r"\mapsto", "↑": r"\uparrow", "↓": r"\downarrow", "∞": r"\infty", "∑": r"\sum", "∏": r"\prod",
    "∫": r"\int", "∮": r"\oint", "∂": r"\partial", "∇": r"\nabla", "√": r"\surd", "∈": r"\in",
    "∉": r"\notin", "∋": r"\ni", "⊂": r"\subset", "⊃": r"\supset", "⊆": r"\subseteq", "⊇": r"\supseteq",
    "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset", "∀": r"\forall", "∃": r"\exists", "∄": r"\nexists",
    "¬": r"\neg", "∧": r"\wedge", "∨": r"\vee", "⊕": r"\oplus", "⊗": r"\otimes", "∘": r"\circ",
    "⋅": r"\cdot", "∙": r"\cdot", "±": r"\pm", "∓": r"\mp", "⊥": r"\perp", "∥": r"\parallel",
    "‖": r"\|", "⟨": r"\langle", "⟩": r"\rangle", "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "ℓ": r"\ell", "ℏ": r"\hbar",
    "ħ": r"\hbar", "∗": r"*", "−": r"-", "′": r"'", "″": r"''", "²": r"^{2}", "³": r"^{3}", "¹": r"^{1}",
    "⁻": r"^{-}", "⁺": r"^{+}", "ⁿ": r"^{n}", "₀": r"_{0}", "₁": r"_{1}", "₂": r"_{2}", "₃": r"_{3}",
    "₄": r"_{4}", "₅": r"_{5}", "₆": r"_{6}", "₇": r"_{7}", "₈": r"_{8}", "₉": r"_{9}", "ₙ": r"_{n}",
    "ᵢ": r"_{i}", "ⱼ": r"_{j}", "ₖ": r"_{k}", "×": r"\times", "÷": r"\div", "°": r"^{\circ}",
    "⌊": r"\lfloor", "⌋": r"\rfloor", "⌈": r"\lceil", "⌉": r"\rceil", "…": r"\dots", "⋯": r"\cdots",
    "⋮": r"\vdots", "⋱": r"\ddots", "∖": r"\setminus", "△": r"\triangle", "□": r"\square", "★": r"\star",
    "⊤": r"\top", "⊢": r"\vdash", "⊨": r"\models", "≪": r"\ll", "≫": r"\gg", "≺": r"\prec", "≻": r"\succ",
    "∠": r"\angle", "⊊": r"\subsetneq", "∣": r"\mid", "ℵ": r"\aleph",
}
UNICODE_TEXT = {"\u00a0": "~", "\u2009": r"\,", "\u202f": r"\,", "\u200b": "", "\u200c": "", "\u200d": "",
                "\ufeff": "", "\u2212": "-", "✓": r"$\checkmark$", "✔": r"$\checkmark$", "✗": r"$\times$",
                "✘": r"$\times$", "→": r"$\to$", "←": r"$\leftarrow$", "⇒": r"$\Rightarrow$",
                "≈": r"$\approx$", "≤": r"$\leq$", "≥": r"$\geq$", "≠": r"$\neq$", "∞": r"$\infty$",
                "±": r"$\pm$", "′": "'", "″": "''", "‐": "-", "‑": "-", "‒": "--", "⁄": "/"}

_CLOSE = {"$": "$", "$$": "$$", "\\(": "\\)", "\\[": "\\]"}
_BOX_BOUNDARY = re.compile(r"^[ \t]*(?:\\begin\{(?:" + "|".join(sorted(BOX_ENVS)) +
                           r")\}|\\(?:sub)*section\*?\{|<figure\b)", re.M)


@dataclass
class Finding:
    code: str
    message: str
    line: int
    pos: int
    end: int
    severity: str = "error"
    replacement: str | None = None  # None -> not mechanically fixable
    hint: str | None = None

    def to_issue(self) -> Issue:
        return Issue(source="lint", severity=self.severity, code=self.code, message=self.message,
                     line=self.line, hint=self.hint)


@dataclass
class LintResult:
    findings: list[Finding] = field(default_factory=list)
    envs: Counter = field(default_factory=Counter)
    labels: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    open_envs: list[tuple[str, int]] = field(default_factory=list)  # unclosed at EOF
    open_math: tuple[str, int] | None = None
    first_command: str | None = None

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def issues(self) -> list[Issue]:
        return [f.to_issue() for f in self.findings]

    @property
    def fixable(self) -> list[Finding]:
        return [f for f in self.findings if f.replacement is not None]


def _read_group(src: str, i: int, open_c: str = "{", close_c: str = "}") -> int:
    """Given src[i] == open_c, return the index just after the matching close."""
    depth = 0
    n = len(src)
    j = i
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
        elif c == "\n" and j + 1 < n and src[j + 1] == "\n" and depth > 0:
            return j  # an argument never spans a paragraph; stop to avoid runaway
        j += 1
    return n


def _skip_ws(src: str, i: int) -> int:
    while i < len(src) and src[i] in " \t":
        i += 1
    return i


def lint_latex(src: str, *, allowed_envs: set[str] | None = None, language: str = "en",
               mode: str = "section", known_labels: set[str] | None = None) -> LintResult:
    allowed = allowed_envs if allowed_envs is not None else ALLOWED_ENVS
    res = LintResult()
    F = res.findings
    n = len(src)
    i = 0
    line = 1
    env_stack: list[tuple[str, int, int]] = []  # (name, line, pos)
    brace_stack: list[tuple[int, int]] = []  # (line, pos)
    math: tuple[str, int, int] | None = None  # (delim, line, pos) for $, $$, \(, \[
    text_frames: list[tuple[int, tuple | None]] = []  # \text{...} inside math: (brace depth, saved math)
    allow_cyrillic = language in ("ru", "uk", "bg", "sr")

    def in_env(names: set[str]) -> bool:
        return any(e[0] in names for e in env_stack)

    def env_math() -> bool:
        return in_env(MATH_ENVS) and not text_frames

    def math_mode() -> bool:
        return math is not None or env_math()

    def add(code, msg, pos, end=None, sev="error", repl=None, hint=None, ln=None):
        F.append(Finding(code, msg, ln or line, pos, end if end is not None else pos + 1, sev, repl, hint))

    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            j = _skip_ws(src, i + 1)
            if j < n and src[j] == "\n":  # blank line = paragraph break
                if math is not None and math[0] in ("$", "\\("):
                    close = _CLOSE[math[0]]
                    add("unclosed_inline_math", f"inline math opened on line {math[1]} is not closed before the paragraph ends",
                        i, i, repl=close, ln=math[1], hint=f"close it with {close}")
                    math = None
                elif math is not None:
                    add("blank_line_in_math", f"blank line inside display math opened on line {math[1]}", i, j, repl="")
                elif env_math():
                    env = next(e for e in reversed(env_stack) if e[0] in MATH_ENVS)
                    add("blank_line_in_math", f"blank line inside {env[0]} environment (line {env[1]})", i, j, repl="")
            i += 1
            continue
        if c == "%":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "\\":
            if i + 1 >= n:
                i += 1
                continue
            nxt = src[i + 1]
            if not nxt.isalpha():
                if nxt in "([":
                    if math_mode():
                        add("nested_math", f"\\{nxt} opened while already in math mode", i, i + 2)
                    else:
                        math = ("\\" + nxt, line, i)
                elif nxt in ")]":
                    want = "\\(" if nxt == ")" else "\\["
                    if math is None or math[0] != want:
                        add("unmatched_math_close", f"\\{nxt} without matching {want}", i, i + 2,
                            repl="" if math is None and not env_math() else None)
                    else:
                        math = None
                i += 2
                continue
            j = i + 1
            while j < n and src[j].isalpha():
                j += 1
            name = src[i + 1 : j]
            star = j < n and src[j] == "*"
            after_name = j + (1 if star else 0)
            if res.first_command is None and name != "label":
                res.first_command = name

            # ---------------------------------------------------------- environments
            if name in ("begin", "end"):
                k = _skip_ws(src, j)
                close = src.find("}", k) if k < n and src[k] == "{" else -1
                if close < 0:
                    add("bad_env", f"malformed \\{name}", i, j)
                    i = j
                    continue
                env = src[k + 1 : close].strip()
                after = close + 1
                if env == "document":
                    add("forbidden", f"\\{name}{{document}} is provided by the system", i, after, repl="")
                    i = after
                    continue
                real = env if env in allowed else ENV_ALIASES.get(env, env)
                if env != real:
                    add("env_alias", f"environment '{env}' is not defined; use '{real}'", k + 1, close, repl=real)
                if name == "begin":
                    res.envs[real] += 1
                    if real not in allowed:
                        add("unknown_env", f"environment '{env}' is not defined in this template", i, after,
                            hint="use definition/theorem/example/remark/keyidea/warning/summary, itemize, align, ...")
                    if real in VERBATIM_ENVS:
                        end_tag = f"\\end{{{env}}}"
                        e = src.find(end_tag, after)
                        body_end = e if e >= 0 else n
                        if any(ord(ch) > 127 for ch in src[after:body_end]):
                            add("unicode_in_verbatim", f"non-ASCII characters inside {env} break pdflatex",
                                after, body_end, sev="warning", hint="use ASCII only inside code listings")
                        start_line = line
                        line += src.count("\n", i, body_end)
                        if e < 0:
                            F.append(Finding("unclosed_env", f"\\begin{{{env}}} on line {start_line} is never closed",
                                             start_line, n, n, "error", f"\n\\end{{{env}}}"))
                            i = n
                        else:
                            if env != real:  # also rename the matching \end of an aliased verbatim env
                                add("env_alias", f"environment '{env}' is not defined; use '{real}'",
                                    e + 5, e + 5 + len(env), repl=real)
                            i = e + len(end_tag)
                        continue
                    if real in MATH_ENVS and math_mode():
                        add("nested_math", f"display math environment {env} inside math mode", i, after)
                    if real in INNER_MATH_ENVS and real != "array" and not math_mode():
                        add("math_env_outside_math", f"{env} must be used inside math mode", i, after,
                            hint=f"wrap it: \\[ \\begin{{{env}}} ... \\end{{{env}}} \\]")
                    env_stack.append((real, line, i))
                else:
                    if real in MATH_ENVS and math is not None:
                        add("unclosed_math", f"math opened with {math[0]} on line {math[1]} is not closed before \\end{{{env}}}",
                            i, i, repl=_CLOSE[math[0]])
                        math = None
                    if env_stack and env_stack[-1][0] == real:
                        env_stack.pop()
                    elif any(e[0] == real for e in env_stack):
                        while env_stack and env_stack[-1][0] != real:
                            o = env_stack.pop()
                            add("env_mismatch", f"\\begin{{{o[0]}}} (line {o[1]}) is not closed before \\end{{{env}}}",
                                i, i, repl=f"\\end{{{o[0]}}}")
                        env_stack.pop()
                    else:
                        add("unmatched_end", f"\\end{{{env}}} without matching \\begin{{{env}}}", i, after, repl="")
                i = after
                continue

            # ---------------------------------------------------------- forbidden commands
            if name in FORBIDDEN_CMDS:
                end = after_name
                if name in _FORBIDDEN_WITH_ARG:
                    k = _skip_ws(src, end)
                    if k < n and src[k] == "[":
                        k = _skip_ws(src, _read_group(src, k, "[", "]"))
                    if k < n and src[k] == "{":
                        end = _read_group(src, k)
                add("forbidden", f"\\{name} is not allowed here: {FORBIDDEN_CMDS[name]}", i, end, repl="")
                line += src.count("\n", i, end)
                i = end
                continue

            # ---------------------------------------------------------- macro definitions (body may contain #)
            if name in DEF_CMDS:
                k = _skip_ws(src, after_name)
                if k < n and src[k] == "{":
                    k = _read_group(src, k)
                elif k < n and src[k] == "\\":
                    k += 1
                    while k < n and src[k].isalpha():
                        k += 1
                for _ in range(2):
                    k = _skip_ws(src, k)
                    if k < n and src[k] == "[":
                        k = _read_group(src, k, "[", "]")
                for _ in range(2 if name.endswith("environment") else 1):
                    k = _skip_ws(src, k)
                    if k < n and src[k] == "{":
                        k = _read_group(src, k)
                bad = [c for c in re.findall(r"\\([A-Za-z]+)", src[after_name:k]) if c in FORBIDDEN_CMDS]
                if bad:  # a macro body must not smuggle in forbidden commands
                    add("forbidden", f"macro definition uses forbidden \\{bad[0]}", i, k, repl="")
                line += src.count("\n", i, k)
                i = k
                continue

            # ---------------------------------------------------------- inline verbatim
            if name in ("verb", "lstinline"):
                k = after_name
                if name == "lstinline" and k < n and src[k] == "[":
                    k = _read_group(src, k, "[", "]")
                if k < n:
                    d = src[k]
                    if d == "{":
                        e = _read_group(src, k)
                    else:
                        e = src.find(d, k + 1)
                        e = (e + 1) if e >= 0 else n
                    i = e
                    continue

            # ---------------------------------------------------------- text islands inside math
            if name in TEXT_IN_MATH and math_mode():
                k = _skip_ws(src, after_name)
                if k < n and src[k] == "{":
                    brace_stack.append((line, k))
                    text_frames.append((len(brace_stack), math))
                    math = None
                    i = k + 1
                    continue

            if name == "item" and not in_env(LIST_ENVS):
                add("item_outside_list", "\\item used outside itemize/enumerate", i, j,
                    hint="wrap the items in \\begin{itemize} ... \\end{itemize}")
            if name == "includegraphics":
                add("external_image", "\\includegraphics refers to an image file that does not exist; "
                    "use a <figure> block with code instead", i, j,
                    hint='replace with <figure kind="matplotlib"> or kind="tikz"')
            if name in CITE_CMDS:
                k = _skip_ws(src, after_name)
                if k < n and src[k] == "[":
                    k = _read_group(src, k, "[", "]")
                if k < n and src[k] == "{":
                    e = _read_group(src, k)
                    sp = i - 1 if i > 0 and src[i - 1] in " ~" else i
                    add("citation", "citations are not supported (no bibliography)", sp, e, sev="warning", repl="")
                    i = e
                    continue

            # ---------------------------------------------------------- raw first argument
            if name in RAW_ARG_CMDS or name.endswith("ref"):
                k = _skip_ws(src, after_name)
                if k < n and src[k] == "[":
                    k = _skip_ws(src, _read_group(src, k, "[", "]"))
                if k < n and src[k] == "{":
                    e = _read_group(src, k)
                    arg = src[k + 1 : e - 1].strip()
                    if name == "label":
                        if arg in res.labels:
                            add("duplicate_label", f"label '{arg}' is defined twice", k + 1, e - 1, sev="warning",
                                repl=f"{arg}-{res.labels.count(arg) + 1}")
                        res.labels.append(arg)
                    elif name in REF_CMDS:
                        res.refs.append(arg)
                    line += src.count("\n", k, e)
                    i = e
                    continue
            i = after_name
            continue

        if c == "{":
            brace_stack.append((line, i))
        elif c == "}":
            if text_frames and len(brace_stack) == text_frames[-1][0]:
                brace_stack.pop()
                math = text_frames.pop()[1]
            elif brace_stack:
                brace_stack.pop()
            else:
                add("extra_brace", "unmatched closing brace }", i, i + 1, repl="")
        elif c == "$":
            if i + 1 < n and src[i + 1] == "$":
                if math is None and not env_math():
                    math = ("$$", line, i)
                    add("double_dollar", "$$...$$ is deprecated; use \\[ ... \\]", i, i + 2, sev="warning", repl="\\[")
                elif math is not None and math[0] == "$$":
                    add("double_dollar", "$$...$$ is deprecated; use \\[ ... \\]", i, i + 2, sev="warning", repl="\\]")
                    math = None
                else:
                    add("bad_math_delim", "$$ inside math mode", i, i + 2)
                i += 2
                continue
            if math is None:
                if env_math():
                    add("dollar_in_math_env", "$ inside a display math environment", i, i + 1, repl="")
                else:
                    math = ("$", line, i)
            elif math[0] == "$":
                math = None
            else:
                add("bad_math_delim", f"$ inside {math[0]} math (opened line {math[1]})", i, i + 1, repl="")
        elif c in "_^" and not math_mode() and not in_env({"tikzpicture", "axis"}):
            repl = "\\_" if c == "_" else "\\textasciicircum{}"
            add("script_outside_math", f"'{c}' outside math mode", i, i + 1, repl=repl,
                hint="put formulas in $...$; write \\_ for a literal underscore")
        elif c == "&" and not in_env(ALIGN_ENVS):
            # in text a literal ampersand was meant; in math (e.g. equation) it is a misplaced alignment
            # tab and turning it into a visible "&" would silently change the formula -> leave to repair
            add("ampersand_outside_table", "'&' outside a table/alignment", i, i + 1,
                repl=None if math_mode() else "\\&",
                hint="use align instead of equation for aligned formulas" if math_mode() else None)
        elif c == "#" and not in_env({"tikzpicture", "axis"}):
            add("hash", "'#' must be written \\#", i, i + 1, repl="\\#")
        elif ord(c) > 127:
            if allow_cyrillic and ("\u0400" <= c <= "\u04ff"):
                pass
            elif math_mode():
                rep = UNICODE_MATH.get(c)
                add("unicode_math", f"Unicode character '{c}' in math", i, i + 1,
                    repl=(rep + (" " if rep and rep[-1].isalpha() else "")) if rep else None,
                    hint="use the LaTeX command")
            elif c in UNICODE_TEXT:
                add("unicode_text", f"Unicode character '{c}' (U+{ord(c):04X})", i, i + 1,
                    sev="warning", repl=UNICODE_TEXT[c])
            elif c in UNICODE_MATH and c not in TEXT_SAFE_PUNCT:
                add("unicode_text", f"Unicode math symbol '{c}' in text", i, i + 1, repl=f"${UNICODE_MATH[c]}$")
            elif "\u00c0" <= c <= "\u024f" or c in TEXT_SAFE_PUNCT:
                pass  # Latin accents and typographic punctuation are fine with T1 + utf8
            elif "\u0400" <= c <= "\u04ff":
                add("cyrillic", "Cyrillic text requires language='ru'", i, i + 1)
            else:
                add("unicode_unsupported", f"unsupported Unicode character '{c}' (U+{ord(c):04X})", i, i + 1,
                    repl="", sev="warning")
        i += 1

    # ---- end of input: close whatever is still open (innermost first)
    if math is not None:
        F.append(Finding("unclosed_math", f"math opened with {math[0]} on line {math[1]} is never closed",
                         math[1], n, n, "error", _CLOSE[math[0]]))
        res.open_math = (math[0], math[1])
    for name, ln, pos in reversed(env_stack):
        at = n
        if name in BOX_ENVS:
            # a box most likely ends where the next box / subsection / figure starts, not at EOF
            m = _BOX_BOUNDARY.search(src, src.find("}", pos) + 1)
            if m:
                at = m.start()
        repl = f"\\end{{{name}}}\n" if at < n else f"\n\\end{{{name}}}"
        F.append(Finding("unclosed_env", f"\\begin{{{name}}} on line {ln} is never closed", ln, at, at, "error", repl))
    res.open_envs = [(e[0], e[1]) for e in env_stack]
    for ln, pos in brace_stack:
        F.append(Finding("unclosed_brace", f"opening brace on line {ln} is never closed", ln, pos, pos + 1,
                         "error", None, "check the braces on that line"))

    if mode == "section" and res.first_command != "section":
        F.append(Finding("no_section", "the output must start with \\section{...}", 1, 0, 0, "warning"))

    known = set(known_labels or ()) | set(res.labels)
    for r in res.refs:
        if r not in known and not r.startswith("fig:"):
            F.append(Finding("undefined_ref", f"\\ref{{{r}}} points to an undefined label", 1, 0, 0, "warning"))

    _markdown_findings(src, F)
    return res


_MD_PATTERNS = [
    ("md_heading", re.compile(r"^(#{1,4})\s+\S.*$", re.M), "Markdown heading; use \\subsection{...}"),
    ("md_bold", re.compile(r"(?<![\\*])\*\*(?=\S)([^*\n]+?)(?<=\S)\*\*(?!\*)"), "Markdown bold; use \\textbf{...}"),
    ("md_fence", re.compile(r"^\s*```", re.M), "Markdown code fence; use \\begin{lstlisting}"),
    ("md_bullet", re.compile(r"^[ \t]*[-*]\s+\S", re.M), "Markdown bullet; use itemize"),
    ("md_link", re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)"), "Markdown link; use \\href{url}{text}"),
]
_MASK_RES = [
    re.compile(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", re.S),
    re.compile(r"<figure\b.*?</figure>", re.S | re.I),
    re.compile(r"\\begin\{(equation|align|gather|multline|flalign|alignat|eqnarray)(\*?)\}.*?\\end\{\1\2\}", re.S),
    re.compile(r"\\\[.*?\\\]", re.S),
]


def mask_for_markdown(src: str) -> str:
    """Blank out code, figure blocks and display math (keeping newlines) before markdown detection."""
    masked = src
    for rx in _MASK_RES:
        masked = rx.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), masked)
    return re.sub(r"(?<!\\)%.*$", lambda m: " " * len(m.group(0)), masked, flags=re.M)


def _markdown_findings(src: str, F: list[Finding]) -> None:
    masked = mask_for_markdown(src)
    for code, rx, msg in _MD_PATTERNS:
        for m in rx.finditer(masked):
            ln = masked.count("\n", 0, m.start()) + 1
            F.append(Finding(code, msg, ln, m.start(), m.end(), "error" if code == "md_fence" else "warning", None))


def summarize(res: LintResult, max_items: int = 12) -> str:
    errs = sorted(res.findings, key=lambda f: (f.severity != "error", f.line))
    return "\n".join(f.to_issue().render() for f in errs[:max_items])
