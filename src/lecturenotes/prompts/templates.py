"""Task prompts. They define the I/O contract of every model call, both at inference
time and in the SFT/RL data -- change them only together with PROMPT_VERSION."""

from __future__ import annotations

import json
import re

from .. import PROMPT_VERSION
from ..latex.capabilities import TexCapabilities
from ..schemas import FigurePlan, LectureRequest, Outline, SectionPlan
from ..utils import number_lines, truncate_middle

LANG_NAMES = {"en": "English", "ru": "Russian", "de": "German", "fr": "French", "es": "Spanish", "uk": "Ukrainian"}

SYSTEM = ("You are LectureScribe, an expert university lecturer and LaTeX typesetter. You write rigorous, "
          "clear and well-structured lecture notes: precise definitions, intuition before formalism, worked "
          "examples, and figures where a picture helps. You follow the requested output format exactly.")

STYLE_CARD = r"""LaTeX rules (the preamble already exists; never write \documentclass, \usepackage, \begin{document}, \title or \maketitle):
- Structure: \subsection{...} inside the section; paragraphs of real prose, not only bullet lists.
- Boxes (optional title in [ ]): definition, theorem, lemma, proposition, corollary, proof, example, remark,
  keyidea (core intuition, 1-3 sentences), warning (common pitfall), summary (end-of-section recap with itemize),
  exercise + solution.
- Open boxes with \begin{keyidea} ... \end{keyidea}; never write tags like [keyidea] or [definition] in the text.
- \term{word} marks a key term where it is defined.
- Be concrete: numbers, worked steps, small examples. No filler ("In summary, X is a powerful tool",
  "plays a crucial role"); end a section with a summary box at most once.
- Math: $...$ inline, \[ ... \] or align/equation for display. amsmath, amssymb, mathtools, bm are loaded.
  Macros: \R \N \Z \Q \C \E \Prob \Var \Cov \argmin \argmax \abs{x} \norm{x} \inner{x}{y} \dd \vect{x}.
- Tables: tabular with \toprule \midrule \bottomrule. Code: \begin{lstlisting}[language=Python] ... \end{lstlisting} (ASCII only).
- Escape special characters in text: \% \& \# \_ ; no Markdown, no Unicode math symbols (write \alpha, \leq, \to).
- No \cite, no \includegraphics, no external files.
Figures are generated from code. Never use \includegraphics; write a figure block instead:
<figure id="short-id" kind="matplotlib" caption="One-sentence caption.">
python code: numpy + matplotlib only, exactly one figure via fig, ax = plt.subplots(); label the axes; no savefig/show
</figure>
<figure id="short-id" kind="tikz" caption="One-sentence caption.">
\begin{tikzpicture} ... \end{tikzpicture}
</figure>
<figure id="short-id" kind="web" query="2-5 word image search query" caption="One-sentence caption."/>
Use matplotlib for plots of functions/data, TikZ for diagrams (graphs, trees, flowcharts, geometry), and kind="web"
for real-world photos or classic illustrations that cannot be drawn with code (an organism, an instrument, a historical
portrait, a microscope image); the system fetches a freely licensed image and adds the credit itself.
Good notes are visual: every section should contain at least one figure, placed right after the paragraph it explains.
A figure must show the concept of its caption (a decoding diagram, not a generic curve); in TikZ keep labels apart.
Refer to a figure as Figure~\ref{fig:short-id}. Captions must not contain double quotes."""

SLIDE_CARD = r"""To show an original slide image write <slide n="5" caption="..."/> (use sparingly, only for complex visuals)."""

PLAN_EXAMPLE = {
    "title": "Fourier Series",
    "subtitle": "Representing periodic functions with sines and cosines",
    "abstract": "Two or three sentences describing what the lecture covers and why it matters.",
    "prerequisites": ["integration by parts", "complex exponentials"],
    "learning_objectives": ["compute Fourier coefficients of a periodic function", "..."],
    "sections": [
        {"title": "Periodic functions and orthogonality",
         "goals": ["motivate trigonometric expansions"],
         "key_points": ["period", "orthogonality of sin and cos", "inner product on [-pi, pi]"],
         "slides": [1, 2, 3],
         "figures": [{"id": "sine-basis", "kind": "matplotlib", "description": "first three sine basis functions"},
                     {"id": "fourier-portrait", "kind": "web", "description": "portrait of Joseph Fourier"}],
         "target_words": 600},
    ],
    "cheatsheet_focus": ["coefficient formulas", "Parseval identity"],
}


def _lang(code: str) -> str:
    return LANG_NAMES.get(code, code)


def _length_words(n: int) -> str:
    return f"about {n} words of prose (formulas, code and figures do not count)"


def _caps_note(caps: TexCapabilities | None) -> str:
    note = caps.prompt_notes() if caps else ""
    return f"\nSystem limitations: {note}" if note else ""


def _msgs(user: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------------------------- plan

def plan_messages(req: LectureRequest, *, total_words: int, n_sections: tuple[int, int],
                  slide_digest: str | None = None, selected_slides: list[int] | None = None) -> list[dict]:
    parts = [
        f"Plan lecture notes on the topic: {req.topic}",
        f"Audience: {req.audience}. Language of the notes: {_lang(req.language)}.",
        f"Target length: {total_words} words of prose in {n_sections[0]}-{n_sections[1]} sections.",
    ]
    if req.course:
        parts.append(f"Course: {req.course}.")
    if req.instructions:
        parts.append(f"Additional instructions from the lecturer: {req.instructions}")
    if slide_digest:
        parts.append("\n## Lecture slides (number. title -- content excerpt)\n" + slide_digest)
        parts.append(f"Assign every one of these slides to exactly one section: {selected_slides}.")
    parts.append(
        "\n## Output\nReturn only a JSON object with exactly these keys (example values):\n"
        + json.dumps(PLAN_EXAMPLE, ensure_ascii=False, indent=1)
        + "\nRules: sections follow a logical teaching order (motivation -> core ideas -> methods/examples -> "
          "summary); key_points are short noun phrases that the section must explain; plan 1-2 figures for every "
          "section (kind matplotlib for plots, tikz for diagrams, web for real photos/illustrations); "
          f"target_words values sum to about {total_words}; slides lists may be empty when there are no slides; "
          "cheatsheet_focus lists the formulas/facts worth putting on a one-page cheat sheet."
    )
    return _msgs("\n".join(parts))


# ------------------------------------------------------------------------------- sections

def _figure_lines(figs: list[FigurePlan]) -> str:
    if not figs:
        return "none planned (add one only if it clearly helps)"
    return "\n".join(f"- id `{f.id}` ({f.kind}): {f.description}" for f in figs)


def section_messages(req: LectureRequest, outline: Outline, sec: SectionPlan, *, slide_text: str | None = None,
                     caps: TexCapabilities | None = None, has_slide_images: bool = False) -> list[dict]:
    idx = next((i for i, s in enumerate(outline.sections) if s.id == sec.id), 0)
    parts = [
        f"Write section {idx + 1} of {len(outline.sections)} of the lecture notes \"{outline.title}\" "
        f"(topic: {req.topic}).",
        f"Audience: {req.audience}. Write in {_lang(req.language)}.",
        "\n## Lecture outline\n" + outline.compact(),
        f"\n## This section\nTitle: {sec.title}",
        "Goals: " + ("; ".join(sec.goals) or "-"),
        "Key points to explain: " + ("; ".join(sec.key_points) or "-"),
        "Planned figures:\n" + _figure_lines(sec.figures),
        f"Length: {_length_words(sec.target_words)}.",
    ]
    if req.instructions:
        parts.append(f"Lecturer's instructions: {req.instructions}")
    if slide_text:
        parts.append("\n## Source slides for this section (extracted text, may be noisy)\n" + slide_text)
    parts.append("\n## Format\n" + STYLE_CARD + ("\n" + SLIDE_CARD if has_slide_images else "") + _caps_note(caps))
    parts.append(f"\nStart with \\section{{{sec.title}}}. Output only the LaTeX of this section.")
    return _msgs("\n".join(parts))


def revise_messages(source: str, feedback: list[str], *, language: str = "en") -> list[dict]:
    fb = "\n".join(f"- {f}" for f in feedback)
    user = (f"Revise the LaTeX section below according to the feedback. Keep everything that is correct, keep "
            f"the figure blocks unless the feedback says otherwise, and write in {_lang(language)}.\n\n"
            f"## Feedback\n{fb}\n\n## Current section\n{source}\n\n## Format\n{STYLE_CARD}\n\n"
            "Return the complete revised section, starting with \\section{...}.")
    return _msgs(user)


# ------------------------------------------------------------------------------- repairs

REPAIR_FORMAT = """Return only edit blocks of this form (one per change):
<<<<<<< SEARCH
exact lines copied from the source (without the line-number prefix)
=======
corrected lines
>>>>>>> REPLACE"""


def repair_latex_messages(source: str, errors: str, *, max_chars: int = 14000) -> list[dict]:
    user = ("The LaTeX section below does not compile. Fix exactly the listed errors with minimal changes; do "
            "not rewrite or shorten anything else.\n\n"
            f"## Errors\n{errors}\n\n## Source (line numbers added for reference)\n"
            f"{truncate_middle(number_lines(source), max_chars)}\n\n## Output\n{REPAIR_FORMAT}")
    return _msgs(user)


def repair_figure_messages(fig_id: str, kind: str, caption: str, code: str, feedback: str, *,
                           caps: TexCapabilities | None = None) -> list[dict]:
    rules = ("Python 3 with numpy/matplotlib (scipy allowed); create exactly one figure with fig, ax = "
             "plt.subplots(); label axes; avoid overlapping text; do not call savefig or show."
             if kind == "matplotlib" else
             "a single tikzpicture using standard TikZ libraries (arrows.meta, positioning, calc, shapes); "
             "colors lnblue, lnteal, lnorange, lnred, lnpurple, lngray are defined.")
    user = (f"A figure for lecture notes failed validation. Fix it.\nFigure id: {fig_id}; kind: {kind}; "
            f"caption: {caption}\n\n## Problems\n{feedback}\n\n## Code (line numbers added for reference)\n"
            f"{number_lines(code)}\n\n## Rules\n{rules}{_caps_note(caps)}\n\n## Output\nReturn the complete "
            f"corrected figure as a single block:\n<figure id=\"{fig_id}\" kind=\"{kind}\" caption=\"...\">\n"
            "code\n</figure>")
    return _msgs(user)


def figure_messages(topic: str, section_title: str, fig: FigurePlan, *, context: str = "",
                    caps: TexCapabilities | None = None) -> list[dict]:
    rules = ("Python 3 with numpy/matplotlib (scipy allowed); exactly one figure via fig, ax = plt.subplots(); "
             "label the axes and add a legend for several series; avoid overlapping labels; no savefig/show."
             if fig.kind == "matplotlib" else
             "one tikzpicture; libraries arrows.meta, positioning, calc, shapes.geometric, fit, backgrounds are "
             "loaded; colors lnblue, lnteal, lnorange, lnred, lnpurple, lngray are defined; keep it under ~14cm wide.")
    user = (f"Create a figure for lecture notes on \"{topic}\", section \"{section_title}\".\n"
            f"Figure id: {fig.id}; kind: {fig.kind}; what it should show: {fig.description}\n"
            + (f"\n## Section context\n{truncate_middle(context, 2500)}\n" if context else "")
            + f"\n## Rules\n{rules}{_caps_note(caps)}\n\n## Output\nReturn a single block:\n"
              f"<figure id=\"{fig.id}\" kind=\"{fig.kind}\" caption=\"One-sentence caption.\">\ncode\n</figure>")
    return _msgs(user)


def web_query_messages(topic: str, section_title: str, description: str, *, context: str = "",
                       failed: list[str] | None = None) -> list[dict]:
    user = (f"Lecture notes on \"{topic}\", section \"{section_title}\" need a real image (photo or classic "
            f"illustration) from Wikimedia Commons showing: {description}\n"
            + (f"\n## Section context\n{truncate_middle(context, 1500)}\n" if context else "")
            + (f"\nThese queries found nothing suitable: {'; '.join(failed)}. Use a simpler, more common query.\n"
               if failed else "")
            + "\nWrite one English image-search query of 2-5 words naming the concrete object (e.g. "
              "\"mitochondrion electron micrograph\", \"Michael Faraday portrait\", \"Wheatstone bridge circuit\"). "
              "Output only the query, without quotes.")
    return _msgs(user)


def pick_image_messages(topic: str, section_title: str, caption: str, candidates_text: str, *,
                        context: str = "") -> list[dict]:
    user = (f"Lecture notes on \"{topic}\", section \"{section_title}\", need an image for the caption: {caption}\n"
            + (f"\n## Section context\n{truncate_middle(context, 1200)}\n" if context else "")
            + f"\n## Candidate images from Wikimedia Commons (title, type, size, description, categories)\n"
              f"{candidates_text}\n\n"
              "Choose the candidate that best shows what the caption describes and is suitable for students "
              "(clear, not a different object with a similar name, not a map/flag/logo unless asked, not tiny, "
              "preferably English or no text). If none of them fits, answer none.\n"
              "Answer in exactly this format:\nchoice: <number or none>\nreject: <comma-separated numbers that "
              "clearly do not fit, or ->\nreason: <one short sentence>")
    return _msgs(user)


def parse_pick(text: str, n: int) -> tuple[int | None, bool]:
    """(1-based choice or None, parsed_ok). 'none' -> (None, True); garbage -> (None, False)."""
    m = re.search(r"choice\s*:\s*(none|\[?\d+\]?)", text or "", re.I)
    if not m:
        return None, False
    v = m.group(1).strip("[]").lower()
    if v == "none":
        return None, True
    k = int(v)
    return (k, True) if 1 <= k <= n else (None, False)


def clean_query(text: str) -> str:
    q = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    q = q.strip().strip("\"'`").removeprefix("Query:").removeprefix("query:").strip().strip("\"'`.")
    return " ".join(q.split()[:8])


# ------------------------------------------------------------------------------- cheat sheet

CHEATSHEET_CARD = r"""Use only boxes of this form, in a sensible order:
\begin{csbox}{Short box title}
compact content: itemize lists, short phrases (not full sentences), key formulas in $...$ or \[ ... \],
small tabular tables; \term{...} for key terms
\end{csbox}
No \section, no figures, no preamble. Escape \% \& \# \_ in text. Every formula must be correct."""


def cheatsheet_messages(title: str, facts: str, *, pages: int, language: str = "en",
                        focus: list[str] | None = None) -> list[dict]:
    user = (f"Create a dense cheat sheet for \"{title}\" that fits on {pages} landscape A4 page(s) in 3 columns. "
            f"Write in {_lang(language)}.\n"
            + (f"Must include: {'; '.join(focus)}.\n" if focus else "")
            + f"\n## Material to condense (extracted from the lecture notes)\n{truncate_middle(facts, 9000)}\n\n"
              f"## Format\n{CHEATSHEET_CARD}\n\nOutput only the LaTeX boxes.")
    return _msgs(user)


def condense_messages(source: str, *, pages: int, budget: int, language: str = "en") -> list[dict]:
    pct = max(10, min(60, int(100 * (pages - budget) / max(pages, 1)) + 10))
    user = (f"This cheat sheet takes {pages} pages but must fit in {budget}. Shorten it by about {pct}%: merge "
            "or drop the least essential boxes, cut words, keep the key formulas. Write in "
            f"{_lang(language)}.\n\n## Cheat sheet\n{source}\n\n## Format\n{CHEATSHEET_CARD}\n\n"
            "Return the complete shortened cheat sheet.")
    return _msgs(user)


TASKS = ("plan", "write_section", "revise_section", "repair_latex", "repair_figure", "figure", "cheatsheet",
         "condense_cheatsheet", "web_query", "pick_image")

__all__ = ["SYSTEM", "STYLE_CARD", "PROMPT_VERSION", "plan_messages", "section_messages", "revise_messages",
           "repair_latex_messages", "repair_figure_messages", "figure_messages", "cheatsheet_messages",
           "condense_messages", "TASKS"]
