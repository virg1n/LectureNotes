"""Deterministic offline 'model' for tests and demos of the harness.

It produces plausible outputs for every task, *deliberately including typical
small-model mistakes* (Markdown leakage, an undefined macro, a broken figure),
so an end-to-end run exercises autofix, LaTeX repair, figure repair and
quarantine paths without a GPU. Its repair behaviour is a rule-based baseline.
"""

from __future__ import annotations

import json
import re

from .base import Completion, GenParams, LLMBackend, Message


def _user(messages: list[Message]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


class MockBackend(LLMBackend):
    name = "mock"
    model = "mock-lecturer"

    def __init__(self, inject_errors: bool = True):
        self.inject_errors = inject_errors

    async def agenerate(self, messages: list[Message], params: GenParams) -> list[Completion]:
        return await self.agenerate_task("write_section", messages, params)

    async def agenerate_task(self, task: str, messages: list[Message], params: GenParams) -> list[Completion]:
        fn = getattr(self, f"_{task}", None)
        text = fn(_user(messages)) if fn else "OK"
        return [Completion(text, "stop", len(str(messages)) // 4, len(text) // 4, 0.0, self.model)
                for _ in range(params.n)]

    # ------------------------------------------------------------------ tasks
    def _plan(self, prompt: str) -> str:
        topic = re.search(r"topic: (.+)", prompt)
        topic = topic.group(1).strip() if topic else "Lecture"
        words = int(re.search(r"Target length: (\d+)", prompt).group(1)) if "Target length" in prompt else 1800
        slides = re.search(r"exactly one section: \[([^\]]*)\]", prompt)
        sl = [int(x) for x in slides.group(1).split(",") if x.strip()] if slides else []
        thirds = [sl[i::3] for i in range(3)] if sl else [[], [], []]
        plan = {
            "title": topic, "subtitle": "Lecture notes",
            "abstract": f"An introduction to {topic}: core definitions, a worked example and common pitfalls.",
            "prerequisites": ["calculus", "linear algebra"],
            "learning_objectives": [f"state the main definitions of {topic}", "work through a standard example"],
            "sections": [
                {"title": "Motivation and definitions", "goals": ["set up notation"],
                 "key_points": ["definition", "notation", "basic example"], "slides": sorted(thirds[0]),
                 "figures": [{"id": "overview-plot", "kind": "matplotlib", "description": "a smooth curve"}],
                 "target_words": words // 3},
                {"title": "Core ideas", "goals": ["explain the main result"],
                 "key_points": ["main theorem", "structure diagram"], "slides": sorted(thirds[1]),
                 "figures": [{"id": "structure", "kind": "tikz", "description": "a two-node diagram"}],
                 "target_words": words // 3},
                {"title": "Worked example and pitfalls", "goals": ["practice"],
                 "key_points": ["worked example", "common mistakes"], "slides": sorted(thirds[2]),
                 "figures": [{"id": "example-data", "kind": "matplotlib", "description": "scatter of samples"}],
                 "target_words": words // 3},
            ],
            "cheatsheet_focus": ["main definition", "main theorem"],
        }
        return "Here is the plan:\n```json\n" + json.dumps(plan, indent=1) + "\n```"

    def _write_section(self, prompt: str) -> str:
        title = re.search(r"Start with \\section\{(.+?)\}\. Output", prompt)
        title = title.group(1) if title else "Section"
        idx = int(re.search(r"Write section (\d+)", prompt).group(1)) if "Write section" in prompt else 1
        para = (f"In this part we study {title.lower()} from first principles. We begin with the informal picture "
                "that motivates the definitions, then make every notion precise and check it against small cases. "
                "Along the way we point out which assumptions are essential and which are only convenient, since "
                "students often conflate the two. The examples were chosen so that each one isolates a single "
                "phenomenon; together they show how the abstract statement is used in practice and where it fails.")
        if idx == 1:
            return rf"""\section{{{title}}}
## Why it matters
{para}
We use **bold** ideas and $x_i$ for samples, and 50% of the time this is enough.

\begin{{definition}}[Main object]\label{{def:main}}
A \term{{main object}} is a pair $(X, d)$ with $d(x,y) \geq 0$.
\end{{definition}}

<figure id="overview-plot" kind="matplotlib" caption="A smooth curve and its derivative.">
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 200)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="f")
ax.plot(x, np.cos(x), label="f'")
ax.set_xlabel("$x$"); ax.set_ylabel("value"); ax.legend()
</figure>
As Figure~\ref{{fig:overview-plot}} shows, the curve is smooth.
\begin{{summary}}
\begin{{itemize}}\item definition \item notation\end{{itemize}}
\end{{summary}}
"""
        if idx == 2:
            bad = r" Note that \qedsymbolx closes the argument." if self.inject_errors else ""
            return rf"""\section{{{title}}}
{para}
\begin{{theorem}}[Main theorem]
For every $\varepsilon > 0$ there is $N$ such that $\abs{{a_n - a}} < \varepsilon$ for $n \geq N$.{bad}
\end{{theorem}}
\begin{{proof}}
Standard.
\end{{proof}}
<figure id="structure" kind="tikz" caption="Two connected concepts.">
\begin{{tikzpicture}}[node distance=3cm]
\node[draw=lnblue, rounded corners] (a) {{Idea}};
\node[draw=lnteal, rounded corners, right=of a] (b) {{Result}};
\draw[-Stealth, thick] (a) -- (b);
\end{{tikzpicture}}
</figure>
\begin{{keyidea}}
The structure diagram in Figure~\ref{{fig:structure}} summarises the dependency.
\end{{keyidea}}
"""
        broken = "ax.plot([1, 2, 3], [1, 2])" if self.inject_errors else "ax.plot([1, 2, 3], [1, 2, 3])"
        return rf"""\section{{{title}}}
{para}
\begin{{example}}[A computation]
We compute $\sum_{{k=1}}^n k = \frac{{n(n+1)}}{{2}}$ and check it for $n = 4$.
\end{{example}}
<figure id="example-data" kind="matplotlib" caption="Samples from the model.">
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
{broken}
ax.set_xlabel("n"); ax.set_ylabel("value")
</figure>
\begin{{warning}}
Do not confuse the index $k$ with the bound $n$.
\end{{warning}}
"""

    def _revise_section(self, prompt: str) -> str:
        m = re.search(r"## Current section\n(.*?)\n\n## Format", prompt, re.S)
        src = m.group(1) if m else "\\section{Revised}\n"
        extra = ("\n\nTo consolidate, we revisit the main idea from a different angle and relate it to the previous "
                 "discussion, which clarifies why the assumptions are needed. A second viewpoint comes from "
                 "geometry: drawing the objects involved often reveals the argument before any formula is written. "
                 "Finally, we contrast the result with a nearby statement that is false, because counterexamples "
                 "mark the boundary of a theorem more sharply than any list of hypotheses.")
        return src + extra

    def _repair_latex(self, prompt: str) -> str:
        """Rule-based baseline: delete undefined control sequences mentioned in the errors."""
        edits = []
        src = re.search(r"## Source \(line numbers added for reference\)\n(.*?)\n\n## Output", prompt, re.S)
        lines = {}
        if src:
            for ln in src.group(1).splitlines():
                m = re.match(r"\s*(\d+)\| ?(.*)$", ln)
                if m:
                    lines[int(m.group(1))] = m.group(2)
        for m in re.finditer(r"line (\d+): Undefined control sequence (\\[A-Za-z@]+)", prompt):
            n, cs = int(m.group(1)), m.group(2)
            if n in lines and cs in lines[n]:
                edits.append(f"<<<<<<< SEARCH\n{lines[n]}\n=======\n{lines[n].replace(cs, '')}\n>>>>>>> REPLACE")
        return "\n".join(edits) or "I could not find the problem."

    def _repair_figure(self, prompt: str) -> str:
        fid = re.search(r"Figure id: ([\w-]+); kind: (\w+)", prompt)
        fid, kind = (fid.group(1), fid.group(2)) if fid else ("fig", "matplotlib")
        if kind == "tikz":
            body = "\\begin{tikzpicture}\\draw[lnblue, thick] (0,0) circle (1);\\end{tikzpicture}"
        else:
            body = ("import numpy as np\nimport matplotlib.pyplot as plt\nn = np.arange(1, 4)\nfig, ax = plt.subplots()\n"
                    "ax.plot(n, n**2, marker='o')\nax.set_xlabel('n'); ax.set_ylabel('value')")
        return f'<figure id="{fid}" kind="{kind}" caption="Repaired figure.">\n{body}\n</figure>'

    def _figure(self, prompt: str) -> str:
        return self._repair_figure(prompt)

    def _web_query(self, prompt: str) -> str:
        m = re.search(r"showing: (.+)", prompt)
        return " ".join(re.findall(r"[A-Za-z]+", m.group(1) if m else "diagram")[:3])

    def _pick_image(self, prompt: str) -> str:
        return "choice: 1\nreject: -\nreason: first candidate matches the caption"

    def _cheatsheet(self, prompt: str) -> str:
        boxes = []
        for i, m in enumerate(re.finditer(r"^\[(\w+)\] (.+)$", prompt, re.M)):
            if i >= 10:
                break
            boxes.append(f"\\begin{{csbox}}{{{m.group(1).title()}}}\n{m.group(2)[:200]}\n\\end{{csbox}}")
        if not boxes:
            boxes = ["\\begin{csbox}{Key facts}\n\\begin{itemize}\\item fact one\\item fact two\\end{itemize}\n\\end{csbox}"]
        return "\n".join(boxes)

    def _condense_cheatsheet(self, prompt: str) -> str:
        m = re.search(r"## Cheat sheet\n(.*?)\n\n## Format", prompt, re.S)
        boxes = re.findall(r"\\begin\{csbox\}.*?\\end\{csbox\}", m.group(1) if m else "", re.S)
        return "\n".join(boxes[: max(1, len(boxes) // 2)])
