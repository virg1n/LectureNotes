"""Typed data contracts shared by the agent, the harness and the data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .utils import slugify

FigureKind = Literal["matplotlib", "tikz", "web"]

_KIND_ALIASES = {
    "matplotlib": "matplotlib", "python": "matplotlib", "plot": "matplotlib", "chart": "matplotlib",
    "graph": "matplotlib", "py": "matplotlib", "numpy": "matplotlib", "pyplot": "matplotlib",
    "tikz": "tikz", "diagram": "tikz", "pgfplots": "tikz", "latex": "tikz", "schematic": "tikz",
    "flowchart": "tikz", "tree": "tikz",
    "web": "web", "image": "web", "photo": "web", "picture": "web", "internet": "web", "wikimedia": "web",
}


def normalize_kind(v: Any) -> str:
    key = str(v or "matplotlib").strip().lower()
    return _KIND_ALIASES.get(key, "tikz" if "tikz" in key or "diagram" in key else "matplotlib")


class LectureRequest(BaseModel):
    """User input to the agent."""

    topic: str
    presentation: Path | None = None
    slides: str | None = Field(None, description='Slide selection, e.g. "1-5,8,10-"')
    expected_length: str | None = Field(None, description='"8 pages", "3000 words", "short"...')
    language: str = "en"
    audience: str = "university students"
    make_cheatsheet: bool = True
    cheatsheet_pages: int = 2
    instructions: str | None = None
    course: str | None = None
    author: str | None = None
    date: str | None = None


class FigurePlan(BaseModel):
    id: str
    kind: FigureKind = "matplotlib"
    description: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _slug(cls, v: Any) -> str:
        return slugify(v or "fig", 32, "fig")

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v: Any) -> str:
        return normalize_kind(v)


class SectionPlan(BaseModel):
    id: str = ""
    title: str
    goals: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    slides: list[int] = Field(default_factory=list)
    figures: list[FigurePlan] = Field(default_factory=list)
    target_words: int = 500

    @field_validator("goals", "key_points", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip(" -*\t") for s in v.split("\n") if s.strip(" -*\t")]
        return [str(x) for x in v if str(x).strip()]

    @field_validator("slides", mode="before")
    @classmethod
    def _slides(cls, v: Any) -> list[int]:
        if v is None:
            return []
        if isinstance(v, (int, float)):
            return [int(v)]
        if isinstance(v, str):
            from .ingest.selection import parse_selection

            return parse_selection(v)
        out = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    @field_validator("target_words", mode="before")
    @classmethod
    def _words(cls, v: Any) -> int:
        try:
            return max(80, min(int(float(v)), 6000))
        except (TypeError, ValueError):
            return 500


class Outline(BaseModel):
    title: str
    subtitle: str = ""
    abstract: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    sections: list[SectionPlan]
    cheatsheet_focus: list[str] = Field(default_factory=list)

    @field_validator("prerequisites", "learning_objectives", "cheatsheet_focus", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> list[str]:
        return SectionPlan._listify(v)

    @model_validator(mode="after")
    def _ids(self) -> "Outline":
        if not self.sections:
            raise ValueError("outline has no sections")
        seen_fig: set[str] = set()
        for i, s in enumerate(self.sections, 1):
            s.id = f"s{i}"
            for f in s.figures:
                base, k = f.id, 2
                while f.id in seen_fig:
                    f.id = f"{base}-{k}"
                    k += 1
                seen_fig.add(f.id)
        return self

    def compact(self) -> str:
        """Short textual rendering for prompts of downstream tasks."""
        lines = [f"Title: {self.title}"]
        for s in self.sections:
            kp = "; ".join(s.key_points[:6])
            lines.append(f"{s.id}. {s.title} -- {kp}")
        return "\n".join(lines)


class Issue(BaseModel):
    """A single problem found by a verifier (lint, compiler, figure runner, critic)."""

    source: Literal["lint", "compile", "figure", "length", "coverage", "structure", "format"]
    severity: Literal["error", "warning", "info"] = "error"
    code: str
    message: str
    line: int | None = None
    figure_id: str | None = None
    hint: str | None = None

    def render(self) -> str:
        loc = f"line {self.line}: " if self.line else ""
        fig = f"[figure {self.figure_id}] " if self.figure_id else ""
        hint = f" Hint: {self.hint}" if self.hint else ""
        return f"- ({self.severity}) {fig}{loc}{self.message}{hint}"
