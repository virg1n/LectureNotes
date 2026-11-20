"""Parse slide selections like "1-5, 8, 10-" or "3..7"."""

from __future__ import annotations

import re


def parse_selection(spec: str | None, total: int | None = None) -> list[int]:
    if spec is None or not str(spec).strip() or str(spec).strip().lower() in ("all", "*"):
        return list(range(1, (total or 0) + 1))
    out: set[int] = set()
    for part in re.split(r"[,;\s]+", str(spec).strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d*)\s*(?:-|\.\.|–|:)\s*(\d*)", part)
        if m:
            a = int(m.group(1)) if m.group(1) else 1
            b = int(m.group(2)) if m.group(2) else (total if total else a)
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"cannot parse slide selection part {part!r}")
    res = sorted(x for x in out if x >= 1)
    if total:
        res = [x for x in res if x <= total]
    return res
