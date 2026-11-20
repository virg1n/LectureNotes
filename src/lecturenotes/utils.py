"""Small shared helpers: hashing, slugs, robust subprocess execution, JSON extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

IS_WINDOWS = sys.platform.startswith("win")


def slugify(text: str, maxlen: int = 40, default: str = "item") -> str:
    """ASCII slug usable as a LaTeX label / file name."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    text = text[:maxlen].strip("-")
    return text or default


def stable_hash(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        if not isinstance(p, (bytes, bytearray)):
            p = json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode()
        h.update(p)
        h.update(b"\x00")
    return h.hexdigest()[:n]


def which(name: str) -> str | None:
    return shutil.which(name)


@dataclass
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_process(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 60.0,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    preexec_fn=None,
) -> ProcResult:
    """Run a command, kill the whole process tree on timeout, never raise on failure."""
    start = time.monotonic()
    kwargs: dict[str, Any] = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
        if preexec_fn is not None:
            kwargs["preexec_fn"] = preexec_fn
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
    except (FileNotFoundError, PermissionError, OSError) as e:
        return ProcResult(127, "", f"failed to start {cmd[0]!r}: {e}", False, 0.0)
    try:
        out, err = proc.communicate(
            input=input_text.encode("utf-8") if input_text is not None else None,
            timeout=timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = b"", b""
        timed_out = True
    return ProcResult(
        returncode=proc.returncode if proc.returncode is not None else -9,
        stdout=out.decode("utf-8", errors="replace") if out else "",
        stderr=err.decode("utf-8", errors="replace") if err else "",
        timed_out=timed_out,
        duration=time.monotonic() - start,
    )


# --------------------------------------------------------------------------- JSON

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def extract_json_object(text: str) -> Any:
    """Extract the first JSON object from model output (tolerates fences/prose/trailing commas)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    candidate = text[start : end + 1] if end > 0 else text[start:] + "}" * max(depth, 0)
    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    # LaTeX inside JSON strings often carries invalid escapes such as "\alpha".
    fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", _TRAILING_COMMA.sub(r"\1", candidate))
    return json.loads(fixed)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def number_lines(text: str, start: int = 1, width: int = 4) -> str:
    return "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(text.splitlines(), start))


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[... truncated ...]\n" + text[-half:]
