#!/usr/bin/env python3
"""Lecture-notes web app: upload slides, the GPU server (ssh v100) writes the notes, results come back here.

    python webapp/server.py --port 5020                    # http://127.0.0.1:5020
    LN_WEB_PASSWORD=... python webapp/server.py            # HTTP basic auth (user: LN_WEB_USER, default "ln")

Standard library only. Each job: slides are copied to the GPU server, scripts/web_job.py is started there detached
(picks a free GPU, starts vLLM with checkpoints/final-model, runs the agent, stops vLLM), this process polls its
status.json and copies notes/cheatsheet back into webapp/jobs/<id>/ once it is done. Jobs run one at a time.
"""
import argparse
import base64
import email.parser
import email.policy
import hmac
import io
import json
import os
import queue
import re
import secrets
import shlex
import subprocess
import tarfile
import threading
import time
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = Path(os.environ.get("LN_WEB_JOBS", HERE / "jobs"))
HOST = os.environ.get("LN_SSH_HOST", "v100")
REMOTE_ROOT = os.environ.get("LN_REMOTE_ROOT", "/public/bsuvernev2_lecturenotes")
REMOTE_JOBS = f"{REMOTE_ROOT}/web_jobs"
USER = os.environ.get("LN_WEB_USER", "ln")
PASSWORD = os.environ.get("LN_WEB_PASSWORD", "")
MAX_UPLOAD = 60 * 1024 * 1024
SLIDE_EXT = {".pdf", ".pptx", ".tex", ".md"}
RESULT_FILES = ["notes.pdf", "cheatsheet.pdf", "notes.tex", "cheatsheet.tex", "report.md"]
FINAL = {"done", "error", "cancelled"}
POLL = 20

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30"]
if os.name != "nt":   # reuse one connection through the gateway (not supported by Windows OpenSSH)
    SSH_OPTS += ["-o", "ControlMaster=auto", "-o", f"ControlPath={Path.home()}/.ssh/cm-ln-%C", "-o", "ControlPersist=10m"]

lock = threading.Lock()
work: "queue.Queue[str]" = queue.Queue()


# ---------- job store ----------

def meta_path(jid: str) -> Path:
    return JOBS / jid / "meta.json"


def load(jid: str) -> dict:
    return json.loads(meta_path(jid).read_text(encoding="utf-8"))


def save(jid: str, **upd) -> dict:
    with lock:
        m = load(jid)
        m.update(upd, updated=time.time())
        tmp = meta_path(jid).with_suffix(".tmp")
        tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(meta_path(jid))
        return m


def all_jobs() -> list[dict]:
    out = []
    for p in JOBS.glob("*/meta.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return sorted(out, key=lambda m: m["created"], reverse=True)


# ---------- remote side ----------

def ssh(cmd: str, *, timeout: int = 120, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, HOST, cmd], timeout=timeout, capture_output=True, **kw)


def run_job(jid: str) -> None:
    m = load(jid)
    if m["state"] in FINAL:
        return
    rdir = f"{REMOTE_JOBS}/{jid}"
    try:
        if not m.get("remote_started"):
            save(jid, state="uploading", message="copying slides to the GPU server")
            spec = {k: m.get(k) for k in ("topic", "length", "pages", "cheatsheet_pages", "lang", "no_cheatsheet")}
            spec["slides"] = m["slides"]
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                tar.add(JOBS / jid / m["slides"], arcname=m["slides"])
                data = json.dumps(spec, ensure_ascii=False).encode()
                info = tarfile.TarInfo("job.json")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            r = ssh(f"mkdir -p {rdir} && tar -xf - -C {rdir}", input=buf.getvalue())
            if r.returncode:
                raise RuntimeError("upload failed: " + r.stderr.decode(errors="replace")[-400:])
            # setsid -f + redirecting every fd: nothing may keep the ssh channel open, or this call hangs
            r = ssh(f"cd {REMOTE_ROOT}/code && source scripts/env.sh >/dev/null 2>&1 && "
                    f"setsid -f python scripts/web_job.py {rdir} > {rdir}/runner.log 2>&1 < /dev/null && echo started")
            if b"started" not in r.stdout:
                raise RuntimeError("could not start the job: " + r.stderr.decode(errors="replace")[-400:])
            m = save(jid, remote_started=time.time(), state="queued", message="waiting for a GPU")
        while True:
            if load(jid).get("cancel"):
                ssh(f"kill $(cat {rdir}/pid) 2>/dev/null; true")
                save(jid, state="cancelled", message="cancelled")
                return
            try:
                r = ssh(f"cat {rdir}/status.json 2>/dev/null; echo; echo @@LOG@@; "
                        f"tail -c 3000 {rdir}/generate.log 2>/dev/null; tail -c 1500 {rdir}/runner.log 2>/dev/null",
                        timeout=90)
                text = r.stdout.decode(errors="replace")
                head, _, log = text.partition("@@LOG@@")
                st = json.loads(head.strip()) if head.strip() else {}
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                time.sleep(POLL)
                continue
            phase = st.get("phase", "queued")
            upd = dict(log=log.strip()[-3000:], gpu=st.get("gpu"))
            if phase == "error":
                save(jid, state="error", message=st.get("message", "failed"), **upd)
                return
            if phase == "done":
                save(jid, state="fetching", message="copying results back", **upd)
                break
            save(jid, state=phase, message=st.get("message", ""), **upd)
            # runner died without writing a final status (e.g. node reboot)
            if r.returncode == 0 and time.time() - st.get("time", time.time()) > 3600:
                alive = ssh(f"kill -0 $(cat {rdir}/pid 2>/dev/null) 2>/dev/null && echo alive").stdout
                if b"alive" not in alive:
                    save(jid, state="error", message="the job process on the GPU server died")
                    return
            time.sleep(POLL)
        names = " ".join(RESULT_FILES + ["figures"])
        p = subprocess.Popen(["ssh", *SSH_OPTS, HOST,
                              f"cd {rdir}/out && tar -cf - $(ls -d {names} 2>/dev/null) -C {rdir} generate.log"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        with tarfile.open(fileobj=p.stdout, mode="r|") as tar:
            tar.extractall(JOBS / jid / "out", filter="data")
        p.wait(600)
        if not (JOBS / jid / "out" / "notes.pdf").exists():
            raise RuntimeError("results could not be copied back")
        ssh(f"rm -rf {rdir}")        # /public quota is shared; results now live here
        save(jid, state="done", message="", finished=time.time())
    except Exception as e:  # noqa: BLE001
        save(jid, state="error", message=str(e)[:500])


def worker() -> None:
    while True:
        jid = work.get()
        try:
            run_job(jid)
        finally:
            work.task_done()


# ---------- HTTP ----------

def parse_multipart(ctype: str, body: bytes) -> dict:
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    fields = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        fname = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        fields[name] = (fname, payload) if fname is not None else payload.decode("utf-8", "replace").strip()
    return fields


class Handler(BaseHTTPRequestHandler):
    server_version = "lecturenotes"

    def log_message(self, fmt, *args):
        pass

    def authed(self) -> bool:
        if not PASSWORD:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                u, _, p = base64.b64decode(h[6:]).decode().partition(":")
                if hmac.compare_digest(u, USER) and hmac.compare_digest(p, PASSWORD):
                    return True
            except ValueError:
                pass
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Lecture notes"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200) -> None:
        self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if not self.authed():
            return
        path = self.path.split("?")[0]
        if path == "/":
            return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/jobs":
            return self.json([{k: v for k, v in m.items() if k != "log"} for m in all_jobs()])
        if mm := re.fullmatch(r"/api/jobs/([0-9a-f]{12})", path):
            return self.json(load(mm[1])) if meta_path(mm[1]).exists() else self.json({"error": "not found"}, 404)
        if mm := re.fullmatch(r"/files/([0-9a-f]{12})/bundle\.zip", path):
            out = JOBS / mm[1] / "out"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(out.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(out))
            name = load(mm[1]).get("slug", "lecture-notes")
            return self.send(200, buf.getvalue(), "application/zip",
                             {"Content-Disposition": f'attachment; filename="{name}.zip"'})
        if mm := re.fullmatch(r"/files/([0-9a-f]{12})/([\w.-]+)", path):
            f = JOBS / mm[1] / "out" / mm[2]
            if mm[2] in RESULT_FILES + ["generate.log"] and f.is_file():
                ctype = {".pdf": "application/pdf"}.get(f.suffix, "text/plain; charset=utf-8")
                return self.send(200, f.read_bytes(), ctype, {"Content-Disposition": f'inline; filename="{f.name}"'})
        self.json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.authed():
            return
        path = self.path.split("?")[0]
        if mm := re.fullmatch(r"/api/jobs/([0-9a-f]{12})/cancel", path):
            if meta_path(mm[1]).exists() and load(mm[1])["state"] not in FINAL:
                save(mm[1], cancel=True, message="cancelling")
            return self.json({"ok": True})
        if mm := re.fullmatch(r"/api/jobs/([0-9a-f]{12})/delete", path):
            if meta_path(mm[1]).exists() and load(mm[1])["state"] in FINAL:
                import shutil
                shutil.rmtree(JOBS / mm[1])
            return self.json({"ok": True})
        if path != "/api/jobs":
            return self.json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            return self.json({"error": "file is too large (60 MB max)"}, 413)
        try:
            f = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(n))
        except Exception:  # noqa: BLE001
            return self.json({"error": "bad form"}, 400)
        fname, data = f.get("slides") if isinstance(f.get("slides"), tuple) else (None, b"")
        ext = Path(fname or "").suffix.lower()
        if not data or ext not in SLIDE_EXT:
            return self.json({"error": "upload a .pdf, .pptx, .tex or .md file"}, 400)
        pages = f.get("pages", "")
        if pages and not re.fullmatch(r"[\d,\s-]+", pages):
            return self.json({"error": "pages: use e.g. 1-12,15"}, 400)
        cs = f.get("cheatsheet_pages", "")
        if cs and not re.fullmatch(r"\d", cs):
            return self.json({"error": "cheat sheet pages: 1-9"}, 400)
        jid = secrets.token_hex(6)
        (JOBS / jid).mkdir(parents=True)
        (JOBS / jid / f"slides{ext}").write_bytes(data)
        stem = Path(fname).stem
        meta = dict(id=jid, created=time.time(), updated=time.time(), state="new", message="queued",
                    slides=f"slides{ext}", slides_name=Path(fname).name,
                    slug=re.sub(r"[^\w-]+", "-", stem)[:60] or "lecture-notes",
                    topic=f.get("topic", "")[:300], length=f.get("length", "")[:40], pages=pages[:60],
                    lang=f.get("lang", "")[:20], cheatsheet_pages=cs, no_cheatsheet=f.get("no_cheatsheet") == "on")
        meta_path(jid).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        work.put(jid)
        self.json({"id": jid})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5020)
    a = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    for m in sorted(all_jobs(), key=lambda m: m["created"]):     # resume after a restart
        if m["state"] not in FINAL:
            work.put(m["id"])
    threading.Thread(target=worker, daemon=True).start()
    print(f"serving on http://{a.host}:{a.port}  (GPU host: {HOST}, auth: {'on' if PASSWORD else 'off'})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
