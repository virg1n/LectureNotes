#!/usr/bin/env python3
"""Run one web-app job on the GPU server: pick a free GPU, start vLLM with the final model, generate, stop vLLM.

    python scripts/web_job.py /public/bsuvernev2_lecturenotes/web_jobs/<id>

The job dir must contain job.json ({"slides": "<file name>", "topic", "length", "pages", "cheatsheet_pages", "lang"})
and the slides file. Progress goes to status.json, outputs to out/. Started detached (setsid nohup) by webapp/server.py,
so it survives the ssh connection dropping. Only GPUs with no processes and almost no memory in use are taken.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
LN_ROOT = Path(os.environ.get("LN_ROOT", "/public/bsuvernev2_lecturenotes"))
MODEL = CODE / "checkpoints" / "final-model"
MAX_USED_MIB = 600          # a GPU counts as free below this and with no compute processes
BASE_PORT = 8300


def status(job: Path, phase: str, **extra) -> None:
    data = {"phase": phase, "time": time.time(), **extra}
    tmp = job / "status.json.tmp"
    tmp.write_text(json.dumps(data))
    tmp.replace(job / "status.json")


def free_gpus() -> list[int]:
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, check=True).stdout.strip().splitlines()
    busy = {line.split(",")[0].strip() for line in q("--query-compute-apps=gpu_uuid") if line.strip()}
    out = []
    for line in q("--query-gpu=index,uuid,memory.used"):
        idx, uuid, used = (x.strip() for x in line.split(","))
        if uuid not in busy and int(used) < MAX_USED_MIB:
            out.append(int(idx))
    return out[::-1]        # high indices first: low ones are the usual defaults for other people


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_vllm(job: Path) -> tuple[subprocess.Popen, int, int]:
    """Start vLLM on the first free GPU that actually comes up. Holds a lock so parallel jobs don't race."""
    lock = open(LN_ROOT / ".web_gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        tried = set()
        while True:
            cands = [g for g in free_gpus() if g not in tried]
            if not cands:
                if tried:
                    raise RuntimeError(f"vLLM failed on GPUs {sorted(tried)}, see vllm.log")
                status(job, "waiting_gpu", message="all GPUs are busy, waiting")
                time.sleep(30)
                continue
            gpu = cands[0]
            tried.add(gpu)
            port = BASE_PORT + gpu
            status(job, "loading", gpu=gpu, message=f"starting the model on GPU {gpu}")
            env = dict(os.environ, GPUS=str(gpu), PORT=str(port), NAME="student")
            log = open(job / "vllm.log", "ab")
            proc = subprocess.Popen(["bash", str(CODE / "scripts/serve_vllm.sh"), str(MODEL)], cwd=CODE, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None and not healthy(port):
                time.sleep(5)
            if proc.poll() is None:
                return proc, gpu, port
            # died during startup (usually someone else grabbed the GPU) -> try the next one
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main() -> int:
    job = Path(sys.argv[1]).resolve()
    spec = json.loads((job / "job.json").read_text())
    (job / "out").mkdir(exist_ok=True)
    (job / "pid").write_text(str(os.getpid()))
    proc = None
    stop = lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    signal.signal(signal.SIGTERM, stop)
    try:
        proc, gpu, port = start_vllm(job)
        status(job, "generating", gpu=gpu, message="writing lecture notes")
        argv = [sys.executable, "-m", "lecturenotes.cli", "generate",
                "--topic", spec.get("topic") or Path(spec["slides"]).stem,
                "--slides", str(job / spec["slides"]),
                "--config", "configs/agent/student_vllm.yaml", "--out", str(job / "out")]
        for key, flag in [("length", "--length"), ("pages", "--pages"), ("lang", "--lang"),
                          ("cheatsheet_pages", "--cheatsheet-pages")]:
            if spec.get(key):
                argv += [flag, str(spec[key])]
        if spec.get("no_cheatsheet"):
            argv.append("--no-cheatsheet")
        env = dict(os.environ, LN_STUDENT_URL=f"http://127.0.0.1:{port}/v1")
        env.pop("LN_VLM_URL", None)        # the VLM judge model is not kept on the server
        with open(job / "generate.log", "ab") as log:
            rc = subprocess.run(argv, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not (job / "out" / "notes.pdf").exists():
            status(job, "error", message=f"generation failed (exit {rc}), see generate.log")
            return 1
        status(job, "done", gpu=gpu)
        return 0
    except KeyboardInterrupt:
        status(job, "error", message="cancelled")
        return 1
    except Exception as e:  # noqa: BLE001
        status(job, "error", message=str(e))
        return 1
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
