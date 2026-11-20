# Lecture-notes web app

Upload slides → the GPU server (`ssh v100`) writes notes + cheat sheet with the final model → results show up on the page.

```
browser ── ln.poh.kz (Cloudflare tunnel ln-poh) ── Pi: webapp/server.py :5020
                                                     │ ssh v100 (via cityu-gateway)
                                                     ▼
                       v100: scripts/web_job.py — free GPU → vLLM(final-model) → lecturenotes generate → stop vLLM
```

- `server.py` — stdlib only (Python ≥3.10). Jobs run one at a time and live in `jobs/<id>/` (`meta.json`, slides, `out/`).
  Unfinished jobs are resumed after a restart. Basic auth when `LN_WEB_PASSWORD` is set.
- `../scripts/web_job.py` — runs on v100, detached (`setsid -f`). Takes only GPUs with no processes and <600 MiB used
  (high indices first), retries another GPU if vLLM fails to start, always stops vLLM at the end. The remote job dir
  `/public/bsuvernev2_lecturenotes/web_jobs/<id>` is removed after the results are copied back.
- Typical time: ~2 min model load + 3–8 min generation.

Run locally (needs `ssh v100` to work without a password prompt):

```
python webapp/server.py --port 5020        # http://127.0.0.1:5020
```

## Deployment on the Pi (`ssh -p 1960 bogdan@nsd60.sytes.net`)

```
/home/bogdan/sites/ln/
├── app/            server.py, index.html          (copy of this folder)
├── runtime/        start-ln.sh, supervise-tunnel.sh, cloudflared.yml (tunnel ln-poh → 127.0.0.1:5020)
├── jobs/  run/  logs/
└── .env            LN_WEB_USER / LN_WEB_PASSWORD  (mode 600)
```

- Started at boot by `@reboot /home/bogdan/sites/ln/runtime/start-ln.sh` (crontab); rerun it to start missing processes.
- Update: `scp -P 1960 webapp/server.py webapp/index.html bogdan@nsd60.sytes.net:sites/ln/app/`, then
  `kill $(cat ~/sites/ln/run/web.pid); ~/sites/ln/runtime/start-ln.sh`.
- Health: `curl -sI http://127.0.0.1:5020/` (401 = up, auth on), `tail ~/sites/ln/logs/{web,tunnel}.log`.
- The Pi's key `~/.ssh/id_ed25519.pub` must be in `~/.ssh/authorized_keys` on cityu-gateway and v100;
  `~/.ssh/config` on the Pi has the `cityu-gateway` / `v100` hosts.
