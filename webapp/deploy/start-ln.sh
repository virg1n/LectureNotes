#!/bin/sh
# Starts the ln.poh.kz web app (127.0.0.1:5020) and its tunnel supervisor if they are not running. Run at boot via cron.
set -eu
BASE=/home/bogdan/sites/ln
mkdir -p "$BASE/logs" "$BASE/run" "$BASE/jobs"
set -a; . "$BASE/.env"; set +a
if ! test -s "$BASE/run/web.pid" || ! kill -0 "$(cat "$BASE/run/web.pid")" 2>/dev/null; then
  LN_WEB_JOBS="$BASE/jobs" nohup /usr/bin/python3 "$BASE/app/server.py" --port 5020 </dev/null >>"$BASE/logs/web.log" 2>&1 &
  echo $! >"$BASE/run/web.pid"
fi
if ! test -s "$BASE/run/tunnel.pid" || ! kill -0 "$(cat "$BASE/run/tunnel.pid")" 2>/dev/null; then
  nohup "$BASE/runtime/supervise-tunnel.sh" </dev/null >>"$BASE/logs/tunnel.log" 2>&1 &
  echo $! >"$BASE/run/tunnel.pid"
fi
