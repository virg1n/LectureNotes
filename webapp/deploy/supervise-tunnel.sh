#!/bin/sh
set -u

BASE=/home/bogdan/sites/ln
CLOUDFLARED=/usr/local/bin/cloudflared
CONFIG="$BASE/runtime/cloudflared.yml"
RETRY_SECONDS="${LN_TUNNEL_RETRY_SECONDS:-10}"
child_pid=

stop_child() {
  if test -n "$child_pid" && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}

trap stop_child INT TERM HUP

while :; do
  printf '%s Starting ln-poh tunnel\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$CLOUDFLARED" --no-autoupdate --config "$CONFIG" tunnel run &
  child_pid=$!
  wait "$child_pid"
  status=$?
  child_pid=
  printf '%s Tunnel exited with status %s; retrying in %ss\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$RETRY_SECONDS"
  sleep "$RETRY_SECONDS" &
  child_pid=$!
  wait "$child_pid" 2>/dev/null || true
  child_pid=
done
