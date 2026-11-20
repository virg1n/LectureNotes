#!/usr/bin/env bash
# Copy the project (code/configs/scripts/tests) to the training server.
#   HOST=v100 DEST=LectureNotes scripts/sync_to_server.sh
set -euo pipefail
HOST=${HOST:-v100}
DEST=${DEST:-/public/bsuvernev2_lecturenotes/code}
cd "$(dirname "$0")/.."
# only the listed paths are sent; do NOT add --exclude=data (it would also drop src/lecturenotes/data)
tar czf - --exclude=.venv --exclude='__pycache__' --exclude='*.egg-info' \
    pyproject.toml README.md src configs scripts tests \
  | ssh "$HOST" "mkdir -p $DEST && tar xzf - -C $DEST"
echo "synced to $HOST:$DEST"
