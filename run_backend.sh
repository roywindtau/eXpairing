#!/usr/bin/env bash
# run_backend.sh — start the FastAPI backend locally with auto-reload.
# nodemon-equivalent: edits to backend/*.py restart the server automatically.
#
# Usage: ./run_backend.sh [--no-reload] [extra uvicorn args...]
#   --no-reload   disable auto-reload (e.g. to avoid reloading the 278MB ALS
#                 model on every save during heavy editing sessions)
set -euo pipefail

# Prefer the project venv if it has uvicorn; else fall back to python3.
PY="python3"
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import uvicorn" 2>/dev/null; then
  PY=".venv/bin/python"
fi

RELOAD="--reload"
ARGS=()
for arg in "$@"; do
  case $arg in
    --no-reload) RELOAD="" ;;
    *)           ARGS+=("$arg") ;;
  esac
done

echo "==> Backend on http://localhost:8000  (${RELOAD:-no reload}, $PY)"
exec "$PY" -m uvicorn backend.main:app --port 8000 $RELOAD "${ARGS[@]}"
