#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
if [ -x .venv/bin/python ]; then TASK_PYTHON=.venv/bin/python; else TASK_PYTHON=../.venv312/bin/python; fi
"$TASK_PYTHON" -m uvicorn backend.api:app --host 127.0.0.1 --port 8010 &
TASK_BACKEND_PID=$!
trap 'kill "$TASK_BACKEND_PID" 2>/dev/null || true' EXIT INT TERM
npm run dev
