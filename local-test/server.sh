#!/bin/sh
# Run one instance per terminal. Model loads on the first image request.
set -eu
cd "$(dirname "$0")/.."
exec env CPU_THREADS="${CPU_THREADS:-4}" PRELOAD_MODEL=false \
  .venv/bin/python -m uvicorn vehicle_pipeline.api:app \
  --host 127.0.0.1 --port "${1:-8001}" --workers 1 --limit-concurrency 1200
