#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/removeBG3-astra}"
cd "$APP_DIR"
# Vast testing only: disable authentication for this launched process.
# The existing CPU deployment and its API_KEY are not modified.
export API_KEY=""
mkdir -p .cache
command -v flock >/dev/null || { echo "Install util-linux (flock) first"; exit 1; }
# Serialize setup and keep a separate lifetime lock for this API only.
exec 9>.cache/vast-setup.lock
flock -n 9 || { echo "Vast setup is already running"; exit 0; }
flock -n .cache/vast-api.lock true || { echo "Vast API is already running; see app.log"; exit 0; }

VENV="$APP_DIR/.venv-gpu"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip uninstall -y onnxruntime
"$VENV/bin/python" -m pip install -r requirements-gpu.txt

export INFERENCE_DEVICE=cuda
export CPU_THREADS="${CPU_THREADS:-4}"
export SEGMENTATION_MODE="${SEGMENTATION_MODE:-single}"
export PRELOAD_MODEL=true
export QUEUE_CAPACITY="${QUEUE_CAPACITY:-1000}"
export U2NET_HOME="${U2NET_HOME:-/root/.u2net}"
export PORT="${PORT:-8000}"
# Never kill other uvicorn processes. An occupied port is a configuration error.
"$VENV/bin/python" - <<'CHECK'
import os, socket
with socket.socket() as sock:
    sock.bind(("0.0.0.0", int(os.environ["PORT"])))
CHECK
nohup flock --nonblock --no-fork .cache/vast-api.lock \
  "$VENV/bin/python" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" \
  --workers 1 --limit-concurrency 1200 > app.log 2>&1 9>&- &
pid=$!
echo "$pid" > .cache/vast-api.pid
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  tail -n 40 app.log
  exit 1
fi
echo "Vast GPU API launched (PID $pid). Check app.log for startup completion."
