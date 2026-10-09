#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-gps/bin/python
"$PY" -c 'import sys, torch; assert sys.platform == "linux", "Run this on the Linux GPU instance"; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
# CPU and GPU ORT wheels install the same module; remove CPU before installing GPU.
"$PY" -m pip uninstall -y onnxruntime
"$PY" -m pip install onnxruntime-gpu==1.22.0
"$PY" -m pip check
"$PY" - <<'PY'
import torch
import onnxruntime as ort
ort.preload_dlls(directory="")
assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDAExecutionProvider missing'
print('GPU preparation enabled. Existing model downloads are reused.')
PY
