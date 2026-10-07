#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${GPS_PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use Python 3.11 or 3.12; set GPS_PYTHON=python3.11 if needed"'
"$PYTHON" -m venv .venv-gps
.venv-gps/bin/python -m pip install --upgrade pip
.venv-gps/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv-gps/bin/python -m pip install -r shadow-lab/gps/requirements.txt
.venv-gps/bin/python -m pip check
.venv-gps/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
.venv-gps/bin/python shadow-lab/gps/download.py
