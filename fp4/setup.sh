#!/usr/bin/env bash
set -eu
FP4_WORK_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$FP4_WORK_DIR"
python -m pip install --no-cache-dir --target "$FP4_WORK_DIR/bootstrap" virtualenv
PYTHONPATH="$FP4_WORK_DIR/bootstrap" python -m virtualenv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir 'https://github.com/huggingface/transformers/archive/e4e4e98845fcf92e2b75f709f356fbc838b55d11.tar.gz' accelerate datasets kernels pillow compressed-tensors==0.20.1
.venv/bin/python -m pip freeze > requirements-freeze.txt
.venv/bin/python prepare_data.py
.venv/bin/python run.py preflight
.venv/bin/python run.py bf16
.venv/bin/python run.py fp4
echo ALL_EXPERIMENTS_DONE
