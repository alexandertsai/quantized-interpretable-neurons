#!/usr/bin/env bash
set -eu
SPARSITY_WORK_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SPARSITY_WORK_DIR"
export HF_HOME="$SPARSITY_WORK_DIR/hf-cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
.venv/bin/python -u experiment.py --precision bf16 --windows 8 --length 256
.venv/bin/python -u experiment.py --precision fp8 --windows 8 --length 256
echo PILOT_FINISHED
