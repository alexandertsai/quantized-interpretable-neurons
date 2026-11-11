#!/usr/bin/env bash
set -eu
FOLLOWUP_WORK_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$FOLLOWUP_WORK_DIR"
export HF_HOME="$FOLLOWUP_WORK_DIR/hf-cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
.venv/bin/python -u followup.py --precision bf16
.venv/bin/python -u followup.py --precision fp8
echo FOLLOWUP_GPU_FINISHED
