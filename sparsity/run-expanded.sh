#!/usr/bin/env bash
set -eu
SPARSITY_WORK_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SPARSITY_WORK_DIR"
mkdir -p initial
if [ ! -f initial/results_bf16.json ]; then
    cp results_bf16.json results_fp8.json progress.jsonl environment.json tokens.json run.log initial/
    cp overlap_*.json initial/
fi
rm -f clean_bf16_*.pt clean_fp8_*.pt masks_*.pt overlap_*.json
rm -f results_bf16.json results_fp8.json
export HF_HOME="$SPARSITY_WORK_DIR/hf-cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
: > progress.jsonl
.venv/bin/python -u experiment.py --precision bf16 --windows 32 --length 512 --document-sampling
.venv/bin/python -u experiment.py --precision fp8 --windows 32 --length 512 --document-sampling
echo EXPANDED_PILOT_FINISHED
