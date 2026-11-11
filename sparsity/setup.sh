#!/usr/bin/env bash
set -eu
PILOT_WORK_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PILOT_WORK_DIR"
python -m pip install --no-cache-dir --target "$PILOT_WORK_DIR/bootstrap" virtualenv
PYTHONPATH="$PILOT_WORK_DIR/bootstrap" python -m virtualenv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir 'https://github.com/huggingface/transformers/archive/e4e4e98845fcf92e2b75f709f356fbc838b55d11.tar.gz' accelerate datasets kernels pillow
.venv/bin/python -m pip freeze > requirements-freeze.txt
.venv/bin/python - <<'PY'
import torch,transformers,kernels
print('ENV_READY',torch.__version__,transformers.__version__,torch.cuda.get_device_name(),flush=True)
from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
k=load_finegrained_fp8_kernel()
print('FP8_KERNEL_LOADED',k,flush=True)
from transformers.integrations.finegrained_fp8 import fp8_grouped_matmul
x=torch.randn(512,2048,device='cuda',dtype=torch.bfloat16)
w=torch.randn(4,1024,2048,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
s=torch.ones(4,8,16,device='cuda')
offsets=torch.tensor([128,256,384,512],device='cuda',dtype=torch.int32)
counts=torch.tensor([128,128,128,128],device='cuda',dtype=torch.int32)
y=fp8_grouped_matmul(x,w,s,offsets,counts,[128,128])
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('FP8_GROUPED_KERNEL_SMOKE_OK',y.shape,y.dtype,flush=True)
PY
