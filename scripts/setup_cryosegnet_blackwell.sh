#!/usr/bin/env bash
# Build a Blackwell-compatible (sm_120) conda env for CryoSegNet INFERENCE.
# CryoSegNet's own environment.yml pins CUDA 11.8 / old torch, which cannot run on
# the RTX PRO 6000 Blackwell. This env uses torch cu128 + segment-anything + the
# libs CryoSegNet imports (it still uses its own repo modules: utils/, models/, etc).
#
#   MC=/workspace/miniconda3 bash scripts/setup_cryosegnet_blackwell.sh
set -uo pipefail
MC="${MC:-/workspace/miniconda3}"
ENV="${ENV:-cryosegnet_bw}"
source "$MC/etc/profile.d/conda.sh"
export PATH="$MC/bin:$PATH"

echo "[1/3] conda create $ENV (python 3.10)"
conda env list | grep -q "/$ENV\$" || conda create -n "$ENV" python=3.10 -y 2>&1 | tail -3

echo "[2/3] torch cu128 (Blackwell sm_120)"
conda run -n "$ENV" pip install -q --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision 2>&1 | tail -3

echo "[3/3] CryoSegNet runtime deps"
conda run -n "$ENV" pip install -q \
    segment-anything opencv-python-headless mrcfile matplotlib scipy pandas numpy \
    tqdm pyyaml scikit-image timm wandb 2>&1 | tail -3

echo "[verify] torch + a real CUDA matmul on this GPU"
conda run -n "$ENV" python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    x = torch.randn(256, 256, device="cuda")
    y = (x @ x).sum().item()
    print("CUDA_MATMUL_OK", isinstance(y, float))
else:
    print("CUDA_UNAVAILABLE")
PY
echo "CS_BW_DONE env=$ENV"
