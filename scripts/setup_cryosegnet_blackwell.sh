#!/usr/bin/env bash
# Build a Blackwell-compatible (sm_120) env for CryoSegNet INFERENCE — using uv
# (no conda: CryoSegNet's pinned CUDA-11.8 conda env can't run on RTX PRO 6000
# Blackwell, and conda create hit channel/ToS issues). torch cu128 + segment-anything
# + the libs CryoSegNet imports; it still uses its own repo modules (utils/, models/).
#
#   VENV=/workspace/cs_bw_venv bash scripts/setup_cryosegnet_blackwell.sh
set -uo pipefail
VENV="${VENV:-/workspace/cs_bw_venv}"
export PATH="$HOME/.local/bin:$PATH"

echo "[1/3] uv venv $VENV (python 3.10)"
uv venv "$VENV" --python 3.10 2>&1 | tail -3

echo "[2/3] torch cu128 (Blackwell sm_120)"
uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 torch torchvision 2>&1 | tail -3

echo "[3/3] CryoSegNet runtime deps (PyPI)"
uv pip install --python "$VENV/bin/python" \
    segment-anything opencv-python-headless mrcfile matplotlib scipy pandas numpy \
    tqdm pyyaml scikit-image timm wandb 2>&1 | tail -3

echo "[verify] torch + a real CUDA matmul on this GPU"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    x = torch.randn(256, 256, device="cuda")
    print("CUDA_MATMUL_OK", isinstance((x @ x).sum().item(), float))
else:
    print("CUDA_UNAVAILABLE")
PY
echo "CS_BW_DONE venv=$VENV"
