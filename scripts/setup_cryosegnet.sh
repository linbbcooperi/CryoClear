#!/usr/bin/env bash
# Heavy one-time CryoSegNet setup on the GPU box (separate from uv).
# Installs miniconda to $WORKSPACE, clones CryoSegNet, downloads weights, builds
# its conda env. Idempotent: safe to re-run. Logs progress with [n/4] markers.
#
#   WORKSPACE=/workspace bash scripts/setup_cryosegnet.sh
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
CS_DIR="$WORKSPACE/CryoSegNet"
MC="$WORKSPACE/miniconda3"

echo "[1/4] miniconda -> $MC"
if [ ! -x "$MC/bin/conda" ]; then
  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MC"
fi
export PATH="$MC/bin:$PATH"
source "$MC/etc/profile.d/conda.sh"

echo "[2/4] clone CryoSegNet -> $CS_DIR"
[ -d "$CS_DIR/.git" ] || git clone --quiet https://github.com/jianlin-cheng/CryoSegNet "$CS_DIR"

echo "[3/4] pretrained weights"
cd "$CS_DIR"
if [ ! -e pretrained_models ] && ! ls ./*.pth >/dev/null 2>&1; then
  curl -sL https://calla.rnet.missouri.edu/CryoSegNet/pretrained_models.tar.gz -o pretrained_models.tar.gz
  tar -xf pretrained_models.tar.gz && rm -f pretrained_models.tar.gz
fi
echo "    weights present:"; ls -1 "$CS_DIR" | grep -iE "pretrained|\.pth" | head

echo "[4/4] conda env from environment.yml"
ENV_NAME="$(grep -E '^name:' environment.yml | head -1 | awk '{print $2}')"
ENV_NAME="${ENV_NAME:-cryosegnet}"
if conda env list | grep -qE "^${ENV_NAME}\s|/${ENV_NAME}$"; then
  echo "    env '$ENV_NAME' already exists; skipping create"
else
  conda env create -f environment.yml 2>&1 | tail -8
fi

echo "[verify] torch + CUDA on this GPU"
conda run -n "$ENV_NAME" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1 | tail -5

echo "CS_SETUP_DONE env=$ENV_NAME"
