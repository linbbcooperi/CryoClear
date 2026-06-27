#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for CryoClear. Idempotent — safe to re-run.
#
#   bash scripts/setup_pod.sh                 # uv env + tests
#   bash scripts/setup_pod.sh --cryosegnet    # also clone CryoSegNet + download weights
#
# Run from the repo root, which should live on the Network Volume (e.g.
# /workspace/CryoClear) so it survives pod restarts. NO secrets are read or
# written by this script.
set -euo pipefail

WITH_CRYOSEGNET=0
[[ "${1:-}" == "--cryosegnet" ]] && WITH_CRYOSEGNET=1

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$REPO/.." && pwd)"
echo "==> Repo:      $REPO"
echo "==> Workspace: $WORKSPACE"

# 1) uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv $(uv --version)"

# 2) project env ------------------------------------------------------------
cd "$REPO"
echo "==> uv sync (env + deps + editable install)"
uv sync
echo "==> Running tests"
uv run pytest -q

# 3) CryoSegNet (optional; its own conda env) -------------------------------
if [[ "$WITH_CRYOSEGNET" == "1" ]]; then
  CS_DIR="$WORKSPACE/CryoSegNet"
  if [[ ! -d "$CS_DIR" ]]; then
    echo "==> Cloning CryoSegNet -> $CS_DIR"
    git clone https://github.com/jianlin-cheng/CryoSegNet "$CS_DIR"
  fi
  if [[ ! -e "$CS_DIR/pretrained_models" && ! -e "$CS_DIR"/*.tar.gz ]]; then
    echo "==> Downloading CryoSegNet pretrained weights"
    ( cd "$CS_DIR" && curl -L https://calla.rnet.missouri.edu/CryoSegNet/pretrained_models.tar.gz \
        -o pretrained_models.tar.gz && tar -xvf pretrained_models.tar.gz && rm -f pretrained_models.tar.gz )
  fi
  echo "==> CryoSegNet fetched. Create its conda env yourself (heavy, separate from uv):"
  echo "      cd $CS_DIR && conda env create -f environment.yml && conda activate cryosegnet"
  echo "      # then install the torch CUDA build that matches this pod (pytorch.org)"
fi

# 4) next steps -------------------------------------------------------------
POD_ID="${RUNPOD_POD_ID:-<POD_ID>}"
cat <<EOF

==> Done. Next:
  1. Download data:
       uv run python scripts/download_cryoppp.py --source cryoppp --empiar 10017 --n-micrographs 15
  2. (GPU) cache CryoSegNet picks:
       uv run python scripts/run_cryosegnet.py --cryosegnet-dir $WORKSPACE/CryoSegNet --empiar 10017
  3. Baseline metric + train junk model:
       uv run python scripts/run_baseline.py --backend cryosegnet --empiar 10017
       uv run python scripts/train_junk_classifier.py --empiar 10017
  4. Launch the UI (bind 0.0.0.0 so the RunPod proxy can reach it):
       make app
     Open:  https://${POD_ID}-8501.proxy.runpod.net
EOF
