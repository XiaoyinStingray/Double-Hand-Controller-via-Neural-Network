#!/usr/bin/env bash
# Set up the training environment on a Linux (cluster) machine.
#
#   bash scripts/setup_env.sh              # core deps only (headless training)
#   bash scripts/setup_env.sh hand --app   # also install desktop/demo deps
#
# Notes:
#   - Compute nodes often have no internet: run this once on a login node,
#     or pre-download wheels with `pip download -r requirements.txt -d wheels/`.
#   - Adjust the CUDA wheel index if the cluster ships a different driver.

set -euo pipefail

ENV_NAME="${1:-hand}"
WITH_APP="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda create -y -n "$ENV_NAME" python=3.9
    fi
    conda activate "$ENV_NAME"
else
    echo "[warn] conda not found, falling back to python venv" >&2
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

if [ "$WITH_APP" = "--app" ]; then
    python -m pip install -r requirements-app.txt
else
    python -m pip install -r requirements.txt
fi

python scripts/download_models.py
python scripts/check_env.py ${WITH_APP:+--require-app}

echo
echo "Done. Activate with: conda activate ${ENV_NAME}"
