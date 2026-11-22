#!/bin/bash
# Create a thin venv that reuses the image's torch, then install ultralytics.
# Run on G101 inside the CUDA image:
#   startdocker ... -c "/bin/bash /ghome/zhaojy/tasks/HandGesture/scripts/cluster/setup_hand_env.sh"
set -euo pipefail

ENV=/gdata1/zhaojy/envs/hand
BASE=/opt/venv/lib/python3.11/site-packages

mkdir -p /gdata1/zhaojy/envs
if [ ! -x "$ENV/bin/python" ]; then
    /opt/venv/bin/python -m venv "$ENV"
fi

# make the image's torch/torchvision/opencv/etc. importable
echo "$BASE" > "$ENV/lib/python3.11/site-packages/00-base.pth"

# --no-deps: do not let pip replace the image's torch
"$ENV/bin/pip" install --no-cache-dir --no-deps \
    ultralytics==8.4.216 ultralytics-thop==2.2.5 polars==1.36.1

"$ENV/bin/python" - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
PY
