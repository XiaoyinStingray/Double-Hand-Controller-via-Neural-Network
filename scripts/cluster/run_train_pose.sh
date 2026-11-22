#!/bin/bash
# Runs inside the PBS container. Extract the dataset to /dev/shm (fast) and train.
set -euo pipefail

TAR=/gdata1/zhaojy/datasets/ipn_pose/pose_bundle.tar
WORK=/dev/shm/ipn_pose
REPO=/ghome/zhaojy/tasks/HandGesture
PY=/gdata1/zhaojy/envs/hand/bin/python

rm -rf "$WORK"
mkdir -p "$WORK"
echo "[$(date)] extracting $TAR"
tar -xf "$TAR" -C "$WORK"
# the bundle YAML points at /gdata1; rewrite it to the tmpfs copy
sed -i "s#^path:.*#path: $WORK#" "$WORK/pose.yaml"
echo "[$(date)] data ready: $(find "$WORK/images" -type f | wc -l) images"

"$PY" -u "$REPO/scripts/train_pose.py" \
    --data "$WORK/pose.yaml" \
    --model "$WORK/weights/yolo11n-pose.pt" \
    --epochs 100 \
    --imgsz 640 \
    --batch 32 \
    --device 0 \
    --workers 8 \
    --project /gdata1/zhaojy/exp/hand_pose \
    --name full \
    --predict 0

echo "[$(date)] training done"
