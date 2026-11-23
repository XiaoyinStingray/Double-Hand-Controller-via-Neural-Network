# 实验结果

最后更新：2026-11-23。所有指标均为 IPN HandS 官方 val（test 封存）。

## 硬件

| 用途 | 配置 |
| --- | --- |
| 本地训练/推理 | RTX 4060 Laptop 8 GB，Python 3.9，torch 2.3.0+cu121 |
| 集群训练 | 2080Ti 11 GB（PBS 单卡），torch 2.7.0+cu118 |

## 数据

| 项 | 值 |
| --- | --- |
| IPN HandS | 200 视频，50 人，801,476 帧，15 类（14 手势 + 非手势） |
| 划分 | 148 train / 52 test（视频级）；训练时再从 train 划 133 / 15 做 train/val |
| 类别分布 | NOG 32.8%，P1F 27.7%，P2F 28.1%，其余 11 类合计 11.4% |
| 检测/姿态抽帧 | 10 fps（stride 3 on 30 fps），26,600 train / 1,500 val |
| 序号分类窗口 | T=16，stride 2，399,287 个窗口 |

## V1：YOLO11n-pose

训练：100 epochs，imgsz 640，batch 16，AMP，2080Ti，约 6.4 小时。

| 指标 | bbox | pose |
| --- | ---: | ---: |
| Precision | 0.968 | 0.907 |
| Recall | 0.931 | 0.805 |
| mAP50 | 0.979 | 0.824 |
| mAP50-95 | 0.822 | 0.578 |

- 本地 `model.val` 复现集群结果，误差 < 0.001。
- 4060 上单帧推理约 2.2 ms；摄像头预览约 30 FPS。
- 权重：`runs/pose/full/best.pt`（6 MB）。

## V2：手势分类

输入：每帧 85 维（21×2 位置 + 21×2 速度 + presence），腕原点/掌长归一化，左手镜像。

### 模型对比（val，GT 关键点）

| 模型 | 参数量 | acc | macro-F1 | MediaPipe acc | MediaPipe F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP-frame | ~29 K | 0.816 | 0.523 | 0.749 | 0.460 |
| TCN | 211 K | 0.886 | 0.629 | 0.845 | 0.579 |
| Transformer | 286 K | 0.891 | 0.622 | 0.867 | 0.583 |
| **GRU** | 510 K | **0.900** | **0.642** | **0.864** | **0.591** |

### 连续评估（min_len 20，smooth 9）

| 模型 | conf | precision | recall | micro-F1 | FP/分钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.3 | 0.493 | 0.625 | 0.551 | 7.23 |
| Transformer | 0.3 | 0.692 | 0.681 | 0.686 | 3.41 |
| Transformer | 0.4 | 0.713 | 0.668 | 0.689 | 3.02 |
| **GRU** | 0.3 | 0.760 | 0.694 | **0.726** | 2.46 |
| **GRU** | 0.4 | 0.837 | 0.623 | 0.714 | **1.36** |

### ONNX（batch=1，CPU，200 次）

| 模型 | 大小 | max abs diff | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.12 MB | 1.4e-6 | 0.059 ms | 0.063 ms |
| Transformer | 1.20 MB | 2.4e-6 | 0.428 ms | 0.452 ms |
| GRU | 2.04 MB | 1.7e-6 | 0.383 ms | 0.618 ms |

## 复现

```bash
# 手势分类
python scripts/build_gesture_dataset.py
python scripts/train_gesture.py --model gru --epochs 40
python scripts/eval_gesture.py --checkpoint runs/gesture_gru/best.pt \
    --data data/gesture --split val
python scripts/eval_continuous.py --preds runs/eval/gru_gt_val/preds.npz \
    --data data/gesture --split val --min-len 20 --smooth 9 --conf 0.3
python scripts/export_gesture_onnx.py --checkpoint runs/gesture_gru/best.pt

# 检测/姿态
python scripts/build_pose_dataset.py --out data/pose
python scripts/train_pose.py --data data/pose/pose.yaml --epochs 100 --batch 16
```

## 说明

- 类别不平衡明显，训练使用类别加权 CE；评估用 macro-F1 和每类指标，不只报 accuracy。
- 4 个 IPN 视频的 AVI 流有损坏，10 fps 抽帧尾部少 3–4 帧，已单独标记，不影响关键点序列。
- IPN HandS 的 MediaPipe Hands 文件在 Google Drive 已 404，改用 MP Holistic 的 hand_landmarks 作为噪声基线。
- 当前规则鼠标、hybrid 集成和自采数据实验尚未纳入本表。
