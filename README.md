# 手势控制的鼠标与双手小游戏

《深度学习实践》大作业，进阶题 1：基于目标检测的手势操控。

用摄像头实时检测手部、识别手势，单手控制鼠标，双手控制一个小游戏。
目前完成 V0 规则鼠标、V1 手部检测/关键点、V2 手势分类；单/双手模式切换、
5 点校准、自采数据微调和最终部署还在做。

## 目标管线

```
摄像头 → YOLO11n-pose（手框 + 21 关键点）→ IoU 跟踪
       → 关键点特征 / 滑窗 → GRU 手势分类
       → 几何规则（指向/捏合/滚轮/握拳）
       → 鼠标控制 / 双手游戏
```

## 目前结果

| 模块 | 数据 | 指标 |
| --- | --- | --- |
| V1 手部检测 | IPN HandS val，1500 图 / 1318 只手 | bbox mAP50 0.979，mAP50-95 0.822 |
| V1 关键点 | 同上 | pose mAP50 0.824，mAP50-95 0.578 |
| V2 手势分类（GRU） | IPN HandS，30 fps 关键点 | GT macro-F1 0.642，MediaPipe 0.591 |
| V2 连续评估（GRU） | val，smooth 9 / min_len 20 | micro-F1 0.726，误触发 2.46 次/分钟 |
| V0 规则鼠标 | 本机摄像头 640×480 | 约 30 FPS（单独运行） |
| GRU ONNX | batch=1，CPU | 0.38 ms |

更完整的模型对比、GT→MediaPipe 掉点、连续指标和复现命令见 [`RESULTS.md`](RESULTS.md)。

## 目录

```
src/
  datasets/   IPN HandS 关键点、滑窗数据集
  models/     GRU / Transformer / TCN / MLP
  tracking/   IoU 跟踪 + EMA
  control/    光标映射、One-Euro、鼠标动作
scripts/      数据下载、抽帧、标签转换、训练、评估、预览、集群
```

数据和模型权重不进仓库。

## 环境

Python 3.9 + torch 2.3.0 (cu121)。Windows 本地：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
conda activate hand
python scripts/check_env.py --require-app
```

Linux 训练：

```bash
bash scripts/setup_env.sh hand
```

## 数据

- [IPN Hand](https://gibranbenitez.github.io/IPN_Hand/)：200 段视频、50 人、13 类手势，CC BY 4.0
- [IPN HandS](https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md)：逐帧人工校正 21 点关键点 + 时序标注

```bash
python scripts/download_data.py
python scripts/audit_ipn.py --landmarks data/raw/IPN_HandS_landmarks_GT
python scripts/extract_frames.py --split train --stride 3
python scripts/build_gesture_dataset.py
python scripts/build_pose_dataset.py
```

`ffmpeg` 建议放在 `tools/ffmpeg/`（已忽略），或在 `PATH` 中。

## 运行

规则鼠标（先 dry-run，确认手势后再实际控制）：

```bash
python scripts/v0_handmouse.py --dry-run
python scripts/v0_handmouse.py
```

V1 手部检测/关键点预览：

```bash
python scripts/preview_pose.py
```

训练 / 评估：

```bash
python scripts/train_gesture.py --model gru --epochs 40
python scripts/eval_gesture.py --checkpoint runs/gesture_gru/best.pt --data data/gesture --split val
python scripts/eval_continuous.py --preds runs/eval/gru_gt_val/preds.npz --data data/gesture --split val
```

V1 训练命令和集群脚本见 `scripts/cluster/`。

## 已知问题

- V1 在 IPN（第三人称、他人拍摄）上训练，本人摄像头存在域差；自采数据微调尚未完成。
- V2 的 15 类是 IPN 手势，不含“捏合/开掌/握拳”；鼠标点击/滚轮目前用几何规则，属于 hybrid 方案。
- 手心/手背朝向相反时，关键点语义可能镜像；跟踪能压位置抖动，压不了语义。
- 5 点校准未完成，当前光标使用固定中央区域映射。
- test split 尚未使用，所有指标均为 val。

## 许可

代码 MIT。IPN Hand / IPN HandS 数据遵循 CC BY 4.0，引用见论文。
