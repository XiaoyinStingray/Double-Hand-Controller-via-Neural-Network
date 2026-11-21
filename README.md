# Double-Hand Controller via Neural Network

基于**目标检测 + 关键点序列建模**的实时手势操控系统：单手浏览网页，双手玩小游戏。

> 《深度学习实践》大作业 · 进阶题 1 —— 基于目标检测的手势操控

## 系统概览

```
摄像头 → YOLO11n-pose（手框 + 21 关键点）→ 多手跟踪 → 关键点序列特征
       → Gesture Transformer → 时序后处理 → 模式仲裁
       → OS 鼠标控制（单手） / WebSocket 事件 → HTML5 游戏（双手）
```

- **单手模式**：食指指向移动光标、单指点击左键、双指右键、手掌上下滚轮、左右挥前进/后退、握拳急停
- **双手模式**：双手各控一个挡板玩接球/打砖块，动态手势触发道具
- **部署**：ONNX Runtime / TensorRT FP16，目标 ≥30 FPS

## 快速开始

> 环境与数据准备脚本将在 P1 阶段补充。

```bash
git clone git@github.com:XiaoyinStingray/Double-Hand-Controller-via-Neural-Network.git
cd Double-Hand-Controller-via-Neural-Network
python -m venv .venv
# 依赖见 requirements.txt（待补充）
```

## 数据集

- [IPN Hand](https://gibranbenitez.github.io/IPN_Hand/)：200 段视频、50 人、13 类手势，CC BY 4.0
- [IPN HandS](https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md)：逐帧人工校正的 21 点关键点 + 细化时序标注
- 自采数据：3 个会话、多背景/光照，用于域适应与端到端评测

**数据集与模型权重不入库**，通过下载脚本 + 校验和还原，详见 `docs/plan.md`。

## 仓库结构

```
configs/       训练与运行配置（YAML）
scripts/       数据下载、抽帧、标签转换、训练、评估、导出
src/
  datasets/    数据集与标签解析
  models/      检测/姿态、手势分类器
  tracking/    多手跟踪
  features/    关键点归一化与序列特征
  postprocess/ 时序平滑、状态机
  control/     鼠标控制、模式仲裁、WebSocket 协议
  runtime/     摄像头循环、推理引擎
apps/
  web/         演示网页与双手小游戏
  server/      本地服务（FastAPI + WebSocket）
docs/          方案、实验记录、报告与图表
tests/         单元测试
```

## 许可

代码 MIT。