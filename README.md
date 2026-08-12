# MXD Automation

冒险岛怀旧服自动化辅助工具。基于 YOLO 目标检测，自动寻怪、攻击、加血、加蓝。

## 架构

三层架构，职责分离：

- **感知层** (`src/perception/`)：只负责"看"，输出结构化数据
  - `screen_capture` — 高性能窗口截图（Win32 PrintWindow）
  - `yolo_detector` — YOLO 目标检测
  - `hp_mp_detector` — 血/蓝条百分比检测
  - `template_matcher` — 模板匹配（占位）

- **决策层** (`src/decision/`)：只负责"想"，输出动作指令
  - `context` — 共享上下文 + 决策引擎
  - `fsm` — 有限状态机（占位）
  - `astar` — A* 拓扑寻路（占位）

- **执行层** (`src/execution/`)：只负责"做"，模拟人类操作
  - `keyboard_controller` — 键盘模拟（PostMessage 注入）
  - `mouse_controller` — 鼠标模拟（PostMessage 注入）
  - `action_executor` — 动作执行器（聚合键鼠）

## 配置

双层配置策略：

- `config/default.yaml` — 全局默认值（勿手动修改）
- `config/user.json` — 用户运行时配置（自动覆盖 default.yaml）

## 运行

```bash
# GUI 模式
python main.py

# CLI 模式（无界面）
python -m src.main
```

## 训练（可选）

```bash
# 1. 准备数据集
python train/scripts/01_prepare_dataset.py

# 2. 训练 YOLO
python train/scripts/02_train_yolo.py
```

训练完成后，`train/runs/exp/weights/best.pt` 会自动复制到 `assets/models/`。
