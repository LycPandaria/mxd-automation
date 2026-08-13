# MXD Automation

冒险岛怀旧服自动化辅助工具。基于 YOLO 目标检测 + A* 全局寻路，自动探索建图、寻怪、攻击、加血、加蓝。

## 架构

三层架构，职责分离：

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  感知层      │ →  │  MapExplorer  │ →  │  GlobalMap   │
│ YOLO/OCR/HP │    │  (SLAM 建图)  │    │  (持久化)    │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
┌─────────────┐    ┌──────────────┐           │
│  ActionExec │ ←  │ DecisionEngine│ ← A*Pathfinder │
│  (方向键)   │    │  (决策)       │    (寻路)      │
└─────────────┘    └──────────────┘    └──────────────┘
```

- **感知层** (`src/perception/`)：只负责"看"，输出结构化数据
  - `screen_capture` — 高性能窗口截图（Win32 `PrintWindow`）
  - `yolo_detector` — YOLO 目标检测（怪物、地板、绳索、玩家）
  - `hp_mp_detector` — 血/蓝条百分比检测（多方法融合）
  - `ocr_name_locator` — RapidOCR 角色名字识别与坐标定位
  - `template_matcher` — 模板匹配（占位）

- **决策层** (`src/decision/`)：只负责"想"，输出动作指令
  - `context` — 共享上下文 + 决策引擎（加血/加蓝/寻路/技能）
  - `global_map` — 全局地图（稀疏网格，SLAM 式逐帧拼接，持久化 JSON）
  - `map_explorer` — 地图探索器（追踪角色位移，拼接多帧 YOLO 数据建图）
  - `astar` — A* 寻路（在 GlobalMap 上搜索最短路径，输出方向键指令）
  - `fsm` — 有限状态机（占位）

- **执行层** (`src/execution/`)：只负责"做"，模拟人类操作
  - `keyboard_controller` — 键盘模拟（PostMessage 注入）
  - `mouse_controller` — 鼠标模拟（PostMessage 注入）
  - `action_executor` — 动作执行器（聚合键鼠，带冷却控制）

## 核心流程

### 探索建图（类似扫地机器人 SLAM）

```
角色在地图里走动 → 每帧 YOLO 检测 floors/ropes
  → MapExplorer 追踪位移 + 拼接到 GlobalMap
  → 保存为 assets/maps/{map_id}.json
```

探索和打怪可以**同时进行**：一边走路一边建图，遇到怪物照样打。

### 实时战斗

```
YOLO 检测到怪物 → 自身坐标 + 怪物坐标 → 窗口坐标转全局坐标
  → A* 在 GlobalMap 上搜索路径 → 方向键指令（← → ↑ ↓ 跳）
  → 到达攻击范围 → 按 Tab 选怪 → 轮转释放技能
```

寻路失败时自动回退到"Tab 选同平台怪 + 技能"的简单模式。

### 自身定位（两级策略）

1. **HP 条偏移**（最准）：HP 条底部 + 偏移量 → 脚底坐标
2. **RapidOCR 名字识别**（兜底）：识别脚底灰色名字 → 计算中心坐标

## 配置

双层配置策略：

- `config/default.yaml` — 全局默认值（勿手动修改）
- `config/user.json` — 用户运行时配置（自动覆盖 default.yaml）

坐标自适应：所有区域参数按参考分辨率 (1366×768) 配置，运行时根据实际窗口尺寸自动缩放。

## 运行

```bash
# GUI 模式
python main.py

# CLI 模式（无界面）
python -m src.main
```

## 训练（可选）

```bash
# 1. 准备数据集（80/20 划分训练集/验证集）
python train/scripts/01_prepare_dataset.py

# 2. 训练 YOLO
python train/scripts/02_train_yolo.py
```

训练完成后，`train/runs/exp/weights/best.pt` 会自动复制到 `assets/models/`。

## 项目结构

```
mxd-automation/
├── main.py                  # 入口（GUI 启动）
├── config/
│   ├── default.yaml         # 默认配置
│   └── user.json            # 用户配置
├── assets/
│   ├── models/              # YOLO 模型文件
│   └── maps/                # 探索生成的地图文件 (*.json)
├── src/
│   ├── main.py              # CLI 入口
│   ├── perception/          # 感知层
│   ├── decision/            # 决策层
│   ├── execution/           # 执行层
│   └── utils/               # 工具（配置加载、日志、几何计算）
├── train/
│   ├── data/                # 训练数据
│   ├── scripts/             # 训练脚本
│   └── model/               # 预训练模型
└── ocr_demo.py              # RapidOCR 演示工具
```