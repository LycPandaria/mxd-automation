# MXD Automation

冒险岛怀旧服自动化辅助工具。基于 YOLO 目标检测 + 反应式决策引擎，实现自动打怪、加血、加蓝、同平台追击、攀爬绳索。

## 架构

三层架构，职责分离：

```
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  感知层           │ →  │  决策层               │ →  │  执行层           │
│ YOLO 目标检测     │    │  DecisionEngine      │    │  ActionExecutor  │
│ HP/MP 颜色检测    │    │  反应式决策 + FSM     │    │  键盘/鼠标注入    │
│ OCR 名字定位      │    │  同平台追击/攀爬/探索  │    │  按键冷却控制    │
└──────────────────┘    └──────────────────────┘    └──────────────────┘
```

- **感知层** (`src/perception/`)：只负责"看"，输出结构化数据
  - `screen_capture` — 高性能窗口截图（Win32 `PrintWindow`）
  - `yolo_detector` — YOLO 目标检测（怪物、地板、绳索）
  - `hp_mp_detector` — 血/蓝条百分比检测（颜色数像素 + 边缘检测）
  - `ocr_name_locator` — RapidOCR 角色名字识别与坐标定位（每帧实时执行）
  - `template_matcher` — 模板匹配（预留）

- **决策层** (`src/decision/`)：只负责"想"，输出动作指令
  - `context` — 共享上下文 + 反应式决策引擎（同平台追击/攀爬/下落/探索/技能）
  - `fsm` — 有限状态机（IDLE / CHASING / ATTACKING / CLIMBING / DROPPING / HEALING / RECOVERING）
  - `distance` — 路径距离推算器（同层判定、绳索路径、平台跳跃路径）

- **执行层** (`src/execution/`)：只负责"做"，模拟人类操作
  - `keyboard_controller` — 键盘模拟（SendInput 真实按键 / PostMessage 后台注入）
  - `mouse_controller` — 鼠标模拟（PostMessage 注入）
  - `action_executor` — 动作执行器（聚合键鼠，带冷却控制）

## 核心流程

### 反应式决策（每帧）

```
截图 → YOLO 检测 → 按类别过滤（怪物/地板/绳索）
  → HP/MP 检测 → OCR 自身定位
  → 组装 Context → DecisionEngine.decide()
```

决策优先级从高到低：
1. HP 低于阈值 → 加血
2. MP 低于阈值 → 加蓝
3. 检测到怪物 → 同平台走过去攻击 / 攀爬绳索 / 跨平台跳跃
4. 没怪 → 往一个方向探索移动

### 战斗流程

```
YOLO 检测到怪物 → 筛选同平台怪物（脚底 Y 差 ≤ attack_range_y）
  → 选最近目标 → 按住方向键走过去
  → 进入攻击范围 → 松开方向键，调整朝向 → 轮转释放技能
  → 怪物死亡/消失 → 自动换下一只
```

### 自身定位

**RapidOCR 文字识别**（唯一方案，每帧实时执行）

- 搜索区域：画面 10%~92%（排除底部 8% UI 面板，避免匹配到固定位置的 UI 角色名）
- 识别角色名字（如"我是立立"）→ 名字中心 → 脚底坐标 → 角色中心点
- 与 YOLO 检测同节奏，每帧独立执行，无缓存兜底

## 配置

双层配置策略：

- `config/user.yaml` — 用户运行时配置（优先加载）
- `config/user.json` — 向后兼容的 JSON 格式
- 内置默认值 — 未配置的字段使用代码内置默认值

坐标自适应：HP/MP 区域参数按参考分辨率 (1366×768) 配置，运行时根据实际窗口尺寸自动缩放。

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `window_title` | `冒险岛怀旧服` | 游戏窗口标题 |
| `start_stop_hotkey` | `F6` | 启动/停止热键 |
| `fps` | `13` | 目标帧率 |
| `keyboard_mode` | `sendinput` | 按键模式：sendinput（前台）/ postmessage（后台） |
| `confidence` | `0.5` | YOLO 检测置信度阈值 |
| `model_path` | 模型路径 | YOLO 模型文件路径 |
| `self_name` | `我是立立` | 角色名字（OCR 定位用） |
| `attack_range` | `100` | 攻击范围（像素），进入此范围后开始攻击 |
| `attack_range_y` | `80` | 同平台垂直容差（像素） |
| `skills` | 技能列表 | `[{name, key, cooldown}, ...]` |

## 运行

```bash
# 安装依赖
pip install -r requirements.txt

# GUI 模式（推荐）
python main.py

# CLI 模式（无界面）
python -m src.main
```

## 训练

详见 [train/README.md](train/README.md)。

### 数据标注

1. 将截图放入 `train/data/raw/` 目录
2. 使用 LabelImg 等工具标注，保存为 Pascal VOC XML 格式
3. 标注的 XML 文件与图片同名，放在同一目录下

类别定义（共 3 类）：`floor`（地板）、`monster`（怪物）、`rope`（绳索）

### 自动标注

```bash
python train/scripts/01_auto_annotate/main.py
```

基于少量手动标注训练模型，自动标注剩余图片。

### 正式训练

```bash
python train/scripts/02_train_yolo/main.py
```

全量数据训练 YOLO 模型，输出 `best.pt`。

## 项目结构

```
mxd-automation/
├── main.py                  # 入口（GUI 模式，PyQt5）
├── requirements.txt         # 依赖列表
├── config/
│   ├── default.yaml         # 默认配置（勿手动修改）
│   ├── user.yaml            # 用户配置（运行时覆盖默认值）
│   └── user.json            # 用户配置（向后兼容）
├── src/
│   ├── main.py              # CLI 入口
│   ├── perception/          # 感知层
│   │   ├── screen_capture.py    # 窗口截图
│   │   ├── yolo_detector.py     # YOLO 目标检测
│   │   ├── hp_mp_detector.py    # HP/MP 检测
│   │   ├── ocr_name_locator.py  # OCR 名字定位
│   │   └── template_matcher.py  # 模板匹配（预留）
│   ├── decision/            # 决策层
│   │   ├── context.py           # 决策引擎 + 上下文
│   │   ├── fsm.py               # 有限状态机
│   │   └── distance.py          # 路径距离推算
│   ├── execution/           # 执行层
│   │   ├── action_executor.py   # 动作执行器
│   │   ├── keyboard_controller.py  # 键盘模拟
│   │   └── mouse_controller.py     # 鼠标模拟
│   └── utils/               # 工具
│       ├── config_loader.py     # 配置加载
│       ├── geometry.py          # 几何计算
│       └── logger.py            # 日志
├── train/
│   ├── data/
│   │   ├── raw/                 # 原始图片 + XML 标注
│   │   └── data.yaml            # 数据集类别配置
│   ├── scripts/
│   │   ├── 01_auto_annotate/    # 自动标注脚本
│   │   └── 02_train_yolo/       # 正式训练脚本
│   └── model/                   # 预训练模型
└── ui/                      # PyQt5 GUI 界面
```