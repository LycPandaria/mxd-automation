# MXD Automation

冒险岛怀旧服自动化辅助工具。基于 YOLO 目标检测 + 反应式决策引擎，实现自动打怪、加血、加蓝、同平台追击、攀爬绳索。

## 架构

三层架构，职责分离：

```
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  感知层           │ →  │  决策层               │ →  │  执行层           │
│ YOLO 目标检测     │    │  DecisionEngine      │    │  ActionExecutor  │
│ HP/MP 颜色检测    │     │  反应式决策 + FSM     │    │  键盘/鼠标注入     │
│ OCR 名字定位      │     │  同平台追击/攀爬/探索  │     │  按键冷却控制      │
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

- `config/user.yaml` — 用户运行时配置
- 内置默认值 — 未配置的字段使用代码内置默认值

坐标自适应：HP/MP 区域参数按参考分辨率 (1366×768) 配置，运行时根据实际窗口尺寸自动缩放。

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `window_title` | `冒险岛怀旧服` | 游戏窗口标题 |
| `start_stop_hotkey` | `F6` | 启动/停止热键 |
| `fps` | `13` | 目标帧率 |
| `keyboard_mode` | `sendinput` | 按键模式（固定使用 SendInput 驱动层注入） |
| `confidence` | `0.5` | YOLO 检测置信度阈值 |
| `model_path` | 模型路径 | YOLO 模型文件路径 |
| `self_name` | `我是立立` | 角色名字（OCR 定位用） |
| `attack_range` | `100` | 攻击范围（像素），进入此范围后开始攻击 |
| `attack_range_y` | `80` | 同平台垂直容差（像素） |
| `skills` | 技能列表 | `[{name, key, cooldown}, ...]` |

## 注意事项

### 游戏窗口要求

- **必须使用窗口模式**，不支持全屏独占 DirectX 模式。全屏模式下截图会失败（BitBlt 只能截取窗口客户区），需要改用 `mss` 或 `dxcam` 方案。
- **窗口标题**必须与配置中的 `window_title` 一致（默认 `冒险岛怀旧服`），否则程序找不到游戏窗口。
- **窗口必须保持前台激活状态**（最上层），因为冒险岛通过 `DirectInput` 读取全局键盘状态，SendInput 的按键只有窗口在前台时才能被游戏接收。运行期间不能切换去操作其他程序。
- 游戏窗口建议分辨率为 **1366×768**（参考分辨率），程序支持自动缩放适配不同分辨率，但 HP/MP 区域坐标需按参考分辨率标定。

### 分辨率与坐标自适应

- HP/MP 区域、`self_offset` 等坐标参数全部按 **参考分辨率 1366×768** 配置（代码内置默认值，可在 `config/user.yaml` 中覆盖）。
- 运行时，程序会根据实际窗口客户区尺寸自动等比例缩放坐标参数。
- 如果你的游戏窗口分辨率不同，**先用 UI 的"框选 HP 区域"功能重新标定**，程序会自动计算并保存为归一化坐标（0~1 之间），下次启动时自动适配。

### 角色名配置（必须）

- **必须配置 `self_name`**（你的角色名字，如 `我是立立`），否则 OCR 定位无法工作，程序无法知道自己的位置。
- 角色名必须与游戏内显示的名字**完全一致**（精确匹配），不能包含额外空格或特殊字符。
- OCR 搜索区域为画面 10%~92%（排除底部 8% UI 面板），避免匹配到聊天框或 UI 固定位置的文字。

### HP/MP 区域标定

- 首次使用必须用 UI 中的"框选 HP 区域"和"框选 MP 区域"功能标定血条/蓝条的位置和颜色。
- 标定时确保游戏窗口处于**前台可见**状态，且血条/蓝条完整显示（未被其他窗口遮挡）。
- 如果更换装备、升级、或游戏 UI 发生变化导致血条位置/颜色改变，需要重新标定。

### YOLO 模型

- 必须提供有效的 YOLO `.pt` 模型文件，并在配置中指定 `model_path`（可在 UI 中设置）。
- 模型文件不存在时，程序会自动回退到 **MockDetector**（模拟检测器），仅用于测试框架逻辑，不会真正检测怪物。
- 模型需要支持 3 个类别：`monster`（怪物）、`floor`（地板）、`rope`（绳索）。
- 训练方法详见 [train/README.md](train/README.md)。

### 按键模式

使用 Win32 `SendInput` API 从驱动层模拟真实全局按键。冒险岛通过 `DirectInput` / `GetAsyncKeyState` 读取全局键盘状态，**只有此模式能正常工作**。

> **重要**：游戏窗口必须处于**前台激活状态**（最上层），运行期间不能切换去操作其他程序，否则按键无法注入到游戏中。这是 SendInput 的工作机制决定的，不是程序 bug。

### Python 依赖注意事项

- **onnxruntime 必须使用 1.20.1 版本**（`requirements.txt` 已锁定）：
  - 1.28.x 与 PyQt5 在 Windows 上存在 DLL 加载冲突（报错"DLL 初始化例程失败"）。
  - 1.17.x 需要 `numpy<2`，与 `numpy 2.x` 不兼容。
- 安装依赖时请严格按照 `requirements.txt` 安装，不要手动升级 onnxruntime。

### 高 DPI 显示器

- 程序在启动时自动调用 `SetProcessDPIAware()` 设置 DPI 感知，确保 BitBlt 截图使用物理像素。
- 同时设置了 `QT_SCALE_FACTOR=1` 禁用 Qt 内部 DPI 缩放，避免 UI 坐标与截图坐标不一致。
- 如果你的显示器缩放比例 ≠ 100%，屏幕坐标、鼠标坐标、截图坐标已全部统一为物理像素，无需额外配置。

### 其他注意事项

- 程序不修改游戏内存，仅通过**键盘/鼠标模拟**操作游戏，但使用过程仍有一定风险，请自行评估。
- 建议在**小号或测试服**上先试用，确认功能正常再用于正式角色。
- 游戏更新可能导致 UI 布局变化，需要重新标定 HP/MP 区域。
- 不要同时运行多个辅助实例，避免多个程序同时操作同一个窗口导致冲突。

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
│   └── user.yaml            # 用户配置（覆盖代码默认值）
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