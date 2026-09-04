# 训练模块

基于 YOLOv8n 的目标检测模型训练与自动标注。

## 目录结构

```
train/
├── data/
│   ├── raw/                  # 原始数据（截图 + Pascal VOC XML 标注），只读不写
│   └── data.yaml             # 数据集配置（类别定义）
├── model/
│   └── yolov8n.pt            # YOLOv8n 预训练权重
├── scripts/
│   ├── 01_auto_annotate/     # 自动标注：小样本训练 → 推理未标注图片 → 生成 XML
│   │   ├── main.py
│   │   └── auto_work/        # 临时训练目录，每次自动清空
│   └── 02_train_yolo/        # 正式训练：全量数据训练 YOLO 模型
│       ├── main.py
│       ├── auto_work/        # 临时训练目录，每次自动清空
│       │   ├── images/       # 训练集/验证集图片
│       │   ├── labels/       # YOLO 格式标注
│       │   └── data.yaml     # 自动生成的训练配置
│       ├── runs/             # 训练输出（多轮实验）
│       │   ├── train/            # 默认训练
│       │   ├── quick_A_默认/     # 实验 A：默认参数
│       │   ├── quick_B_关闭垂直翻转/  # 实验 B：关闭垂直翻转
│       │   ├── quick_C_降低HSV/      # 实验 C：降低 HSV 增强
│       │   └── quick_D_关闭旋转/     # 实验 D：关闭旋转增强
│       └── model/            # 最终产出的 best.pt
```

## 环境要求

- Python 3.12+
- CUDA 12.4（推荐 RTX 4060 及以上）
- 依赖：`pip install ultralytics torch torchvision`

## 数据标注

使用 LabelImg 等工具对 `data/raw/` 下的截图进行标注，保存为 **Pascal VOC XML** 格式。

类别定义（共 4 类）：

| ID | 类别名    | 说明 |
|----|----------|------|
| 0  | floor    | 地板/平台 |
| 1  | monster  | 怪物 |
| 2  | rope     | 绳索 |
| 3  | player   | 玩家（用于自身定位） |

## 脚本说明

### 1. 自动标注 (`01_auto_annotate/main.py`)

**适用场景**：只有少量手动标注，想快速扩大标注量。

**流程**：`raw/` 中有 XML 的图片 → 训练模型 → 对无标注图片推理 → 生成 XML 保存到 `raw/`

```bash
python train/scripts/01_auto_annotate/main.py
```

**配置参数**（修改 `main.py` 顶部变量）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CONFIDENCE_THRESHOLD | 0.3 | 推理置信度阈值，越低框越多 |
| MAX_PREDICT | 1000 | 单次推理上限，None = 全部 |
| EPOCHS | 100 | 训练轮数 |
| SKIP_TRAIN | False | True 时跳过训练，直接用已有模型推理 |
| DEVICE | 0 | GPU 设备号，"cpu" = 仅 CPU |

**使用流程**：

1. 往 `raw/` 放一些截图，手动标注其中几张（保存 XML）
2. 运行脚本 → 训练 → 自动标注未标注的图片
3. 检查自动生成的 XML 质量，修正错误标注
4. 重复以上步骤，逐步扩大训练集
5. 效果满意后，设 `MAX_PREDICT = None` 一次性标注全部

### 2. 训练模型 (`02_train_yolo/main.py`)

**适用场景**：标注数据足够多，想训练一个正式模型用于实际推理。

**流程**：从 `raw/` 读取所有标注数据 → 复制到 `auto_work/` → 划分训练集/验证集 → 训练 → 输出 `best.pt`

```bash
python train/scripts/02_train_yolo/main.py
```

**配置参数**（修改 `main.py` 顶部变量）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| EPOCHS | 150 | 训练轮数 |
| BATCH | 16 | 批次大小，显存不够可调小（如 4 或 8） |
| IMG_SIZE | 640 | 输入图片尺寸 |
| VAL_SPLIT | 0.2 | 验证集比例（20%） |
| SEED | 42 | 随机种子，保证划分可复现 |
| DEVICE | 0 | GPU 设备号，"cpu" = 仅 CPU |
| MODEL_PATH | `train/model/yolov8n.pt` | 预训练权重路径 |
| patience | 20 | 早停轮数，验证集 loss 不降则提前停止 |

**流程**：

1. 清空 `auto_work/`（每次训练重新开始）
2. 从 `data/raw/` 复制有 XML 标注的图片到 `auto_work/images/`
3. 将 Pascal VOC XML 转为 YOLO 格式保存到 `auto_work/labels/`
4. 随机划分 80% 训练集 / 20% 验证集
5. 训练 YOLO 模型，日志和权重保存到 `runs/`
6. 将 `best.pt` 复制到 `model/`

## 数据流

```
data/raw/ (手动标注 XML + 截图)
    │
    ├── 01_auto_annotate ──→ 复制到 scripts/01_auto_annotate/auto_work/
    │                        → 训练 → 推理未标注图片
    │                        → 新 XML 写回 data/raw/
    │
    └── 02_train_yolo ──→ 复制到 scripts/02_train_yolo/auto_work/
                          → 划分 train/val → 训练
                          → best.pt → scripts/02_train_yolo/model/
```

**核心原则**：`data/raw/` 是唯一的数据源，脚本只读取它，不会修改已有标注。`auto_work/` 是临时工作区，每次运行自动清空重建。