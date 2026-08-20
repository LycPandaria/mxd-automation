# 训练模块

基于 YOLOv8n 的目标检测模型训练与自动标注。

## 目录结构

```
train/
├── data/
│   ├── raw/                  # 原始数据（图片 + Pascal VOC XML），只读不写
├── model/
│   └── yolov8n.pt            # YOLOv8n 预训练权重（自动下载）
├── scripts/
│   ├── 01_auto_annotate/     # 自动标注脚本
│   │   ├── main.py
│   │   └── auto_work/        # 自动标注临时训练目录，每次自动清空
│   └── 02_train_yolo/        # 训练脚本
│       ├── auto_work/            # 02_train_yolo 临时训练目录，每次自动清空
│       ├── main.py
│       ├── runs/              # 训练输出（自动生成）
│       └── model/             # 训练产出的 best.pt
```

## 环境要求

- Python 3.12+
- CUDA 12.4（推荐 RTX 4060 及以上）
- 依赖：`pip install ultralytics torch torchvision`

## 数据标注

使用 LabelImg 等工具对 `data/raw/` 下的图片进行标注，保存为 **Pascal VOC XML** 格式。

类别定义（共 3 类）：

| ID | 类别名 |
|----|--------|
| 0  | floor  |
| 1  | monster |
| 2  | rope   |

## 脚本说明

### 1. 自动标注 (`01_auto_annotate/main.py`)

`raw/` 中有 XML 的图片 → 训练模型 → 对无标注图片推理 → 生成 XML 保存到 `raw/`

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

从 `raw/` 读取标注数据，复制到 `auto_work/`，划分训练集/验证集，训练 YOLO 模型。

```bash
python train/scripts/02_train_yolo/main.py
```

**配置参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| EPOCHS | 100 | 训练轮数 |
| BATCH | 16 | 批次大小，显存不够可调小 |
| IMG_SIZE | 640 | 输入图片尺寸 |
| VAL_SPLIT | 0.2 | 验证集比例（20%） |
| SEED | 42 | 随机种子，保证划分可复现 |
| DEVICE | 0 | GPU 设备号 |

**流程**：

1. 清空 `auto_work/`
2. 从 `raw/` 复制有 XML 的图片到 `auto_work/`
3. 随机划分 80% 训练集 / 20% 验证集
4. 训练 YOLO 模型，训练日志保存到 `scripts/02_train_yolo/runs/`
5. 将 `best.pt` 复制到 `scripts/02_train_yolo/model/`

## 数据流

```
raw/ (手动标注 XML + 图片)
    |
    ├── 01_auto_annotate ──→ 复制到 scripts/01_auto_annotate/auto_work/ → 训练 → 推理 → 新 XML 写回 raw/
    |
    └── 02_train_yolo ──→ 复制到 data/auto_work/ → 划分 train/val → 训练 → best.pt → scripts/02_train_yolo/model/
```

**核心原则**：`raw/` 是唯一的数据源，脚本只读取它，不会修改已有标注。`auto_work/` 是临时工作区，每次运行自动清空。