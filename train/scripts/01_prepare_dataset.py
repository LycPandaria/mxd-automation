"""数据集预处理脚本（占位）。

将 ``train/data/raw/`` 中的截图整理为 YOLO 训练所需的数据集结构：
  train/data/
    images/train/    ← 训练图片
    images/val/      ← 验证图片
    labels/train/    ← YOLO 格式标注（.txt）
    labels/val/

TODO:
  - 按比例划分 train/val（默认 80/20）
  - 支持标注文件（YOLO .txt）的解析与转换
  - 生成 data.yaml 供 ultralytics 使用
"""
import os
import shutil
from pathlib import Path


def prepare(raw_dir: str = "train/data/raw",
            output_dir: str = "train/data",
            train_ratio: float = 0.8):
    """整理数据集（占位实现：仅拷贝图片，不划分）。

    TODO: 实现 train/val 划分 + 标注文件联动处理。
    """
    raw = Path(raw_dir)
    out = Path(output_dir)

    if not raw.exists():
        print(f"[跳过] 原始图片目录不存在: {raw}")
        return

    img_files = list(raw.glob("*.png")) + list(raw.glob("*.jpg"))
    if not img_files:
        print(f"[跳过] 原始目录无图片: {raw}")
        return

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    # 临时：全部放入 train（待实现 train/val 划分）
    for f in img_files:
        shutil.copy2(f, out / "images" / "train" / f.name)

    print(f"[完成] {len(img_files)} 张图片已拷贝到 {out}/images/train/")


if __name__ == "__main__":
    prepare()
