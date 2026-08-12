"""数据集预处理脚本。

将 ``train/data/raw/`` 中的截图与标注文件整理为 YOLO 训练结构：
  train/data/
    images/train/    ← 训练图片
    images/val/      ← 验证图片
    labels/train/    ← YOLO 格式标注（.txt）
    labels/val/
    data.yaml        ← 数据集描述文件
"""
import os
import random
import shutil
from pathlib import Path


def prepare(raw_dir: str = "train/data/raw",
            output_dir: str = "train/data",
            train_ratio: float = 0.8,
            seed: int = 42):
    raw = Path(raw_dir)
    out = Path(output_dir)

    if not raw.exists():
        print(f"[错误] 原始图片目录不存在: {raw}")
        return

    img_files = sorted(list(raw.glob("*.png")) + list(raw.glob("*.jpg")))
    if not img_files:
        print(f"[错误] 原始目录无图片: {raw}")
        return

    # 清空旧输出
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            p = out / sub / split
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    # 打乱并划分
    random.seed(seed)
    random.shuffle(img_files)
    split = int(len(img_files) * train_ratio)
    train_files = img_files[:split]
    val_files = img_files[split:]

    def copy_set(files, target_split):
        for f in files:
            shutil.copy2(f, out / "images" / target_split / f.name)
            label = f.with_suffix(".txt")
            if label.exists():
                shutil.copy2(label, out / "labels" / target_split / label.name)
            else:
                print(f"[警告] 缺少标注文件: {label.name}")

    copy_set(train_files, "train")
    copy_set(val_files, "val")

    # 识别类别名
    classes_file = raw / "classes.txt"
    names = {}
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    names[i] = name
    if not names:
        names = {0: "monster", 1: "floor", 2: "rope", 3: "player"}

    # 生成 data.yaml
    yaml_path = out / "data.yaml"
    yaml_content = [
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for idx, name in sorted(names.items()):
        yaml_content.append(f"  {idx}: {name}")
    yaml_content.append(f"nc: {len(names)}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_content) + "\n")

    print(f"[完成] 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")
    print(f"[完成] data.yaml 已生成: {yaml_path}")
    print(f"[完成] 类别: {dict(sorted(names.items()))}")


if __name__ == "__main__":
    prepare()