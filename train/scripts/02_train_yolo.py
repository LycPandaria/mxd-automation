"""YOLO 训练脚本（占位）。

使用 ``ultralytics`` 加载预训练 YOLO 权重，在预处理后的数据集上微调。
训练产出保存在 ``train/runs/`` 下，训练完成后需将 ``best.pt`` 复制到
``assets/models/`` 供主程序使用。

依赖：``pip install ultralytics``

TODO:
  - 从 ``train/data/data.yaml`` 读取数据集配置
  - 支持断点续训（resume）
  - 自动导出 best.pt 到 assets/models/
"""
import os
import shutil


def train(data_yaml: str = "train/data/data.yaml",
          model_name: str = "yolov8n.pt",
          epochs: int = 100,
          imgsz: int = 640,
          batch: int = 16,
          resume: bool = False):
    """启动 YOLO 训练（占位）。"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics，请先运行: pip install ultralytics")
        return

    if resume:
        # TODO: 查找最近的 runs/ 目录用于续训
        print("[占位] 断点续训逻辑待实现")
        return

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="train/runs",
        name="exp",
    )

    # 拷贝 best.pt 到 assets/models/
    best_path = os.path.join("train", "runs", "exp", "weights", "best.pt")
    if os.path.exists(best_path):
        dest = os.path.join("assets", "models", "best.pt")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(best_path, dest)
        print(f"[完成] best.pt 已复制到 {dest}")
    else:
        print("[警告] 未找到 best.pt，请检查训练输出")


if __name__ == "__main__":
    train()
