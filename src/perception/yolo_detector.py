"""YOLO 目标检测。

模型就绪后，``create_detector()`` 会自动加载（需安装 ``ultralytics``）。
模型未就绪时使用 ``MockDetector``，返回空结果，方便先调试 UI 与按键逻辑。
"""
import os
from dataclasses import dataclass
from typing import List, Callable, Optional

import numpy as np


@dataclass
class Detection:
    """单个检测结果。坐标基于传入的 frame。"""
    cls_name: str
    confidence: float
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


class Detector:
    """检测器基类。子类实现 detect()。"""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class MockDetector(Detector):
    """占位检测器，返回空结果。"""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []


class YoloDetector(Detector):
    """基于 ultralytics YOLO 的检测器。

    参考: 模型训练好后，设置 config.detection.model_path 指向 .pt 文件即可自动启用。
    """

    def __init__(self, model_path: str, conf: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self._path = model_path  # 供上层判断是否需要重建检测器

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.conf, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(Detection(
                    cls_name=self.model.names[int(box.cls)],
                    confidence=float(box.conf),
                    x=int(x1), y=int(y1),
                    w=int(x2 - x1), h=int(y2 - y1),
                ))
        return dets


def create_detector(model_path: str, conf: float = 0.5,
                    on_log: Optional[Callable[[str], None]] = None) -> Detector:
    """根据配置创建检测器。模型不可用时回退到 MockDetector。"""
    log = on_log or (lambda m: None)
    if model_path and os.path.exists(model_path):
        try:
            det = YoloDetector(model_path, conf)
            log(f"[检测] 已加载 YOLO 模型: {model_path}")
            return det
        except ImportError:
            log("[警告] 未安装 ultralytics，回退到 Mock 检测器。安装: pip install ultralytics")
        except Exception as e:
            log(f"[警告] 加载模型失败: {e}，回退到 Mock 检测器")
    else:
        if model_path:
            log(f"[警告] 模型文件不存在: {model_path}，使用 Mock 检测器")
        else:
            log("[提示] 未设置 YOLO 模型路径，使用 Mock 检测器（不会检测到目标）")
    return MockDetector()
