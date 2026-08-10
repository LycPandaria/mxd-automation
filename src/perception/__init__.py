"""感知层：只负责"看"，输出结构化数据。

模块：
  - screen_capture:    高性能窗口截图（Win32 PrintWindow）
  - yolo_detector:     YOLO 目标检测
  - hp_mp_detector:    血/蓝条百分比检测（OpenCV）
  - template_matcher:  模板匹配（占位，识别固定 UI 元素）
"""
from .screen_capture import ScreenCapture, WindowCapture  # noqa: F401
from .yolo_detector import (
    Detection, Detector, MockDetector, YoloDetector, create_detector,  # noqa: F401
)
from .hp_mp_detector import detect_bar_ratio, detect_region_color  # noqa: F401
