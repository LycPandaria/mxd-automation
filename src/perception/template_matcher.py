"""模板匹配（占位）。

识别固定 UI 元素（如符文图标、弹窗、技能图标）。
后续实现：基于 ``cv2.matchTemplate`` 在指定 ROI 内多尺度模板匹配，
返回 ``TemplateMatch`` 列表（位置 + 置信度）。

资源目录：``assets/templates/``
"""
from typing import List, Optional

import numpy as np


class TemplateMatch:
    """模板匹配结果（占位数据结构）。"""

    def __init__(self, name: str, x: int, y: int, w: int, h: int, confidence: float):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence


class TemplateMatcher:
    """模板匹配器（占位）。

    TODO:
        - 加载 ``assets/templates/*.png``
        - ``cv2.matchTemplate`` 多尺度匹配
        - NMS 去重
    """

    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = templates_dir
        self._templates = {}  # name -> ndarray

    def load(self):
        """加载模板图片。"""
        # TODO: 实现
        pass

    def match(self, frame: np.ndarray, threshold: float = 0.8) -> List[TemplateMatch]:
        """在 frame 中匹配所有已加载模板。"""
        # TODO: 实现
        return []
