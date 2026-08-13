"""RapidOCR 名字定位器。

通过 OCR 识别画面中的角色名字，计算名字文字区域中心坐标，
即为角色脚底位置。

用法:
    locator = OCRNameLocator(on_log=print)
    pos = locator.locate(frame, "我是立立")  # -> (cx, cy) or None
"""

from typing import Callable, Optional, Tuple

import numpy as np


class OCRNameLocator:
    """基于 RapidOCR 的角色名字定位器。

    Args:
        on_log: 日志回调
        ocr_interval: OCR 执行间隔（帧数），默认 30 帧
    """

    def __init__(self, on_log: Optional[Callable[[str], None]] = None,
                 ocr_interval: int = 30):
        self._on_log = on_log or (lambda m: None)
        self._interval = ocr_interval
        self._engine = None
        self._frame_count = 0

    # ---- 公开接口 ----

    def locate(self, frame: np.ndarray, name: str,
               min_confidence: float = 0.5) -> Optional[Tuple[int, int]]:
        """在画面中查找指定名字，返回名字区域中心坐标。

        Args:
            frame: BGR 截图 (H, W, 3)
            name: 要查找的角色名字（如 "我是立立"）
            min_confidence: 最低置信度阈值

        Returns:
            (cx, cy) 中心像素坐标，未找到返回 None
        """
        name = name.strip()
        if not name:
            return None

        # 帧间隔控制，避免每帧都跑 OCR
        self._frame_count += 1
        if self._frame_count % self._interval != 0:
            return None

        engine = self._get_engine()
        if engine is None:
            return None

        result, _ = engine(frame)
        if result is None:
            return None

        for box, text, confidence in result:
            if name in text and confidence >= min_confidence:
                cx = int((box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4)
                cy = int((box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4)
                return (cx, cy)

        return None

    def locate_all(self, frame: np.ndarray,
                   min_confidence: float = 0.5) -> list:
        """返回画面中所有识别到的文字及其坐标。

        Returns:
            [(text, (cx, cy), confidence), ...]
        """
        engine = self._get_engine()
        if engine is None:
            return []

        result, _ = engine(frame)
        if result is None:
            return []

        items = []
        for box, text, confidence in result:
            if confidence >= min_confidence:
                cx = int((box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4)
                cy = int((box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4)
                items.append((text, (cx, cy), confidence))
        return items

    # ---- 内部 ----

    def _get_engine(self):
        """延迟初始化 RapidOCR 引擎。"""
        if self._engine is not None:
            return self._engine if self._engine is not False else None

        self._on_log("[定位] 正在初始化 RapidOCR ...")
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._on_log("[定位] RapidOCR 初始化完成")
            return self._engine
        except Exception as e:
            self._on_log(f"[定位] RapidOCR 初始化失败: {e}")
            self._engine = False
            return None