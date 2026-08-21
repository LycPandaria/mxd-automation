"""RapidOCR 名字定位器。

通过 OCR 识别画面中的角色名字，定位角色坐标。

================================================================================
原理（冒险岛实际布局）
================================================================================

  角色名字显示在角色脚下。从名字区域往上推算角色位置：

       ╔═════╗    ← 角色中心点（名字中心 - offset*0.5）
       ║ 角色 ║
       ╚═════╝
           │
           │  name_offset 像素（向上）
           │
           ▼
    ┌─────────────────┐
    │   我是立立       │  ← 名字区域（OCR 识别，在脚下）
    └────────┬────────┘
             ▼
            ═══  ← 脚底坐标（≈ 名字中心 y）

================================================================================
用法
================================================================================

  locator = OCRNameLocator(name_offset=40, on_log=print)
  # 返回 (center_x, center_y, foot_x, foot_y) 或 None
  result = locator.locate(frame, "我是立立")
  if result:
      cx, cy, fx, fy = result

================================================================================
注意事项
================================================================================

  - OCR 模型首次加载较慢（约 1-2 秒），后续帧很快
  - 建议配合缓存使用：OCR 帧找到后缓存坐标，非 OCR 帧直接返回缓存
  - 如果名字和其他文字重叠，可能匹配失败
  - name_offset 需要根据角色实际高度调整（通常 30~60 像素）
"""
from typing import Callable, Optional, Tuple

import numpy as np


class OCRNameLocator:
    """基于 RapidOCR 的角色名字定位器。

    冒险岛角色名字显示在脚下，通过名字中心向上偏移推算角色位置。

    每帧调用 locate()，内部有帧间隔控制（默认每 30 帧执行一次 OCR）。
    同时返回角色中心点和脚底坐标。

    Args:
        name_offset:    名字中心到角色脚底的向上偏移像素数，默认 40
        ocr_interval:   OCR 执行间隔（帧数），默认 30 帧
        exact_match:    是否精确匹配名字（True=完全相等，False=包含即可）
        on_log:         日志回调
    """

    def __init__(self, name_offset: int = 40,
                 ocr_interval: int = 30,
                 exact_match: bool = False,
                 on_log: Optional[Callable[[str], None]] = None):
        self._name_offset = name_offset
        self._interval = ocr_interval
        self._exact_match = exact_match
        self._on_log = on_log or (lambda m: None)
        self._engine = None
        self._frame_count = 0
        self._init_ok = False
        self._last_center: Optional[Tuple[int, int]] = None
        self._last_foot: Optional[Tuple[int, int]] = None

    # ---- 公开接口 ----

    def locate(self, frame: np.ndarray, name: str,
               min_confidence: float = 0.5,
               search_region: Optional[Tuple[int, int, int, int]] = None
               ) -> Optional[Tuple[int, int, int, int]]:
        """在画面中查找指定名字，返回 (中心点x, 中心点y, 脚底x, 脚底y)。

        冒险岛角色名字在脚下，所以:
          - 脚底 ≈ 名字中心 y（名字就在脚底位置）
          - 角色中心 = 名字中心 - name_offset * 0.5（向上）

        Args:
            frame:         BGR 截图 (H, W, 3)
            name:          要查找的角色名字（如 "我是立立"）
            min_confidence: 最低置信度阈值
            search_region:  可选搜索区域 (x, y, w, h)，裁剪后 OCR 更快

        Returns:
            (center_x, center_y, foot_x, foot_y) 或 None
        """
        name = name.strip()
        if not name:
            return None

        # 帧间隔控制
        self._frame_count += 1
        if self._frame_count % self._interval != 0:
            # 非 OCR 帧返回上次结果（如果有）
            if self._last_center and self._last_foot:
                return (*self._last_center, *self._last_foot)
            return None

        engine = self._get_engine()
        if engine is None:
            return None

        # 裁剪搜索区域（加速 OCR）
        roi = frame
        offset_x, offset_y = 0, 0
        if search_region is not None:
            sx, sy, sw, sh = search_region
            h, w = frame.shape[:2]
            x1 = max(0, sx)
            y1 = max(0, sy)
            x2 = min(w, sx + sw)
            y2 = min(h, sy + sh)
            if x2 <= x1 or y2 <= y1:
                return None
            roi = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        # 执行 OCR
        result, _ = engine(roi)
        if result is None:
            return None

        # 匹配名字
        best = None
        best_conf = 0.0
        for box, text, confidence in result:
            if not self._match(name, text):
                continue
            if confidence < min_confidence:
                continue
            if confidence > best_conf:
                best_conf = confidence
                best = box

        if best is None:
            return None

        # 名字区域中心（全局坐标）
        name_cx = int((best[0][0] + best[1][0] + best[2][0] + best[3][0]) / 4) + offset_x
        name_cy = int((best[0][1] + best[1][1] + best[2][1] + best[3][1]) / 4) + offset_y

        # 脚底 = 名字中心（名字就在脚边）
        foot = (name_cx, name_cy)
        # 角色中心 = 名字中心向上偏移 offset * 0.5
        center = (name_cx, name_cy - self._name_offset // 2)

        self._last_center = center
        self._last_foot = foot
        return (*center, *foot)

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

    @property
    def ready(self) -> bool:
        """OCR 引擎是否已就绪。"""
        return self._init_ok

    # ---- 内部 ----

    def _match(self, target: str, text: str) -> bool:
        """匹配名字。"""
        text = text.strip()
        if self._exact_match:
            return text == target
        return target in text

    def _get_engine(self):
        """延迟初始化 RapidOCR 引擎。"""
        if self._engine is not None:
            return self._engine if self._engine is not False else None

        self._on_log("[定位] 正在初始化 RapidOCR ...")
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._init_ok = True
            self._on_log("[定位] RapidOCR 初始化完成")
            return self._engine
        except Exception as e:
            self._on_log(f"[定位] RapidOCR 初始化失败: {e}")
            self._engine = False
            return None