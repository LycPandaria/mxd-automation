"""RapidOCR 名字定位器。

通过 OCR 识别画面中的角色名字，定位角色坐标。

================================================================================
原理（冒险岛实际布局）
================================================================================

  角色名字显示在角色脚下。名字中心 ≈ 角色脚底，从名字中心向上
  延伸"人物高度一半"得到角色中心点（角色身体在名字上方）：

    ╔═══════════╗   ← 名字中心 y - 30 = 角色中心点 y（往上）
    ║  我是立立   ║
    ╚═══╤═══════╝
        │  ↑ 名字中心 = 脚底
        │
        │  ↑ - 人物高度一半（character_height // 2）
        │
    ╔═══╧═══╗
    ║ 角色   ║
    ║       ║  ← 人物高度（character_height，约 60px）
    ╚═══════╝

================================================================================
用法
================================================================================

  locator = OCRNameLocator(character_height=60, on_log=print)
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
  - character_height 需要根据角色实际高度调整（通常 50~70 像素）
"""
from typing import Callable, Optional, Tuple

import numpy as np


class OCRNameLocator:
    """基于 RapidOCR 的角色名字定位器。

    冒险岛角色名字显示在脚下，名字中心 ≈ 脚底；
    角色中心点 = 名字中心向上延伸"人物高度一半"。

    每帧调用 locate()，内部有帧间隔控制（默认每 30 帧执行一次 OCR）。
    同时返回角色中心点和脚底坐标。

    Args:
        character_height: 人物高度像素数，默认 60；角色中心 = 名字中心 - 高度一半
        ocr_interval:     OCR 执行间隔（帧数），默认 30 帧
        exact_match:      是否精确匹配名字（True=完全相等，False=包含即可）
        on_log:           日志回调
    """

    def __init__(self, character_height: int = 60,
                 ocr_interval: int = 30,
                 exact_match: bool = False,
                 on_log: Optional[Callable[[str], None]] = None):
        self._character_height = character_height
        self._interval = ocr_interval
        self._exact_match = exact_match
        self._on_log = on_log or (lambda m: None)
        self._engine = None
        self._frame_count = 0
        self._init_ok = False
        self._last_center: Optional[Tuple[int, int]] = None
        self._last_foot: Optional[Tuple[int, int]] = None

        # 帧间跳变过滤：OCR 偶尔会把画面中固定位置的 UI 文字（聊天框、
        # 任务栏、怪物名等）误识别为角色名（匹配是"包含"逻辑），
        # 导致角色坐标瞬间跳到别处，决策层基于错误坐标把同平台的怪
        # 判成跨层 → 一直移动不攻击。
        # 策略：新识别位置与上次有效位置偏移超过阈值 → 直接判定为
        # 误识别，沿用上次位置且不更新缓存。角色正常移动每帧最多几十
        # px，超过上限只能是误识别（游戏无瞬移）；换图后 OCR 找不到
        # 名字会返回 None，由决策层按未定位处理，不会用错误位置。
        self._max_jump = 200  # 单次识别相对上次有效位置的偏移上限(px)

        # 连续误识别超时重置：如果首帧就识别到了错误位置（比如匹配到
        # UI 固定文字），后续所有帧都会被 _max_jump 过滤掉，位置永远
        # 卡在错误坐标上。策略：连续 N 帧都被跳变过滤 → 接受新位置
        # 并重置缓存，防止首帧错误永久锁死定位。
        self._skip_counter = 0
        self._skip_threshold = 5

        # 位置长期未变强制复位：OCR 首帧匹配到 UI 固定文字（如聊天框
        # 中的角色名）后，后续帧因"离上次最近"策略永远选同一位置，
        # 导致定位锁死在错误坐标。策略：连续 N 次 OCR 位置未变（偏移
        # < 3px），强制改用"置信度最高"选候选，打破死锁。
        self._stale_counter = 0
        self._stale_threshold = 10

    # ---- 公开接口 ----

    def locate(self, frame: np.ndarray, name: str,
               min_confidence: float = 0.5,
               search_region: Optional[Tuple[int, int, int, int]] = None
               ) -> Optional[Tuple[int, int, int, int]]:
        """在画面中查找指定名字，返回 (中心点x, 中心点y, 脚底x, 脚底y)。

        和 YOLO 一样，每帧都执行 OCR，不做缓存、不做跳变过滤、
        不做"离上次最近"策略。直接取置信度最高的匹配结果。

        冒险岛角色名字在脚下，所以:
          - 脚底 ≈ 名字中心 y（名字就在脚底位置）
          - 角色中心 = 名字中心 - character_height // 2（向上延伸人物高度一半）

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

        # 执行 OCR（引擎执行异常时降级返回 None，不让异常扩散到主循环）
        try:
            result, _ = engine(roi)
        except Exception as e:
            self._on_log(f"[定位] OCR 执行异常: {e}")
            return None
        if result is None:
            return None

        # 匹配名字：取置信度最高的候选（和 YOLO 一样，每帧独立计算，无缓存）
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

        # 脚底 = 名字中心（名字就在脚边，画面 y 向下增大）
        foot = (name_cx, name_cy)
        # 角色中心 = 名字中心向上延伸"人物高度一半"（-30px）
        center = (name_cx, name_cy - self._character_height // 2)

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

        try:
            result, _ = engine(frame)
        except Exception as e:
            self._on_log(f"[定位] OCR 执行异常: {e}")
            return []
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