"""测谎弹窗检测（反外挂）。

================================================================================
用途
================================================================================

  游戏（冒险岛怀旧服）的"测谎探测仪"弹窗出现时，必须在几秒内放下机器人、
  把真实鼠标交给玩家去玩"光标跟随图形"的小游戏（弹窗正文写明"3秒后，
  测谎游戏开始"）。本模块负责在每一帧里发现这个弹窗。

  检测到之后该做什么（停机 + 报警）不在这里，见 src/main.py 与 ui/main_window.py。

================================================================================
原理：模板匹配
================================================================================

  弹窗是固定外观的 UI 图形（蓝色描边 + 白色标题栏 + 正文 + 金色图形），
  所以用 OpenCV 的 cv2.matchTemplate() 做模板匹配，比 OCR 快几个数量级、
  也不需要任何额外依赖或语言包。

  三个关键优化（均为实测结论，1366x768）：

  1) 整帧降采样 DOWNSAMPLE=4 倍再匹配
     模板 446x394 直接全图匹配要几秒；降到 1/4 后单档约 12ms，且降采样
     本身会抹平文字细节 —— 弹窗里的倒计时数字/措辞变化不再影响得分
     （实测把所有正文涂掉仍得 0.83，只有连金色图形一起涂掉才掉到 0.47）。

  2) 搜索范围限制在画面正中 SEARCH_MARGIN 内
     弹窗恰好居中（实测中心 (682.5,383.5) vs 客户区正中 (683,384)），
     所以只需在中心附近搜索。单档耗时从 12ms 降到约 0.45ms。

  3) 模板与缩放后的小图都做缓存
     同一分辨率下每帧的尺度组合是固定的，首次算完就缓存。

  实测：本检测整帧耗时约 2.6ms（7 档尺度，搜索边距 ±160px；单档首算后缓存），
  而同一帧的 YOLO 推理约 30ms，fps=10 时一帧预算 100ms
  —— 因此可以每帧都跑。

================================================================================
尺度自适应
================================================================================

  模板按 TEMPLATE_REF_SIZE 裁切（模板图片的原始分辨率）。检测时按
      sx = 帧宽 / 模板参考宽,  sy = 帧高 / 模板参考高
  得到基准比例（与 config_loader.scale_region() 同一套约定），再叠一个
  ±SCALE_JITTER 的等比小幅扫描。

  为什么 x/y 要分开算：实测把画面按 1368x800 各向异性缩放后，用按宽高
  分别推导的模板得分 0.96，而用等比 1.0 只有 0.77 —— 分开算对"UI 随窗口
  等比/非等比缩放"两种情况都稳。

  注意：模板对缩放很敏感（实测 ±2% 就掉到 0.79/0.73，±5% 只有 0.38），
  所以尺度扫描不能省。

================================================================================
误报（实测）
================================================================================

  正样本（test/ 那张弹窗截图）：0.962
  负样本（backup/ 训练集普通打怪帧）：
      全图搜索（最保守的上界）：最高 0.42，p99 0.40（100 张）
      本模块的居中限定搜索    ：最高 0.39，平均 0.27（120 张）

  两者差距极大，阈值 0.78 有充足安全边际。若日后出现误报，先看日志里的
  score，再决定调阈值还是换模板。

================================================================================
换模板 / 重新标定
================================================================================

  1. 让游戏出现测谎弹窗（或从已有截图里找），截图保存
  2. 确认弹窗外框（含 1px 白色描边）的像素坐标 (x1,y1)-(x2,y2)
  3. 裁 [y1:y2, x1:x2] 覆盖写到 assets/templates/lie_detector.png
  4. 同步改 TEMPLATE_REF_SIZE 为该截图的客户区分辨率
  5. 跑 .tmp/verify_lie_detector.py 复核正/负样本得分
"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---- 模板裁切时的客户区分辨率（弹窗几何随此分辨率记录）----
TEMPLATE_REF_SIZE = (1366, 768)

# ---- 默认参数（可被 config 覆盖）----
DEFAULT_DOWNSAMPLE = 4       # 整帧降采样倍数
DEFAULT_SEARCH_MARGIN = 160  # 居中搜索范围（原始像素，±值）
DEFAULT_THRESHOLD = 0.78     # 匹配得分阈值
DEFAULT_SCALE_JITTER = 0.03  # 尺度抖动范围（±3%）
DEFAULT_SCALE_STEP = 0.01    # 尺度扫描步长（1%）


class LieDetector:
    """测谎弹窗检测器（模板匹配）。

    用法:
        det = LieDetector("assets/templates/lie_detector.png")
        hit, score = det.detect(frame)     # frame: BGR 或灰度 numpy 数组

    线程安全：detect() 只读缓存，可在工作线程里反复调用。

    Args:
        template_path: 模板图片路径（不存在时 available=False，detect 恒返回 False）
        threshold:     命中阈值（0.0~1.0）
        downsample:    整帧降采样倍数，越大越快越钝（默认 4）
        search_margin: 居中搜索范围（原始像素，±值），覆盖弹窗位置漂移
        scale_jitter:  尺度抖动范围（±，默认 0.03）
        scale_step:    尺度扫描步长（默认 0.01）
        ref_size:      模板裁切时的分辨率 (w, h)
        on_log:        日志回调
    """

    def __init__(self,
                 template_path: Optional[str] = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 downsample: int = DEFAULT_DOWNSAMPLE,
                 search_margin: int = DEFAULT_SEARCH_MARGIN,
                 scale_jitter: float = DEFAULT_SCALE_JITTER,
                 scale_step: float = DEFAULT_SCALE_STEP,
                 ref_size: Tuple[int, int] = TEMPLATE_REF_SIZE,
                 on_log=None):
        self.template_path = template_path
        self.threshold = float(threshold)
        self.downsample = max(1, int(downsample))
        self.search_margin = max(0, int(search_margin))
        self.ref_size = ref_size
        self._log = on_log or (lambda m: None)

        # 尺度扫描档位：1.0 及两侧 ±jitter
        jit = max(0.0, float(scale_jitter))
        step = max(1e-3, float(scale_step))
        n = int(round(jit / step))
        self._scale_factors: List[float] = [
            1.0 + i * step for i in range(-n, n + 1)
        ]

        # 模板（灰度）与各尺度缓存；缓存键为 (sx, sy) 四舍五入后的值
        self._template: Optional[np.ndarray] = None
        self._cache = {}

        # 最近一次检测结果（供日志/调试）
        self.last_score: float = 0.0
        self.last_scale: Optional[float] = None

        self._load()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _load(self):
        """读取模板图片（灰度）。失败则 available 为 False，不影响主循环。"""
        if not self.template_path:
            self._log("[测谎] 未配置模板路径，测谎检测已禁用")
            return
        if not os.path.isfile(self.template_path):
            self._log(f"[测谎] 模板不存在，测谎检测已禁用: {self.template_path}")
            return
        # 用 IMREAD_GRAYSCALE 读取：matchTemplate 要求模板与图像同类型，
        # 彩色模板会直接抛 AssertionError(type == _templ.type())
        tpl = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None or tpl.size == 0:
            self._log(f"[测谎] 模板读取失败，测谎检测已禁用: {self.template_path}")
            return
        self._template = tpl
        self._log(f"[测谎] 弹窗模板已加载：{self.describe()}")

    @property
    def available(self) -> bool:
        """模板是否可用。"""
        return self._template is not None

    def describe(self) -> str:
        """一行状态描述（供启动日志，便于确认阈值/尺度档位是否符合预期）。"""
        if self._template is None:
            return "未加载（模板不可用）"
        th, tw = self._template.shape[:2]
        return (f"模板 {tw}x{th} 阈值={self.threshold} "
                f"尺度档位={len(self._scale_factors)} "
                f"降采样=1/{self.downsample} 搜索边距=±{self.search_margin}px")

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Tuple[bool, float]:
        """检测当前帧是否出现测谎弹窗。

        Args:
            frame: 当前帧（BGR 或灰度 numpy 数组）

        Returns:
            (hit, score)：hit 为是否命中（score >= threshold），
            score 为本次最高匹配得分（未加载模板时为 0.0）
        """
        if self._template is None or frame is None or frame.size == 0:
            return False, 0.0

        ds = self.downsample
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=1.0 / ds, fy=1.0 / ds,
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]

        # 基准缩放：与 config_loader.scale_region() 同一套约定（x/y 分开）
        fw, fh = frame.shape[1], frame.shape[0]
        ref_w, ref_h = self.ref_size
        base_x = fw / float(ref_w) if ref_w else 1.0
        base_y = fh / float(ref_h) if ref_h else 1.0

        margin = self.search_margin // ds
        best, best_scale = -1.0, None
        for f in self._scale_factors:
            sx, sy = round(base_x * f, 4), round(base_y * f, 4)
            tpl = self._cache.get((sx, sy))
            if tpl is None:
                tpl = cv2.resize(self._template, None, fx=sx / ds, fy=sy / ds,
                                 interpolation=cv2.INTER_AREA)
                self._cache[(sx, sy)] = tpl
            th, tw = tpl.shape[:2]
            if th >= sh or tw >= sw:
                continue
            # 只在中点附近搜索（弹窗居中；margin 覆盖位置漂移）
            y1 = max(0, sh // 2 - th // 2 - margin)
            y2 = min(sh, sh // 2 + th // 2 + margin)
            x1 = max(0, sw // 2 - tw // 2 - margin)
            x2 = min(sw, sw // 2 + tw // 2 + margin)
            res = cv2.matchTemplate(small[y1:y2, x1:x2], tpl,
                                    cv2.TM_CCOEFF_NORMED)
            score = float(cv2.minMaxLoc(res)[1])
            if score > best:
                best, best_scale = score, f

        self.last_score = max(0.0, best)
        self.last_scale = best_scale
        return self.last_score >= self.threshold, self.last_score

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def reset(self):
        """清空尺度缓存（换分辨率/换模板后用）。"""
        self._cache.clear()
