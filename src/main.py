"""主程序入口与自动打怪主循环。

``Automation`` 类整合感知层（ScreenCapture + Detector）、决策层
（DecisionEngine）和执行层（ActionExecutor），在独立线程运行主循环：

  截图 → YOLO 检测 → HP/MP 检测 → 决策（执行动作） → 预览回调

通过 start/stop 控制，检测和预览结果通过回调（信号）传回 UI 线程。
"""
import os
import time
import threading
from typing import Callable, Optional, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .perception.screen_capture import ScreenCapture
from .perception.yolo_detector import Detector, create_detector
from .perception.hp_mp_detector import detect_bar_ratio
from .execution.action_executor import ActionExecutor
from .decision.context import Context, DecisionEngine
from .utils.config_loader import Config


class Automation:
    """自动打怪主循环控制器。

    Args:
        config: 全局配置
        detector: YOLO 检测器（None 时使用 MockDetector）
        on_log:  回调 (str) -> None，日志
        on_frame: 回调 (frame_bgr, detections, hp_ratio, mp_ratio) -> None，预览更新
    """

    def __init__(self, config: Config, detector: Optional[Detector] = None,
                 on_log: Optional[Callable[[str], None]] = None,
                 on_frame: Optional[Callable[..., None]] = None):
        self.config = config
        self.capture = ScreenCapture()
        self.detector = detector if detector is not None else create_detector(
            config.model_path, config.confidence, on_log or (lambda m: None)
        )
        self.executor = ActionExecutor()
        self.engine = DecisionEngine(
            config, self.executor, self.capture,
            on_log=on_log or (lambda m: None)
        )
        self.on_log = on_log or (lambda m: None)
        self.on_frame = on_frame or (lambda f, d, h, m: None)

        self._running = False
        self._thread = None
        self._name_template = None  # 名字渲染图（用于自身定位）
        self._render_name_template()

    # ---- 窗口管理 ----

    def list_windows(self):
        return self.capture.list_windows()

    def lock_window(self, title: str) -> str:
        locked = self.capture.lock(title=title)
        # 把窗口句柄传给执行层（PostMessage 注入）
        self.executor.set_target_window(self.capture.hwnd)
        return locked

    def unlock_window(self):
        self.capture.unlock()

    @property
    def window_locked(self):
        return self.capture.locked

    def get_window_rect(self):
        return self.capture.get_rect() if self.capture.locked else None

    # ---- 检测器 ----

    def set_detector(self, detector: Detector):
        self.detector = detector

    # ---- 主循环 ----

    @property
    def running(self):
        return self._running

    def start(self):
        if self._running:
            return
        if not self.capture.locked:
            raise RuntimeError("请先锁定游戏窗口")
        # 配置可能被 UI 更新过，同步给决策引擎
        self.engine.update_config(self.config)
        self.engine.reset()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.on_log("[启动] 自动打怪已开始")

    def stop(self):
        if not self._running:
            return
        self._running = False
        self.on_log("[停止] 自动打怪已停止")

    def _loop(self):
        interval = 1.0 / max(1, self.config.fps)
        while self._running:
            t0 = time.time()
            try:
                frame = self.capture.grab()
            except Exception as e:
                self.on_log(f"[错误] 截图失败: {e}")
                time.sleep(interval)
                continue

            # YOLO 检测
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                self.on_log(f"[错误] 检测失败: {e}")
                detections = []
            monster_classes = self._monster_classes()
            monsters = [d for d in detections if d.cls_name in monster_classes]
            floors = [d for d in detections if d.cls_name in self._floor_classes()]
            ropes = [d for d in detections if d.cls_name in self._rope_classes()]
            players = [d for d in detections if d.cls_name in self._player_classes()]

            # HP 检测
            hp_ratio = detect_bar_ratio(
                frame, self.config.hp_region,
                tuple(self.config.hp_color) if self.config.hp_color else None,
                self.config.hp_tolerance,
            )

            # MP 检测
            mp_ratio = detect_bar_ratio(
                frame, self.config.mp_region,
                tuple(self.config.mp_color) if self.config.mp_color else None,
                self.config.mp_tolerance,
            )

            # 自身位置：HP条偏移优先，名字模板匹配兜底
            self_pos = self._locate_self(frame)

            # 决策与执行
            ctx = Context(
                monsters=monsters,
                floors=floors,
                ropes=ropes,
                players=players,
                self_position=self_pos,
                hp_ratio=hp_ratio,
                mp_ratio=mp_ratio,
                detections=detections,
            )
            self.engine.decide(ctx)

            # 预览回调
            self.on_frame(frame, detections, hp_ratio, mp_ratio)

            # 控制 FPS
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _monster_classes(self):
        return [c.strip() for c in self.config.monster_classes.split(",") if c.strip()]

    def _floor_classes(self):
        return [c.strip() for c in self.config.floor_classes.split(",") if c.strip()]

    def _rope_classes(self):
        return [c.strip() for c in self.config.rope_classes.split(",") if c.strip()]

    def _player_classes(self):
        return [c.strip() for c in self.config.player_classes.split(",") if c.strip()]

    # ---- 自身定位 ----

    def _render_name_template(self):
        """将 self_name 文本渲染为模板图片，用于 cv2.matchTemplate 匹配。"""
        name = self.config.self_name.strip()
        if not name:
            self._name_template = None
            return

        # 尝试匹配游戏内字体，找不到则用默认字体
        font_paths = [
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simhei.ttf"),
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc"),
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyhbd.ttc"),
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, 14)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()

        # 测量文字尺寸
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        padding = 4
        w, h = tw + padding * 2, th + padding * 2

        # 渲染：灰色背景 + 白色文字（模拟脚底名字区域）
        img = Image.new("RGB", (w, h), color=(60, 60, 60))
        draw = ImageDraw.Draw(img)
        draw.text((padding - bbox[0], padding - bbox[1]), name, fill=(255, 255, 255), font=font)

        self._name_template = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.on_log(f"[定位] 名字模板已渲染: {name} ({w}x{h})")

    def set_self_name(self, name: str):
        """运行时更新自身名字并重新渲染模板。"""
        self.config.self_name = name
        self._render_name_template()

    def _locate_self(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """定位自身脚底坐标。

        策略：
          1. 有 HP 条区域 → HP 条底部偏移推算（最准）
          2. 有名字模板 → cv2.matchTemplate 匹配脚底名字
          3. 都没有 → None
        """
        # 方案1: HP条偏移
        if self.config.hp_region:
            hx, hy, hw, hh = self.config.hp_region
            return (hx + hw // 2, hy + hh + 85)

        # 方案2: 名字模板匹配
        if self._name_template is not None:
            th, tw = self._name_template.shape[:2]
            result = cv2.matchTemplate(frame, self._name_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= 0.7:
                # 名字在脚底，匹配位置就是脚底位置
                return (max_loc[0] + tw // 2, max_loc[1] + th // 2)

        return None


def main():
    """CLI 入口（无 GUI）：加载配置并启动主循环，按 Ctrl+C 退出。"""
    from .utils.logger import get_logger
    log = get_logger()

    cfg = load_config()
    if not cfg.window_title:
        log.error("未配置 window_title，请在 config/user.json 中设置")
        return

    auto = Automation(cfg, on_log=lambda m: log.info(m))
    try:
        locked = auto.lock_window(cfg.window_title)
        log.info(f"已锁定窗口: {locked}")
    except Exception as e:
        log.error(f"锁定窗口失败: {e}")
        return

    auto.start()
    try:
        while auto.running:
            time.sleep(1)
    except KeyboardInterrupt:
        auto.stop()


if __name__ == "__main__":
    from .utils.config_loader import load_config
    main()