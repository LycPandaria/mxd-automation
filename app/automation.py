"""自动打怪主循环：截图 → YOLO 检测 → 决策(移动/放技能/加血) → 按键触发。

在独立线程运行，通过 start/stop 控制。检测和预览结果通过回调（信号）传回 UI 线程。
"""
import time
import threading

from .window_capture import WindowCapture
from .detector import Detector, detect_bar_ratio
from .controller import Controller
from .config import Config


class Automation:
    def __init__(self, config: Config, detector: Detector,
                 on_log=None, on_frame=None):
        """
        Args:
            on_log: 回调 (str) -> None，日志
            on_frame: 回调 (frame_bgr, detections, hp_ratio) -> None，预览更新
        """
        self.config = config
        self.capture = WindowCapture()
        self.detector = detector
        self.controller = Controller()
        self.on_log = on_log or (lambda m: None)
        self.on_frame = on_frame or (lambda f, d, h: None)

        self._running = False
        self._thread = None
        self._skill_index = 0

    # ---- 窗口管理 ----
    def list_windows(self):
        return self.capture.list_windows()

    def lock_window(self, title: str) -> str:
        return self.capture.lock(title=title)

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
            monsters = [d for d in detections
                        if d.cls_name in self._monster_classes()]

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

            # 决策与执行
            self._decide(monsters, hp_ratio, mp_ratio)

            # 预览回调
            self.on_frame(frame, detections, hp_ratio, mp_ratio)

            # 控制 FPS
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _monster_classes(self):
        return [c.strip() for c in self.config.monster_classes.split(",") if c.strip()]

    def _decide(self, monsters, hp_ratio, mp_ratio):
        # 1) 没血优先加血
        if hp_ratio is not None and hp_ratio < self.config.hp_threshold:
            if self.controller.press_key(self.config.hp_key, cooldown=1.5):
                self.on_log(
                    f"[加血] HP={hp_ratio:.0%} < {self.config.hp_threshold:.0%}，"
                    f"按下 {self.config.hp_key}"
                )
                return

        # 2) 没蓝加蓝
        if mp_ratio is not None and mp_ratio < self.config.mp_threshold:
            if self.controller.press_key(self.config.mp_key, cooldown=1.5):
                self.on_log(
                    f"[加蓝] MP={mp_ratio:.0%} < {self.config.mp_threshold:.0%}，"
                    f"按下 {self.config.mp_key}"
                )
                return

        # 3) 检测到怪物
        if monsters:
            target = max(monsters, key=lambda d: d.w * d.h)
            cx, cy = target.center
            self.on_log(
                f"[检测] {target.cls_name} conf={target.confidence:.2f} @ ({cx},{cy})"
            )

            # 可选：点击移动到怪位置（窗口内坐标转屏幕坐标）
            if self.config.move_to_monster:
                rect = self.capture.get_rect()
                self.controller.click(rect[0] + cx, rect[1] + cy)
                return

            # 选中目标
            self.controller.press_key(self.config.target_key, cooldown=0.8)

            # 轮转释放技能（挑一个冷却好的）
            skills = self.config.skills
            for _ in range(len(skills)):
                skill = skills[self._skill_index % len(skills)]
                self._skill_index += 1
                if self.controller.press_key(skill["key"], skill["cooldown"]):
                    self.on_log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                    break
        else:
            # 没怪，按选中键自动寻找目标
            self.controller.press_key(self.config.target_key, cooldown=1.5)
