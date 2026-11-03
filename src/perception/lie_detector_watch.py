"""测谎仪（谎言探测仪）弹窗监视 + 声音报警。

游戏会不定时弹出「谎言探测仪」窗口（3 秒后开始用鼠标跟图形的小游戏），
需要人来完成（或影子跟踪接管鼠标）。弹出时主循环停止一切按键，这里循环响铃提醒，
弹窗消失 RESUME_AFTER_SECONDS 秒后停止响铃。
测谎通过后游戏弹「成功通过测谎仪的测试…」对话框（SUCCESS_KEYWORDS）：认出来就记 passed、停止响铃；
对话框关掉（连续 CLEAR_CHECKS 次没看到）就马上结束，不等 RESUME_AFTER_SECONDS——主循环据此按 Esc 关对话框、恢复打怪。
没通过（失败后被关进小黑屋）不会有这个对话框，passed 一直是 False，主循环就一直停着。

做法：后台线程每 CHECK_SECONDS 秒对最新一帧的中间区域跑一次 OCR，
识别到 KEYWORDS 里任意一个词就算弹出。用独立的 RapidOCR 实例，
不和名字定位抢同一个引擎，也不拖慢主循环。
"""
import threading
import time
from typing import Callable, Optional

import numpy as np

KEYWORDS = ("谎言探测", "测谎", "探测仪")
"""弹窗标题「谎言探测仪」、正文「测谎小游戏」里的词，OCR 认出任意一个就算弹出"""

SUCCESS_KEYWORDS = ("成功通过", "通过测谎")
"""测谎通过后的对话框「成功通过测谎仪的测试，谢谢配合。祝你冒险愉快！！」"""

CLEAR_CHECKS = 2
"""通过后对话框连续这么多次检查都没看到，才算关掉了（OCR 偶尔漏认一次，不能马上当关了）"""

ESC_TRIES = 3
"""测谎通过后按 Esc 关对话框最多试几次，还关不掉就留给人手动关"""

CHECK_SECONDS = 1.0
"""平时多久检查一次（秒）"""

RESUME_AFTER_SECONDS = 5.0
"""弹窗连续这么久没再出现才停止响铃（秒）"""

CROP_X = (0.2, 0.8)
CROP_Y = (0.1, 0.9)
"""只看画面中间这一块（弹窗出现在游戏窗口正中），按宽高比例"""


class LieDetectorWatcher:
    """用法：
        w = LieDetectorWatcher(on_log)
        w.start()
        每帧: w.feed(frame); if w.active: 停止一切动作
        w.stop()
    """

    def __init__(self, on_log: Optional[Callable[[str], None]] = None):
        self._log = on_log or (lambda m: None)
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._engine = None
        self._last_seen = 0.0
        self._frame_at = 0.0
        self.active = False   # 弹窗在（或刚消失不到 RESUME_AFTER_SECONDS 秒）
        self.passed = False   # 这次测谎看到了「成功通过」对话框
        self.passed_at = 0.0  # 最近一张还看得到通过对话框的画面是什么时候截的（time.time()）
        self._clear = 0
        self._beep_thread: Optional[threading.Thread] = None

    # ---- 公开接口 ----

    def start(self):
        if self._running:
            return
        self._running = True
        self.active = False
        self._last_seen = 0.0
        self.reset_result()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.active = False

    def feed(self, frame: np.ndarray):
        """主循环每帧调用，交给后台线程最新一帧（不复制，只换引用）。"""
        with self._lock:
            self._frame = frame
            self._frame_at = time.time()

    def reset_result(self):
        """主循环恢复打怪后调用：清掉这次测谎的结果，等下一次弹窗。"""
        self.passed = False
        self.passed_at = 0.0
        self._clear = 0

    def check(self, frame: np.ndarray) -> bool:
        """对一帧做一次检查，返回是否看到了测谎弹窗（也供测试直接调用）。"""
        return any(k in self.read(frame) for k in KEYWORDS)

    def read(self, frame: np.ndarray) -> str:
        """画面中间那块 OCR 出来的文字（OCR 不可用或出错时返回空串）。"""
        engine = self._get_engine()
        if engine is None:
            return ""
        h, w = frame.shape[:2]
        roi = frame[int(h * CROP_Y[0]):int(h * CROP_Y[1]), int(w * CROP_X[0]):int(w * CROP_X[1])]
        try:
            result, _ = engine(np.ascontiguousarray(roi))
        except Exception as e:
            self._log(f"[测谎] OCR 出错: {e}")
            return ""
        return "".join(item[1] for item in (result or []))

    # ---- 内部 ----

    def _get_engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                # 只给 2 个线程：后台慢慢查就行，别和 YOLO 抢 CPU 拖低帧率
                self._engine = RapidOCR(det_limit_type="max", det_limit_side_len=1280,
                                        use_cls=False, intra_op_num_threads=2,
                                        inter_op_num_threads=1)
            except Exception as e:
                self._log(f"[测谎] OCR 初始化失败，测谎提醒不可用: {e}")
                self._engine = False
        return self._engine or None

    def _loop(self):
        while self._running:
            t0 = time.time()
            with self._lock:
                frame, frame_at = self._frame, self._frame_at
            if frame is not None:
                self.update(self.read(frame), frame_at)
            # 弹出期间查得勤一点，弹窗一消失就停止响铃；平时 1 秒一次
            wait = (0.5 if self.active else CHECK_SECONDS) - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

    def next_step(self, esc_at: float, esc_count: int) -> str:
        """测谎通过后主循环该做什么：
          "esc"    对话框还开着，而且有一张比上次按 Esc（esc_at）更晚截的画面确认它还在——
                   什么窗口都没开时按 Esc 会弹出游戏菜单（之后按键都被它吃掉），所以不能凭旧画面按；
          "resume" 对话框关掉了，接着打怪；
          ""       没通过（一直停着），或者按了 Esc 还在等新画面，或者 ESC_TRIES 次都没关掉（等人手动关）。"""
        if not self.passed:
            return ""
        if self.active:
            return "esc" if self.passed_at > esc_at and esc_count < ESC_TRIES else ""
        return "resume"

    def update(self, text: str, frame_at: float, now: Optional[float] = None):
        """一次 OCR 结果进来，更新状态（后台线程调用；测试可以直接喂文字）。frame_at：这张画面是什么时候截的。"""
        now = time.time() if now is None else now
        seen = any(k in text for k in KEYWORDS)
        if seen:
            self._last_seen = now
            if not self.active:
                self.active = True
                self.reset_result()  # 新的一次测谎
                self._log("[测谎] 检测到谎言探测仪弹窗！已停止所有动作并响铃报警，由程序接管鼠标完成测谎（点一下鼠标左键才交还给你）")
                self._start_beep()
        if any(k in text for k in SUCCESS_KEYWORDS):
            if not self.passed:
                self._log("[测谎] 认出「成功通过测谎仪的测试」对话框，测谎通过，停止响铃")
            self.passed, self.passed_at, self._clear = True, frame_at, 0
        elif self.active and self.passed and not seen:
            self._clear += 1
            if self._clear >= CLEAR_CHECKS:
                self.active = False
                self._log("[测谎] 通过对话框已关掉，测谎结束")
        elif self.active and now - self._last_seen >= RESUME_AFTER_SECONDS:
            self.active = False
            self._log(f"[测谎] 弹窗已消失 {RESUME_AFTER_SECONDS:.0f} 秒，停止响铃。没看到「成功通过」对话框（可能没通过），"
                      f"不自动恢复打怪，请处理后手动点「停止」再点「开始」")

    def _start_beep(self):
        if self._beep_thread is not None and self._beep_thread.is_alive():
            return
        self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
        self._beep_thread.start()

    def _beep_loop(self):
        """弹窗在的时候一直响：高低两声一组，每秒一组。"""
        try:
            import winsound
        except ImportError:
            winsound = None
        while self._running and self.active and not self.passed:
            if winsound is not None:
                winsound.Beep(1500, 250)
                winsound.Beep(1000, 250)
                time.sleep(0.5)
            else:
                print("\a", end="", flush=True)
                time.sleep(1.0)
