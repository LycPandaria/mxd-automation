"""测谎触发录屏（现场取证）。

================================================================================
用途
================================================================================

  测谎弹窗命中时，本工具会立即停机并把鼠标交还给玩家（见 src/main.py）。
  但"发现弹窗的这一刻"以及玩家接手后的处置过程没有留下影像，事后无法复核：
  到底是不是误报、玩家有没有及时作答、小游戏是怎么失败的。

  本模块负责在触发后录一段固定时长的视频（默认 30 秒），存到
  logs/lie_detected/，与同时保存的 PNG 现场截图共用时间戳，方便对照。

================================================================================
实现要点
================================================================================

  1) 独立线程 + 截图函数注入
     Automation.stop() 会结束主循环，所以录屏不能挂在主循环上 —— 这里开一条
     daemon 线程，按 fps 定时调用传入的 grab_fn（即 ScreenCapture.grab，
     BitBlt 客户区截图），即便主循环已停也能继续录。

  2) 首帧决定分辨率与编码
     VideoWriter 必须在拿到真实帧尺寸后才能创建（窗口分辨率运行时才知道），
     因此第一帧到来时才打开编码器；mp4v 不可用时依次回退 XVID / MJPG。

  3) 编码失败 / 截图失败都不影响停机和报警
     录屏只是附加取证功能，任何异常都只写日志，绝不抛到主循环。

  4) 应用退出时落盘
     线程是 daemon，进程退出会被直接杀掉，VideoWriter 不 close 会留下残缺
     文件。所以 MainWindow.closeEvent 会调 Automation.shutdown() →
     recorder.close()，先把当前录像正常收尾。

================================================================================
线程安全
================================================================================

  同一时刻只允许一个录像任务：上一次还没结束就再次触发（例如玩家在 30 秒内
  重新开始又被拦下），直接忽略并写日志，避免两个 VideoWriter 同时写同一窗口。
"""
import os
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


# ---- 默认参数（可被 config 覆盖）----
DEFAULT_SECONDS = 30.0   # 触发后录制时长（秒）
DEFAULT_FPS = 10         # 录制帧率（fps 越低文件越小，10 足够看清小游戏操作）

# 编码回退顺序：(fourcc, 扩展名)。mp4v 是 opencv-python 自带 ffmpeg 的通用选择；
# 个别精简构建缺 H.264/MPEG-4 支持时回退 AVI 容器。
_CODECS: Tuple[Tuple[str, str], ...] = (
    ("mp4v", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
)


class TriggeredScreenRecorder:
    """触发式录屏：start() 后在后台录固定时长，结束自动落盘。

    用法:
        rec = TriggeredScreenRecorder(on_log=print)
        rec.start(capture.grab, out_dir, "lie_20261028_120000_123456",
                  seconds=30, fps=10)
        # ... 30 秒后自动停止并保存；程序退出前调用 rec.close()

    Args:
        on_log: 日志回调，参数 (message: str)
    """

    def __init__(self, on_log: Optional[Callable[[str], None]] = None):
        self._log = on_log or (lambda m: None)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        """当前是否有录像在进行。"""
        th = self._thread
        return th is not None and th.is_alive()

    def start(self, grab_fn: Callable[[], np.ndarray], out_dir: str,
              base_name: str, seconds: float = DEFAULT_SECONDS,
              fps: int = DEFAULT_FPS) -> bool:
        """开始录制（非阻塞）。

        Args:
            grab_fn:   无参函数，返回 BGR 帧（如 ScreenCapture.grab）
            out_dir:   输出目录（不存在会自动创建）
            base_name: 文件名主干（不含扩展名），建议与现场截图同名
            seconds:   录制时长（秒），<=0 表示不录
            fps:       录制帧率

        Returns:
            True 表示已启动；False 表示未启动（本次不录 / 上一次还在录）
        """
        if seconds is None or float(seconds) <= 0:
            return False
        fps = max(1, min(30, int(fps or DEFAULT_FPS)))

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._log("[录屏] 上一次录屏尚未结束，本次跳过")
                return False
            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(grab_fn, out_dir, base_name, float(seconds), fps),
                name="lie-recorder",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, timeout: float = 3.0):
        """请求停止录制并等待线程收尾（视频正常关闭）。"""
        self._stop_evt.set()
        th = self._thread
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout)

    def close(self):
        """程序退出前调用：停止录制并把已录内容落盘。"""
        self.stop()

    # ------------------------------------------------------------------
    # 录制线程
    # ------------------------------------------------------------------

    def _run(self, grab_fn, out_dir, base_name, seconds, fps):
        """录制循环：按 fps 抓帧写盘，到时长后正常收尾。"""
        writer = None
        size: Tuple[int, int] = (0, 0)
        path = ""
        frames = 0
        fail_streak = 0
        fail_limit = max(5, int(seconds * fps * 0.5))  # 连续失败过半才放弃
        interval = 1.0 / fps
        t_start = time.time()
        next_t = t_start

        try:
            os.makedirs(out_dir, exist_ok=True)

            while not self._stop_evt.is_set():
                if time.time() - t_start >= seconds:
                    break

                try:
                    frame = grab_fn()
                except Exception as e:
                    fail_streak += 1
                    if fail_streak <= 3:
                        self._log(f"[录屏] 截图失败（连续 {fail_streak} 次）: {e}")
                    if fail_streak >= fail_limit:
                        self._log("[录屏] 连续截图失败过多，提前结束")
                        break
                    self._stop_evt.wait(interval)
                    continue

                if frame is None or getattr(frame, "size", 0) == 0:
                    fail_streak += 1
                    self._stop_evt.wait(interval)
                    continue
                fail_streak = 0

                # 首帧：按真实尺寸打开编码器（mp4v → XVID → MJPG）
                if writer is None:
                    writer, size, path = self._open_writer(
                        out_dir, base_name, frame.shape, fps
                    )
                    if writer is None:
                        self._log("[录屏] 无法创建视频编码器，本次录屏取消")
                        return
                    self._log(
                        f"[录屏] 开始录制 {seconds:.0f}s @{fps}fps → {path}"
                    )

                h, w = frame.shape[:2]
                if (w, h) != size:
                    frame = cv2.resize(frame, size,
                                       interpolation=cv2.INTER_AREA)
                writer.write(frame)
                frames += 1

                # 按固定节拍推进；落后太多就不追帧（避免录制线程空转）
                next_t += interval
                remain = next_t - time.time()
                if remain > 0:
                    if self._stop_evt.wait(remain):
                        break
                else:
                    next_t = time.time()
        except Exception as e:
            self._log(f"[录屏] 录制异常，已中止: {e}")
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

            elapsed = time.time() - t_start
            if frames and path:
                self._log(
                    f"[录屏] 已保存 {frames} 帧 / {elapsed:.1f}s: {path}"
                )
            elif path:
                # 一帧都没写进去：删掉空文件，避免留下 0 字节"证据"
                try:
                    os.remove(path)
                except OSError:
                    pass
                self._log("[录屏] 未录到任何画面，已丢弃空文件")

            with self._lock:
                self._thread = None

    @staticmethod
    def _open_writer(out_dir: str, base_name: str, shape, fps: int):
        """按候选编码依次尝试打开 VideoWriter。

        Returns:
            (writer, (w, h), path)；全部失败时 writer 为 None
        """
        # 偶数宽高：mp4v 对奇数尺寸会写失败（部分构建直接返回未打开）
        h, w = int(shape[0]), int(shape[1])
        size = (max(2, w - w % 2), max(2, h - h % 2))
        for fourcc, ext in _CODECS:
            path = os.path.join(out_dir, base_name + ext)
            try:
                writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*fourcc), float(fps), size
                )
            except Exception:
                writer = None
            if writer is not None and writer.isOpened():
                return writer, size, path
            if writer is not None:
                writer.release()
            # 编码器打不开时 cv2 仍可能建出 0 字节文件，清掉再试下一种
            try:
                if os.path.exists(path) and os.path.getsize(path) == 0:
                    os.remove(path)
            except OSError:
                pass
        return None, size, ""
