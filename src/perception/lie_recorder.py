"""测谎录制：弹窗出现时录下游戏画面 + 鼠标光标位置，供离线评估跟踪算法。

玩家手动完成测谎时光标基本就在目标图形上，所以逐帧光标坐标可以当作目标位置的真值；
影子跟踪接管过鼠标的那次光标是程序放的，meta.json 里 cursor_reference_valid = false。
截图（BitBlt）截不到光标，光标位置单独用 GetCursorPos 读取后换算成客户区坐标。

输出到 lie_recordings/<时间>/：
  - video.avi    画面中间那块（弹窗所在区域，范围见 CROP_X / CROP_Y），MJPG
  - frames.csv   每帧：序号、相对开始的秒数、光标坐标（客户区 / 视频内）、左键是否按下
  - meta.json    窗口尺寸、裁剪区域、目标/实际帧率、丢帧数、开始/结束原因、结果
  - log.txt      这一局的日志：弹窗前 LOG_BEFORE_S 秒到录完后 LOG_AFTER_S 秒
另外每局往 lie_recordings/results.csv 加一行，看总共过了几局。
截图和写盘分两个线程，写盘慢时丢帧（记在 dropped），不拖慢截图节奏。
"""
import csv
import json
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

CROP_X = (0.15, 0.85)
CROP_Y = (0.05, 0.95)
"""只录画面中间（测谎弹窗在游戏窗口正中），按宽高比例；四周的地图/UI 用不上"""

JPEG_QUALITY = 95
"""淡出后的目标只剩很淡的轮廓，压得狠了离线就看不出来（两帧对比模型靠的就是这点差别）"""

LOG_BEFORE_S = 120.
"""存弹窗前这么久的日志：认出晚了的话，原因（比如正在补飞镖）在弹窗之前"""
LOG_AFTER_S = 30.
"""录完后再记这么久，关对话框、接着打怪或停住的日志都在这段"""
RESULTS_FILE = "results.csv"


def client_cursor(hwnd) -> Optional[Tuple[int, int, bool]]:
    """光标在窗口客户区里的坐标 + 左键是否按下；非 Windows 或读取失败返回 None。"""
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
    except (ImportError, AttributeError):
        return None
    pt = ctypes.wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)) or not user32.ScreenToClient(hwnd, ctypes.byref(pt)):
        return None
    left_down = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
    return pt.x, pt.y, left_down


class LieLog:
    """日志转发给 on_log，同时记下最近的几行；open 之后再写进这一局的 log.txt。
    各个线程都会写日志，所以要加锁；文件写不了不影响日志本身。"""

    def __init__(self, on_log: Callable[[str], None], clock: Callable[[], float] = time.time):
        self._on_log = on_log
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: "deque" = deque(maxlen=2000)
        self._file = None
        self._close_at: Optional[float] = None

    def write(self, message: str):
        with self._lock:
            now = self._clock()
            line = f"{time.strftime('%H:%M:%S', time.localtime(now))} {message}"
            self._recent.append((now, line))
            if self._file is not None and self._close_at is not None and now > self._close_at:
                self._close()
            self._put(line)
        self._on_log(message)

    def open(self, out_dir: str):
        """这一局开始：写入弹窗前的日志，之后的日志接着写。"""
        with self._lock:
            self._close()
            try:
                self._file = open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8")
            except OSError:
                return
            start = self._clock() - LOG_BEFORE_S
            for at, line in self._recent:
                if at >= start:
                    self._put(line)

    def finish(self):
        """录完了：再记 LOG_AFTER_S 秒。"""
        with self._lock:
            self._close_at = self._clock() + LOG_AFTER_S

    def close(self):
        """立即停止写文件（程序退出时用；正常情况录完后下一次写日志会检查是否该关）。"""
        with self._lock:
            self._close()

    def _put(self, line: str):
        if self._file is None:
            return
        try:
            self._file.write(line + "\n")
            self._file.flush()
        except OSError:
            self._close()

    def _close(self):
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        self._file, self._close_at = None, None


class LieRecorder:
    """用法：
        r = LieRecorder(capture.grab, lambda: client_cursor(capture.hwnd), on_log)
        测谎弹出: r.start("检测到弹窗")
        弹窗消失: r.stop("弹窗消失")
    超过 max_seconds 会自己停。
    """

    def __init__(self, grab: Callable[[], np.ndarray],
                 cursor: Callable[[], Optional[Tuple[int, int, bool]]],
                 on_log: Optional[Callable[[str], None]] = None,
                 out_root: str = "lie_recordings", fps: float = 60.,
                 max_seconds: float = 90., shadow=None, run_log: Optional[LieLog] = None,
                 result: Optional[Callable[[], str]] = None):
        self._grab = grab
        self._cursor = cursor
        self._log = on_log or (lambda m: None)
        self.out_root = out_root
        self.fps = fps
        self.max_seconds = max_seconds
        self.shadow = shadow  # 可选的影子跟踪（LieShadow）：同步拿到每帧，结果写到同一目录
        self.run_log = run_log  # 可选：这一局的日志存到录像目录
        self._result = result or (lambda: "")  # 这一局的结果，录完时问一次
        self._stop = threading.Event()
        self._queue: "queue.Queue" = queue.Queue(maxsize=int(fps))
        self._grab_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None
        self.out_dir: Optional[str] = None
        self._meta: dict = {}

    @property
    def recording(self) -> bool:
        return self._write_thread is not None and self._write_thread.is_alive()

    def start(self, reason: str = "") -> Optional[str]:
        """开始录制，返回输出目录；已在录制时什么也不做。"""
        if self.recording:
            return None
        self.out_dir = os.path.join(self.out_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.out_dir, exist_ok=True)
        self._stop.clear()
        self._queue = queue.Queue(maxsize=int(self.fps))  # 最多积 1 秒整帧，写盘跟不上就丢帧（记在 dropped）
        self._meta = {"started_at": datetime.now().isoformat(timespec="seconds"),
                      "start_reason": reason, "target_fps": self.fps,
                      "crop_x": CROP_X, "crop_y": CROP_Y, "dropped": 0,
                      "cursor_note": "client_x/y 为游戏窗口客户区坐标，video_x/y = client 减去 crop_left/top；"
                                     "截图里没有光标，光标位置只在 frames.csv 里"}
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        if self.shadow is not None:
            self.shadow.start(self.out_dir)
        if self.run_log is not None:
            self.run_log.open(self.out_dir)
        self._write_thread.start()
        self._grab_thread.start()
        self._log(f"[测谎录制] 开始录制 → {self.out_dir}（画面 + 鼠标位置）")
        return self.out_dir

    def stop(self, reason: str = ""):
        """停止并等写盘结束（最多几秒）。"""
        if not self.recording:
            return
        self._meta.setdefault("stop_reason", reason)
        self._meta.setdefault("result", self._result())
        self._stop.set()
        for t in (self._grab_thread, self._write_thread):
            if t is not None and t is not threading.current_thread():
                t.join(timeout=10.)

    # ---- 内部 ----

    def _grab_loop(self):
        interval = 1. / self.fps
        t_start = time.perf_counter()
        next_tick = t_start
        index = 0
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                if now - t_start >= self.max_seconds:
                    self._meta.setdefault("stop_reason", f"超过最长录制时间 {self.max_seconds:.0f} 秒")
                    break
                try:
                    frame = self._grab()
                except Exception as e:
                    self._log(f"[测谎录制] 截图失败: {e}")
                    frame = None
                if frame is not None:
                    item = (index, now - t_start, frame, self._cursor())
                    if self.shadow is not None:
                        self.shadow.offer(*item)
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        self._meta["dropped"] += 1
                    index += 1
                next_tick += interval
                wait = next_tick - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_tick = time.perf_counter()  # 跟不上就别补拍，按实际节奏继续
        finally:
            if self.shadow is not None:
                self.shadow.stop()  # 先停影子，写 meta 时才知道程序最后有没有拿过鼠标
            self._queue.put(None)

    def _write_loop(self):
        writer = None
        crop = None
        rows = 0
        last_t = 0.
        csv_file = open(os.path.join(self.out_dir, "frames.csv"), "w", newline="", encoding="utf-8")
        try:
            out = csv.writer(csv_file)
            out.writerow(["frame", "seconds", "client_x", "client_y", "video_x", "video_y", "left_down"])
            while True:
                item = self._queue.get()
                if item is None:
                    break
                index, t, frame, cursor = item
                if writer is None:
                    h, w = frame.shape[:2]
                    crop = (int(w * CROP_X[0]), int(h * CROP_Y[0]),
                            int(w * CROP_X[1]), int(h * CROP_Y[1]))
                    size = (crop[2] - crop[0], crop[3] - crop[1])
                    writer = cv2.VideoWriter(os.path.join(self.out_dir, "video.avi"),
                                             cv2.VideoWriter_fourcc(*"MJPG"), self.fps, size)
                    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, JPEG_QUALITY)
                    self._meta.update(window_size=[w, h], crop_left_top_right_bottom=list(crop))
                    self._meta["first_size"] = (w, h)
                if frame.shape[1::-1] != self._meta["first_size"]:
                    frame = cv2.resize(frame, self._meta["first_size"])  # 窗口中途变大小也保持同一坐标系
                writer.write(np.ascontiguousarray(frame[crop[1]:crop[3], crop[0]:crop[2]]))
                if cursor is None:
                    out.writerow([index, f"{t:.4f}", "", "", "", "", ""])
                else:
                    x, y, down = cursor
                    out.writerow([index, f"{t:.4f}", x, y, x - crop[0], y - crop[1], int(down)])
                rows += 1
                last_t = t
        finally:
            if writer is not None:
                writer.release()
            csv_file.close()
            self._meta.pop("first_size", None)
            self._meta.update(frames=rows, duration_seconds=round(last_t, 3),
                              actual_fps=round(rows / last_t, 2) if last_t > 0 else None,
                              stopped_at=datetime.now().isoformat(timespec="seconds"),
                              video="video.avi" if writer is not None else None)
            self._meta.setdefault("stop_reason", "")
            self._meta.setdefault("result", self._result())
            if getattr(self.shadow, "took_over", False):
                self._meta.update(program_moved_mouse=True, cursor_reference_valid=False,
                                  cursor_reference_note="程序接管过鼠标，光标是程序放的、不是玩家跟的，不能当答案（用回放标注）")
            offset = getattr(self.shadow, "crosshair_offset", None)
            if offset is not None:
                self._meta["crosshair_offset"] = list(offset)  # 准星相对光标，平时约 (0, 0)，偏的局约 (30, 30)
            with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, indent=2)
            self._add_result_row()
            self._log(f"[测谎录制] 已保存 {rows} 帧（{last_t:.1f} 秒，丢帧 {self._meta['dropped']}）"
                      f"→ {self.out_dir}，把整个文件夹打包发过来即可")
            if self.run_log is not None:
                self.run_log.finish()

    def _add_result_row(self):
        """results.csv 加一行。带 BOM，Excel 打开中文不乱码；正被 Excel 开着时写不进去，只记日志。"""
        path = os.path.join(self.out_root, RESULTS_FILE)
        meta = self._meta
        offset = meta.get("crosshair_offset")
        try:
            is_new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                out = csv.writer(f)
                if is_new:
                    out.writerow(["开始时间", "结果", "结束原因", "录了几秒", "程序拿过鼠标", "文件夹", "准星偏移"])
                out.writerow([meta.get("started_at", ""), meta.get("result", ""), meta.get("stop_reason", ""),
                              meta.get("duration_seconds", ""), "是" if meta.get("program_moved_mouse") else "否",
                              os.path.basename(self.out_dir), "" if offset is None else f"({offset[0]}, {offset[1]})"])
        except OSError as e:
            self._log(f"[测谎录制] {RESULTS_FILE} 写不进去（是不是开着？）: {e}")
