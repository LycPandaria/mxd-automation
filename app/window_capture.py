"""锁定游戏窗口 + 持续截图。

使用 Win32 按标题查找窗口，用 mss 截取窗口在屏幕上的区域。
适合窗口模式（可见）的游戏。若游戏是全屏独占 DirectX，需改用 DXcam 等方案。
"""
import win32gui
import mss
import numpy as np
import cv2


class WindowCapture:
    def __init__(self):
        self._hwnd = None
        self._title = None
        self._sct = mss.mss()

    def list_windows(self):
        """列出所有可见且有标题的窗口，返回 [(hwnd, title), ...]。"""
        results = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    results.append((hwnd, title))

        win32gui.EnumWindows(_cb, None)
        return results

    def lock(self, title=None, hwnd=None):
        """锁定窗口。可按标题或 hwnd。返回锁定后的窗口标题。"""
        if hwnd is None and title is None:
            raise ValueError("需要提供 title 或 hwnd")
        if hwnd is None:
            hwnd = win32gui.FindWindow(None, title)
            if not hwnd:
                raise ValueError(f"找不到窗口: {title}")
        self._hwnd = hwnd
        self._title = win32gui.GetWindowText(hwnd)
        return self._title

    def unlock(self):
        self._hwnd = None
        self._title = None

    def get_rect(self):
        """返回窗口在屏幕中的矩形 (left, top, width, height)。"""
        if not self._hwnd:
            raise RuntimeError("未锁定窗口")
        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        return (left, top, right - left, bottom - top)

    def grab(self):
        """截取锁定窗口画面，返回 BGR numpy 数组。"""
        if not self._hwnd:
            raise RuntimeError("未锁定窗口")
        left, top, w, h = self.get_rect()
        shot = self._sct.grab({"left": left, "top": top, "width": w, "height": h})
        frame = np.array(shot)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    @property
    def title(self):
        return self._title

    @property
    def locked(self):
        return self._hwnd is not None
