"""锁定游戏窗口 + 持续截图。

使用 Win32 按标题查找窗口，用 PrintWindow API 直接按窗口句柄捕获。
PrintWindow 直接从窗口 DC 读取像素，不受屏幕遮挡影响，不会把自己的工具窗截进去。
适合窗口模式（可见）的游戏。若游戏是全屏独占 DirectX，需改用 DXcam 等方案。
"""
import ctypes
import win32gui
import win32con
import numpy as np
import cv2


class WindowCapture:
    def __init__(self):
        self._hwnd = None
        self._title = None

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
        """截取锁定窗口画面，返回 BGR numpy 数组。

        使用 PrintWindow API 直接从窗口 DC 捕获，不受屏幕遮挡影响。
        """
        if not self._hwnd:
            raise RuntimeError("未锁定窗口")

        hwnd = self._hwnd
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            raise RuntimeError("窗口尺寸无效")

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        hwnd_dc = user32.GetWindowDC(hwnd)
        if not hwnd_dc:
            raise RuntimeError("获取窗口 DC 失败")

        mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        save_bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        gdi32.SelectObject(mfc_dc, save_bitmap)

        PW_RENDERFULLCONTENT = 0x00000002
        result = user32.PrintWindow(hwnd, mfc_dc, PW_RENDERFULLCONTENT)

        if not result:
            # 兜底：使用 BitBlt 从窗口 DC 拷贝
            result = gdi32.BitBlt(
                mfc_dc, 0, 0, width, height, hwnd_dc, 0, 0, win32con.SRCCOPY
            )

        # 读取位图数据
        bmi = ctypes.create_string_buffer(32)
        gdi32.GetObjectA(save_bitmap, 32, bmi)
        data_size = ((width * 32 + 31) // 32) * 4 * height
        bmp_data = ctypes.create_string_buffer(data_size)
        gdi32.GetBitmapBits(save_bitmap, data_size, bmp_data)

        # 释放资源
        gdi32.DeleteObject(save_bitmap)
        gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        # 转换为 numpy 数组 (BGRA -> BGR)
        arr = np.frombuffer(bmp_data.raw, dtype=np.uint8)
        # 每行对齐到 4 字节
        row_size = ((width * 32 + 31) // 32) * 4
        arr = arr.reshape(height, row_size)[:, :width * 4]
        frame = arr.reshape(height, width, 4)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    @property
    def title(self):
        return self._title

    @property
    def locked(self):
        return self._hwnd is not None
