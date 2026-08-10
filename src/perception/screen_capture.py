"""窗口锁定与高性能截图。

锁定游戏窗口后使用 Win32 ``PrintWindow`` API 直接从窗口 DC 读取像素：
  - 不受屏幕遮挡影响（即使被其他窗口盖住也能截到）
  - 不会截到自己的工具窗
  - 适合窗口模式（可见）的游戏

若游戏是全屏独占 DirectX，需改用 ``mss`` 或 ``dxcam`` 等方案（保留接口、
替换实现即可，外部调用方代码无需改动）。

参考：项目硬约束要求截屏优先使用 mss（比 pyautogui 快 20 倍）；当前
PrintWindow 实现针对"窗口锁定 + 不受遮挡"场景，二者取舍可在 ``ScreenCapture``
子类中切换。
"""
import ctypes
import win32gui
import win32con
import numpy as np
import cv2


class ScreenCapture:
    """窗口截图基类。子类实现 ``grab()``。

    对外保持与原 ``app/window_capture.py`` 的 ``WindowCapture`` 同名方法，
    以便上层代码无感迁移。
    """

    def __init__(self):
        self._hwnd = None
        self._title = None

    # ---------------- 窗口枚举/锁定 ----------------

    @staticmethod
    def list_windows():
        """列出所有可见且有标题的窗口，返回 [(hwnd, title), ...]。"""
        results = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    results.append((hwnd, title))

        win32gui.EnumWindows(_cb, None)
        return results

    def lock(self, title=None, hwnd=None) -> str:
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

    # ---------------- 截图 ----------------

    def grab(self) -> np.ndarray:
        """截取锁定窗口画面，返回 BGR numpy 数组。

        使用 ``PrintWindow`` API 直接从窗口 DC 捕获，不受屏幕遮挡影响。
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

    # ---------------- 属性 ----------------

    @property
    def title(self):
        return self._title

    @property
    def locked(self):
        return self._hwnd is not None

    @property
    def hwnd(self):
        """锁定的窗口句柄（供执行层做 PostMessage 注入用）。"""
        return self._hwnd


# 向后兼容别名（旧代码用 WindowCapture 类名）
WindowCapture = ScreenCapture
