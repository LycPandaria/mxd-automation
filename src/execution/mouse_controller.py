"""鼠标输入控制。

使用 Windows ``PostMessage`` 直接向指定窗口句柄发送鼠标消息，
不经过全局鼠标队列，即使切换到其他应用也不会误触发。
锁定窗口未设置时回退到 ``mouse`` 库的全局操作。
"""
import time
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
MK_LBUTTON = 0x0001
MK_RBUTTON = 0x0002


class MouseController:
    """鼠标控制器。

    通过 ``PostMessage`` 向锁定窗口发送鼠标按下/抬起消息。传入坐标为
    屏幕坐标，内部使用 ``ScreenToClient`` 转为客户区坐标。
    """

    def __init__(self):
        self._hwnd = None

    # ---------------- 窗口绑定 ----------------

    def set_target_window(self, hwnd):
        """设置目标窗口句柄。"""
        self._hwnd = hwnd

    # ---------------- 点击 ----------------

    def click(self, x: int, y: int, button: str = "left"):
        """在屏幕坐标点击。

        如果有锁定窗口，转换为窗口客户区坐标后通过 ``PostMessage`` 发送。
        否则回退到 ``mouse`` 库的全局操作。

        Args:
            x, y: 屏幕坐标
            button: "left" 或 "right"
        """
        if self._hwnd:
            try:
                # 屏幕坐标转窗口客户区坐标
                point = wintypes.POINT(x, y)
                user32.ScreenToClient(self._hwnd, ctypes.byref(point))
                cx, cy = point.x, point.y
                lparam = (cy << 16) | (cx & 0xFFFF)
                down_msg = WM_LBUTTONDOWN if button == "left" else WM_RBUTTONDOWN
                up_msg = WM_LBUTTONUP if button == "left" else WM_RBUTTONUP
                wparam = MK_LBUTTON if button == "left" else MK_RBUTTON
                user32.PostMessageW(self._hwnd, down_msg, wparam, lparam)
                time.sleep(0.02)
                user32.PostMessageW(self._hwnd, up_msg, 0, lparam)
            except Exception:
                pass
        else:
            try:
                import mouse
                mouse.move(x, y, duration=0)
                mouse.click(button)
            except Exception:
                pass

    def move(self, x: int, y: int):
        """移动鼠标到屏幕坐标。

        注意：``PostMessage`` 不支持移动光标，仅能发送点击消息。
        若需要真实移动光标，会回退到 ``mouse`` 库。
        """
        try:
            import mouse
            mouse.move(x, y, duration=0)
        except Exception:
            pass
