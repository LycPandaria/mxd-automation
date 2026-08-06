"""按键 / 鼠标输入控制。

使用 Windows PostMessage / SendMessage 直接向指定窗口句柄发送按键和点击，
不经过全局键盘/鼠标队列，即使切换到其他应用也不会误触发。
"""
import time
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001

# 常用虚拟键码映射
VK_MAP = {
    "tab": 0x09, "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78,
    "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "esc": 0x1B, "enter": 0x0D, "space": 0x20, "shift": 0x10,
    "ctrl": 0x11, "alt": 0x12, "tab": 0x09,
}
# 数字键
for i in range(10):
    VK_MAP[str(i)] = 0x30 + i
# 字母键
for c in "abcdefghijklmnopqrstuvwxyz":
    VK_MAP[c] = ord(c.upper())


def _make_lparam(vk_code, scan_code, flags):
    """构造 lParam 参数。"""
    return (scan_code << 16) | flags


class Controller:
    def __init__(self):
        self._last_press = {}  # key -> 上次触发时间戳
        self._hwnd = None      # 目标窗口句柄

    def set_target_window(self, hwnd):
        """设置目标窗口句柄。"""
        self._hwnd = hwnd

    def _get_vk_code(self, key: str) -> int:
        """获取虚拟键码。"""
        key_lower = key.lower().strip()
        if key_lower in VK_MAP:
            return VK_MAP[key_lower]
        # 单字符直接取 ASCII
        if len(key) == 1:
            vk = ord(key.upper())
            if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
                return vk
        return 0

    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        """按键。通过 PostMessage 直接发到锁定窗口，即使切换应用也不影响。"""
        if not key:
            return False
        now = time.time()
        if cooldown > 0 and now - self._last_press.get(key, 0) < cooldown:
            return False
        self._last_press[key] = now

        vk_code = self._get_vk_code(key)
        if not vk_code:
            return False

        if self._hwnd:
            # 直接发到锁定窗口
            scan = user32.MapVirtualKeyW(vk_code, 0)
            lparam_down = _make_lparam(vk_code, scan, 0x00000000)
            lparam_up = _make_lparam(vk_code, scan, 0xC0000000)
            user32.PostMessageW(self._hwnd, WM_KEYDOWN, vk_code, lparam_down)
            time.sleep(0.02)
            user32.PostMessageW(self._hwnd, WM_KEYUP, vk_code, lparam_up)
        else:
            # 兜底：全局发送
            try:
                import keyboard
                keyboard.send(key)
            except Exception:
                return False
        return True

    def can_press(self, key: str, cooldown: float = 0.0) -> bool:
        now = time.time()
        return now - self._last_press.get(key, 0) >= cooldown

    def click(self, x: int, y: int, button: str = "left"):
        """在屏幕坐标点击。如果有锁定窗口，转换为窗口内坐标。"""
        if self._hwnd:
            import win32gui
            import win32con
            try:
                # 屏幕坐标转窗口客户区坐标
                point = wintypes.POINT(x, y)
                user32.ScreenToClient(self._hwnd, ctypes.byref(point))
                cx, cy = point.x, point.y
                lparam = (cy << 16) | (cx & 0xFFFF)
                user32.PostMessageW(self._hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
                time.sleep(0.02)
                user32.PostMessageW(self._hwnd, WM_LBUTTONUP, 0, lparam)
            except Exception:
                pass
        else:
            try:
                import mouse
                mouse.move(x, y, duration=0)
                mouse.click(button)
            except Exception:
                pass

    def reset(self):
        self._last_press.clear()