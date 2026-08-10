"""键盘输入控制。

使用 Windows ``PostMessage`` 直接向指定窗口句柄发送按键消息，不经过全局
键盘队列，即使切换到其他应用也不会误触发。锁定窗口未设置时回退到
``keyboard`` 库的全局发送。
"""
import time
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102

# 常用虚拟键码映射
VK_MAP = {
    "tab": 0x09, "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78,
    "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "esc": 0x1B, "enter": 0x0D, "space": 0x20, "shift": 0x10,
    "ctrl": 0x11, "alt": 0x12,
}
# 数字键
for _i in range(10):
    VK_MAP[str(_i)] = 0x30 + _i
# 字母键
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK_MAP[_c] = ord(_c.upper())
del _i, _c


def _make_lparam(vk_code, scan_code, flags):
    """构造 lParam 参数。"""
    return (scan_code << 16) | flags


class KeyboardController:
    """键盘控制器。

    通过 ``PostMessage`` 直接向锁定窗口发送 WM_KEYDOWN/WM_KEYUP，
    支持按键冷却（同一键在冷却时间内不重复触发）。
    """

    def __init__(self):
        self._last_press = {}  # key -> 上次触发时间戳
        self._hwnd = None      # 目标窗口句柄

    # ---------------- 窗口绑定 ----------------

    def set_target_window(self, hwnd):
        """设置目标窗口句柄。"""
        self._hwnd = hwnd

    # ---------------- 按键 ----------------

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
        """按键。

        通过 ``PostMessage`` 直接发到锁定窗口，即使切换应用也不影响。
        冷却时间内的重复按键会被忽略。

        Returns:
            True 表示按键已发送；False 表示键无效或在冷却中
        """
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
        """查询某键当前是否可按下（冷却时间已过）。"""
        now = time.time()
        return now - self._last_press.get(key, 0) >= cooldown

    def reset(self):
        """清空按键冷却记录。"""
        self._last_press.clear()
