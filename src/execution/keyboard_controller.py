"""键盘输入控制。

================================================================================
两种发送模式
================================================================================

  1. PostMessage 注入（优先）
     原理: 调用 Windows API PostMessageW() 直接向目标窗口发送
           WM_KEYDOWN / WM_KEYUP 消息，不经过全局键盘队列。
     优点: 不占用真实键盘，切换应用不影响，可以后台运行
     条件: 必须已锁定窗口（set_target_window() 已调用）

  2. keyboard 库全局发送（兜底）
     原理: 调用 keyboard.send() 模拟全局键盘输入
     缺点: 占用真实键盘，切换应用时会误触发
     条件: 未锁定窗口时使用

================================================================================
虚拟键码（VK Code）
================================================================================

  Windows 用虚拟键码标识按键，例如:
    - 字母 A: 0x41 (即 ord('A'))
    - 数字 1: 0x31
    - F1:     0x70
    - Tab:    0x09
    - Enter:  0x0D

  PostMessage 需要的参数:
    uMsg:    WM_KEYDOWN (0x0100) 或 WM_KEYUP (0x0101)
    wParam:  虚拟键码 (VK Code)
    lParam:  打包的扫描码 + 标志位

================================================================================
按键冷却
================================================================================

  _last_press 字典记录每个键上次触发的时间戳。
  press_key(key, cooldown=1.0) 在冷却时间内再次调用会返回 False，
  不会重复发送按键。

  例如: cooldown=1.5 表示按键后 1.5 秒内不会再次触发。
"""
import time
import ctypes
from ctypes import wintypes

# ---- Windows API 常量 ----
user32 = ctypes.windll.user32

WM_KEYDOWN = 0x0100  # 按键按下消息
WM_KEYUP = 0x0101    # 按键抬起消息
WM_CHAR = 0x0102     # 字符消息（一般不需要）

# ---- 虚拟键码映射表 ----
# 将可读的按键名映射到 Windows 虚拟键码
VK_MAP = {
    # 功能键
    "tab": 0x09, "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78,
    "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    # 特殊键
    "esc": 0x1B, "enter": 0x0D, "space": 0x20,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    # 方向键
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}
# 数字键 0-9: VK 0x30-0x39
for _i in range(10):
    VK_MAP[str(_i)] = 0x30 + _i
# 字母键 A-Z: VK 0x41-0x5A (即 ASCII 大写字母)
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK_MAP[_c] = ord(_c.upper())
del _i, _c  # 清理循环变量，避免污染命名空间


def _make_lparam(vk_code, scan_code, flags):
    """构造 PostMessage 的 lParam 参数。

    lParam 是一个 32 位整数，格式如下:
      bits 0-15:   重复计数
      bits 16-23:  扫描码
      bits 24-28:  扩展键标志等
      bits 29:     上下文码 (0=按下, 1=抬起)
      bits 30:     之前按键状态
      bits 31:     转换状态 (0=按下, 1=抬起)

    Args:
        vk_code:   虚拟键码
        scan_code: 硬件扫描码（通过 MapVirtualKeyW 获取）
        flags:     标志位组合
    """
    return (scan_code << 16) | flags


class KeyboardController:
    """键盘控制器。

    通过 PostMessage 直接向锁定窗口发送 WM_KEYDOWN/WM_KEYUP，
    支持按键冷却（同一键在冷却时间内不重复触发）。

    用法:
        kb = KeyboardController()
        kb.set_target_window(hwnd)  # 设置目标窗口
        kb.press_key("f")           # 按 F 键
        kb.press_key("1", cooldown=1.0)  # 按 1 键，冷却 1 秒
    """

    def __init__(self):
        self._last_press = {}  # key -> 上次触发时间戳（秒）
        self._hwnd = None      # 目标窗口句柄

    # =========================================================================
    # 窗口绑定
    # =========================================================================

    def set_target_window(self, hwnd):
        """设置目标窗口句柄。

        设置后，所有按键通过 PostMessage 直接发送到该窗口。
        未设置时回退到 keyboard 库全局发送。

        Args:
            hwnd: Windows 窗口句柄（整数）
        """
        self._hwnd = hwnd

    # =========================================================================
    # 按键
    # =========================================================================

    def _get_vk_code(self, key: str) -> int:
        """将按键名转换为 Windows 虚拟键码。

        支持:
          - 功能键名: "tab", "f1", "enter", "space" 等
          - 单字母: "a"-"z"
          - 单数字: "0"-"9"

        Args:
            key: 按键名（不区分大小写）

        Returns:
            虚拟键码，0 表示无法识别
        """
        key_lower = key.lower().strip()
        if key_lower in VK_MAP:
            return VK_MAP[key_lower]
        # 单字符直接取 ASCII 大写
        if len(key) == 1:
            vk = ord(key.upper())
            if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
                return vk
        return 0

    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        """按下指定键。

        【PostMessage 流程】
        1. 获取虚拟键码
        2. 获取硬件扫描码（MapVirtualKeyW）
        3. 构造 lParam（按下/抬起）
        4. PostMessageW(hwnd, WM_KEYDOWN, vk, lParam_down)
        5. sleep(0.02s) 让游戏有时间处理
        6. PostMessageW(hwnd, WM_KEYUP, vk, lParam_up)

        【冷却机制】
        cooldown > 0 时，在冷却时间内再次调用返回 False

        Args:
            key:      按键名
            cooldown: 冷却时间（秒），0 表示无冷却

        Returns:
            True:  按键已发送
            False: 键无效 或 在冷却中
        """
        if not key:
            return False

        # 冷却检查
        now = time.time()
        if cooldown > 0 and now - self._last_press.get(key, 0) < cooldown:
            return False
        self._last_press[key] = now

        # 获取虚拟键码
        vk_code = self._get_vk_code(key)
        if not vk_code:
            return False

        if self._hwnd:
            # ---- 方案1: PostMessage 注入 ----
            # 获取硬件扫描码
            scan = user32.MapVirtualKeyW(vk_code, 0)

            # 构造 lParam
            # 按下: flags=0x00000000 (第30位=0表示之前未按下)
            # 抬起: flags=0xC0000000 (第30位=1, 第31位=1)
            lparam_down = _make_lparam(vk_code, scan, 0x00000000)
            lparam_up = _make_lparam(vk_code, scan, 0xC0000000)

            # 发送按下消息
            user32.PostMessageW(self._hwnd, WM_KEYDOWN, vk_code, lparam_down)
            # 短暂延迟让游戏处理
            time.sleep(0.02)
            # 发送抬起消息
            user32.PostMessageW(self._hwnd, WM_KEYUP, vk_code, lparam_up)
        else:
            # ---- 方案2: keyboard 库全局发送（兜底）----
            try:
                import keyboard
                keyboard.send(key)
            except Exception:
                return False
        return True

    def can_press(self, key: str, cooldown: float = 0.0) -> bool:
        """查询某键当前是否可按下（冷却时间已过）。

        Args:
            key:      按键名
            cooldown: 冷却时间

        Returns:
            True 表示冷却已过，可以按下
        """
        now = time.time()
        return now - self._last_press.get(key, 0) >= cooldown

    def reset(self):
        """清空所有按键冷却记录，用于停止/重启时重置状态。"""
        self._last_press.clear()