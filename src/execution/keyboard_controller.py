"""键盘输入控制。

================================================================================
发送模式
================================================================================

  使用 SendInput() 从驱动层模拟真实全局按键，经过系统全局键盘输入队列，
  DirectX/DirectInput 游戏可以收到。

  优点: 对冒险岛这类读取全局键盘状态的游戏有效。
  代价: 占用真实键盘，游戏窗口必须处于前台/激活状态，
        运行期间不能切换去操作其它程序。

  只在锁定窗口后生效。未锁定窗口时 press_key() 会返回 False 并记录日志。

================================================================================
虚拟键码（VK Code）
================================================================================

  Windows 用虚拟键码标识按键，例如:
    - 字母 A: 0x41 (即 ord('A'))
    - 数字 1: 0x31
    - F1:     0x70
    - Tab:    0x09
    - Enter:  0x0D

  SendInput 使用扫描码模式: 扫描码由 MapVirtualKeyW(vk, 0) 获得（KEYEVENTF_SCANCODE）。

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

# ---- SendInput 常量 ----
INPUT_KEYBOARD = 1               # INPUT 类型: 键盘输入

KEYEVENTF_EXTENDEDKEY = 0x0001   # 扩展键标志（方向键/Insert/Delete 等）
KEYEVENTF_KEYUP = 0x0002         # 键抬起标志
KEYEVENTF_SCANCODE = 0x0008      # 使用硬件扫描码（更接近真实硬件按键）

SW_RESTORE = 9                   # ShowWindow: 恢复窗口

# ---- SendInput 结构体 ----
class _KEYBDINPUT(ctypes.Structure):
    """键盘输入结构。"""
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),  # ULONG_PTR
    ]


class _MOUSEINPUT(ctypes.Structure):
    """鼠标输入结构（键盘模式不使用，占位对齐用）。"""
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    """硬件输入结构（不使用，占位对齐用）。"""
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    """输入联合体（键盘/鼠标/硬件，键盘模式用 ki）。"""
    _fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    """SendInput 的输入结构（type + 联合体）。

    _anonymous_ = ("u",) 使联合体字段提升到结构体顶层，
    即 INPUT(type=..., ki=...) 直接可用；缺失会导致
    "no field named 'ki'" 的 TypeError（SendInput 静默失效）。
    """
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUTUNION),
    ]


SendInput = user32.SendInput
SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), wintypes.UINT)
SendInput.restype = wintypes.UINT

# 需要扩展键标志的键（有独立的扩展扫描码）
_EXTENDED_KEYS = {
    0x21,  # PageUp
    0x22,  # PageDown
    0x23,  # End
    0x24,  # Home
    0x25,  # Left
    0x26,  # Up
    0x27,  # Right
    0x28,  # Down
    0x2D,  # Insert
    0x2E,  # Delete
}

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
    # 编辑/翻页键（独立物理按键，不是多字母组合）
    "end": 0x23, "home": 0x24,
    "pup": 0x21, "pgup": 0x21, "pdn": 0x22, "pgdn": 0x22,
    "ins": 0x2D, "insert": 0x2D, "del": 0x2E, "delete": 0x2E,
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


class KeyboardController:
    """键盘控制器。

    使用 SendInput 从驱动层模拟真实全局按键。冒险岛怀旧服读取全局键盘状态，
    游戏窗口必须保持在前台激活状态。

    用法:
        kb = KeyboardController()
        kb.set_target_window(hwnd)  # 设置目标窗口
        kb.press_key("f")           # 按 F 键（点按）
        kb.key_down("right")        # 按住右方向键（持续移动）
        kb.key_up("right")          # 释放右方向键（停止移动）
    """

    def __init__(self, on_log=None):
        """构造键盘控制器。

        Args:
            on_log: 日志回调 (message: str) -> None
        """
        self._last_press = {}  # key -> 上次触发时间戳（秒）
        self._held_keys = set()  # 当前被按住未释放的键
        self._hwnd = None      # 目标窗口句柄
        self._on_log = on_log or (lambda msg: None)

    # =========================================================================
    # 窗口绑定
    # =========================================================================

    @property
    def locked(self) -> bool:
        """是否已锁定目标窗口（可注入按键）。"""
        return self._hwnd is not None

    def set_target_window(self, hwnd):
        """设置目标窗口句柄。

        锁定后每次按键前会自动把游戏窗口激活到前台（SendInput 全局按键，
        只有前台窗口才能收到）。

        Args:
            hwnd: Windows 窗口句柄（整数）
        """
        self._hwnd = hwnd
        if hwnd:
            self._on_log(
                f"[按键] 目标窗口已锁定, hwnd=0x{hwnd:X}"
                "（请保持游戏窗口在前台）"
            )

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

    # -------------------------------------------------------------------------
    # SendInput 实现（前台真实按键，默认模式）
    # -------------------------------------------------------------------------

    def _si_build_input(self, vk_code: int, keyup: bool = False) -> INPUT:
        """构造 SendInput 键盘输入结构（扫描码模式）。"""
        scan = user32.MapVirtualKeyW(vk_code, 0)
        flags = KEYEVENTF_SCANCODE
        if vk_code in _EXTENDED_KEYS:
            flags |= KEYEVENTF_EXTENDEDKEY
        if keyup:
            flags |= KEYEVENTF_KEYUP
        return INPUT(
            type=INPUT_KEYBOARD,
            ki=_KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags),
        )

    def _activate_window(self) -> bool:
        """将锁定窗口激活到前台。

        SendInput 是全局输入，只有前台窗口才能收到按键。
        通过 恢复窗口 + 释放前台锁 + SetForegroundWindow 激活。
        若激活失败（被 Windows 前台锁限制），仅记录日志，不阻断按键。

        Returns:
            True 激活成功 / False 激活失败
        """
        if not self._hwnd:
            return False
        try:
            user32.ShowWindow(self._hwnd, SW_RESTORE)
            # 空按键释放 Windows 的"前台锁定"，提高 SetForegroundWindow 成功率
            user32.keybd_event(0, 0, 0, 0)
            ok = bool(user32.SetForegroundWindow(self._hwnd))
            if not ok:
                self._on_log("[按键] 激活窗口失败，请手动把游戏窗口点到前台")
            return ok
        except Exception as e:
            self._on_log(f"[按键] 激活窗口异常: {e}")
            return False

    def _si_send_key(self, key: str, vk_code: int, keyup: bool) -> bool:
        """SendInput 发送一次键盘事件（down 或 up）。

        Returns:
            True 发送成功 / False 失败
        """
        self._activate_window()
        inp = self._si_build_input(vk_code, keyup=keyup)
        sent = SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if not sent:
            self._on_log(
                f"[按键] SendInput 发送失败 (err={ctypes.get_last_error()}): "
                f"key={key}, up={keyup}"
            )
            return False
        return True

    # -------------------------------------------------------------------------
    # 点按 / 按住 / 释放
    # -------------------------------------------------------------------------

    def _press_single_key(self, key: str) -> bool:
        """点按单个按键。

        流程: 激活窗口 → KEYDOWN → sleep → KEYUP

        Returns:
            True: 按键已发送
            False: 键无效 / 未锁定窗口 / 发送失败
        """
        vk_code = self._get_vk_code(key)
        if not vk_code:
            self._on_log(f"[按键] 无法识别的按键: {key}")
            return False

        if not self._hwnd:
            self._on_log(f"[按键] 未锁定窗口，拒绝发送按键: {key}")
            return False

        if not self._si_send_key(key, vk_code, keyup=False):
            return False
        time.sleep(0.03)  # 让游戏有时间处理按下事件
        return self._si_send_key(key, vk_code, keyup=True)

    def key_down(self, key: str) -> bool:
        """按住指定键（持续按住直到 key_up / reset）。

        用于持续移动/攀爬：按住期间角色会一直移动，
        直到调用 key_up() 释放或 reset() 统一释放。

        Args:
            key: 按键名，如 "right", "up"

        Returns:
            True: 已按住；False: 键无效 / 未锁定窗口 / 发送失败
        """
        vk_code = self._get_vk_code(key)
        if not vk_code:
            self._on_log(f"[按键] 无法识别的按键: {key}")
            return False
        if not self._hwnd:
            self._on_log(f"[按键] 未锁定窗口，拒绝发送按键: {key}")
            return False
        if key in self._held_keys:
            return True  # 已按住，避免重复发送 KEYDOWN

        ok = self._si_send_key(key, vk_code, keyup=False)
        if not ok:
            self._on_log(f"[按键] 按住失败: key={key}")
            return False
        self._held_keys.add(key)
        return True

    def key_up(self, key: str) -> bool:
        """释放指定键。

        Args:
            key: 按键名

        Returns:
            True: 已释放（或本就未按住）；False: 键无效 / 发送失败
        """
        if key not in self._held_keys:
            return True  # 未按住，无需释放
        vk_code = self._get_vk_code(key)
        if not vk_code:
            return False

        ok = self._si_send_key(key, vk_code, keyup=True)
        if not ok:
            self._on_log(f"[按键] 释放失败: key={key}")
            return False
        self._held_keys.discard(key)
        return True

    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        """按下指定键，支持多字符序列（如 "hm" 依次按 h、m）。

        【流程】
        1. 获取虚拟键码
        2. 激活窗口到前台（SetForegroundWindow）
        3. 获取硬件扫描码（MapVirtualKeyW）
        4. SendInput 发送 KEYDOWN（扫描码模式，扩展键加标志）
        5. sleep(0.03s) 让游戏有时间处理
        6. SendInput 发送 KEYUP

        【冷却机制】
        cooldown > 0 时，在冷却时间内再次调用返回 False。
        冷却作用于整个 key 序列（如 "hm" 整体冷却），而不是单个字母。

        Args:
            key:      按键名，如 "f", "tab", "hm"（多字符会依次发送每个键）
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

        # 多字符序列（如 "hm"）：先判断是否为映射表中的多字符功能键名
        key_lower = key.lower().strip()
        if len(key) > 1 and key_lower not in VK_MAP:
            for k in key:
                if not self._press_single_key(k):
                    return False
            return True

        return self._press_single_key(key)

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
        """清空所有按键冷却，并释放所有被按住的键（停止时防止方向键卡住）。"""
        for key in list(self._held_keys):
            self.key_up(key)
        self._held_keys.clear()
        self._last_press.clear()