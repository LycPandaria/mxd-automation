"""键盘输入控制。

================================================================================
发送模式
================================================================================

  本模块支持两种按键注入模式，通过构造参数 mode 选择（默认 sendinput）:

  ┌──────────────┬────────────────────────────────────────────────────────────┐
  │ mode         │ 说明                                                        │
  ├──────────────┼────────────────────────────────────────────────────────────┤
  │ sendinput    │ 前台真实按键（默认）                                        │
  │              │   原理: SendInput() 从驱动层模拟真实全局按键，经过系统全局  │
  │              │         键盘输入队列，DirectX/DirectInput 游戏可以收到。    │
  │              │   优点: 对冒险岛这类读取全局键盘状态的游戏有效              │
  │              │          （PostMessage 注入对这类游戏无效，已验证）         │
  │              │   代价: 占用真实键盘，游戏窗口必须处于前台/激活状态，       │
  │              │         运行期间不能切换去操作其它程序                      │
  ├──────────────┼────────────────────────────────────────────────────────────┤
  │ postmessage  │ 后台注入                                                    │
  │              │   原理: PostMessageW() 向目标窗口消息队列投递              │
  │              │         WM_KEYDOWN / WM_KEYUP，游戏在窗口过程(WndProc)     │
  │              │         中处理这些消息。                                    │
  │              │   优点: 纯后台输入，窗口无需在前台，可自由切换程序。        │
  │              │   代价: 只对通过"窗口消息"接收键盘的游戏有效；             │
  │              │         若游戏用 DirectInput / GetAsyncKeyState 读键盘     │
  │              │         （冒险岛就是这样），此模式完全无效。                │
  └──────────────┴────────────────────────────────────────────────────────────┘

  【重要】两种模式都只在锁定窗口后生效。
  未锁定窗口时 press_key() 会返回 False 并记录日志。

  PostMessage 模式的 lParam 按标准键盘消息编码:
    bit 0-15  重复计数(repeat count) = 1
    bit 16-23 硬件扫描码(scan code)
    bit 24    扩展键标志(extended key，方向键等)
    bit 29    上下文代码(context code)
    bit 30    先前键状态(previous key state，KEYUP 时=1)
    bit 31    转换状态(transition state，KEYUP 时=1)

================================================================================
虚拟键码（VK Code）
================================================================================

  Windows 用虚拟键码标识按键，例如:
    - 字母 A: 0x41 (即 ord('A'))
    - 数字 1: 0x31
    - F1:     0x70
    - Tab:    0x09
    - Enter:  0x0D

  两种模式都依赖虚拟键码:
    - SendInput:    扫描码由 MapVirtualKeyW(vk, 0) 获得（KEYEVENTF_SCANCODE）
    - PostMessage:  wParam=虚拟键码，lParam 内嵌扫描码

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

# ---- 模式常量 ----
MODE_SENDINPUT = "sendinput"     # 前台真实按键（默认，冒险岛有效）
MODE_POSTMESSAGE = "postmessage" # 后台注入（仅对走窗口消息的游戏有效）

# ---- 窗口消息常量（PostMessage 模式）----
WM_KEYDOWN = 0x0100  # 按键按下消息
WM_KEYUP = 0x0101    # 按键抬起消息
WM_CHAR = 0x0102     # 字符消息（一般不需要）

# ---- PostMessage lParam 位标志 ----
# bit 24: 扩展键标志（方向键/Insert/Delete 等有独立扩展扫描码）
_LPARAM_EXTENDED = 0x01000000
# bit 30: 先前键状态（KEYUP 时为 1）
_LPARAM_PREV_DOWN = 0x40000000
# bit 31: 转换状态（KEYUP 时为 1）
_LPARAM_TRANSITION = 0x80000000

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

    支持两种注入模式（构造参数 mode 指定）:
      - "sendinput"（默认）: 前台真实按键，游戏窗口必须在前台。
        冒险岛怀旧服读取全局键盘状态，必须使用此模式。
      - "postmessage": 后台注入，无需前台，但对冒险岛无效。

    用法:
        kb = KeyboardController(mode="sendinput")
        kb.set_target_window(hwnd)  # 设置目标窗口
        kb.press_key("f")           # 按 F 键（点按）
        kb.key_down("right")        # 按住右方向键（持续移动）
        kb.key_up("right")          # 释放右方向键（停止移动）
    """

    def __init__(self, mode: str = MODE_SENDINPUT, on_log=None):
        """构造键盘控制器。

        Args:
            mode:   "sendinput"（默认，前台真实按键）或 "postmessage"（后台注入）
            on_log: 日志回调 (message: str) -> None
        """
        if mode not in (MODE_SENDINPUT, MODE_POSTMESSAGE):
            mode = MODE_SENDINPUT
        self._mode = mode
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

        两种模式差异:
          - sendinput: 锁定后每次按键前会自动把游戏窗口激活到前台
          - postmessage: 仅投递消息到该窗口，无需激活（对冒险岛无效）

        Args:
            hwnd: Windows 窗口句柄（整数）
        """
        self._hwnd = hwnd
        if hwnd:
            if self._mode == MODE_SENDINPUT:
                self._on_log(
                    f"[按键] 目标窗口已锁定, hwnd=0x{hwnd:X}"
                    "（SendInput 真实按键模式: 请保持游戏窗口在前台）"
                )
            else:
                self._on_log(
                    f"[按键] 目标窗口已锁定, hwnd=0x{hwnd:X}"
                    "（PostMessage 后台注入模式: 无需保持游戏窗口在前台）"
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
    # PostMessage 实现（后台注入，可选模式）
    # -------------------------------------------------------------------------

    def _make_lparam(self, vk_code: int, keyup: bool = False) -> int:
        """构造键盘消息的 lParam（扫描码 + 扩展键 + 按键状态标志）。"""
        scan = user32.MapVirtualKeyW(vk_code, 0)
        lparam = 1  # 重复计数 = 1
        lparam |= scan << 16
        if vk_code in _EXTENDED_KEYS:
            lparam |= _LPARAM_EXTENDED
        if keyup:
            lparam |= _LPARAM_PREV_DOWN | _LPARAM_TRANSITION
        return lparam

    def _post_key(self, key: str, vk_code: int, keyup: bool) -> bool:
        """向目标窗口投递一条键盘消息（PostMessage 异步）。"""
        msg = WM_KEYUP if keyup else WM_KEYDOWN
        lparam = self._make_lparam(vk_code, keyup)
        ok = bool(user32.PostMessageW(self._hwnd, msg, vk_code, lparam))
        if not ok:
            self._on_log(
                f"[按键] PostMessage 投递失败: key={key}, up={keyup}"
            )
        return ok

    # -------------------------------------------------------------------------
    # 点按 / 按住 / 释放（按模式分发）
    # -------------------------------------------------------------------------

    def _press_single_key(self, key: str) -> bool:
        """点按单个按键。

        SendInput 流程:    激活窗口 → KEYDOWN → sleep → KEYUP
        PostMessage 流程:  投递 KEYDOWN → sleep → 投递 KEYUP

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

        if self._mode == MODE_POSTMESSAGE:
            # PostMessage: 异步投递，无需激活窗口
            if not self._post_key(key, vk_code, keyup=False):
                return False
            time.sleep(0.03)  # 让游戏有时间处理按下事件
            return self._post_key(key, vk_code, keyup=True)

        # SendInput: 真实全局按键，需激活窗口到前台
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

        if self._mode == MODE_POSTMESSAGE:
            ok = self._post_key(key, vk_code, keyup=False)
        else:
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

        if self._mode == MODE_POSTMESSAGE:
            ok = self._post_key(key, vk_code, keyup=True)
        else:
            ok = self._si_send_key(key, vk_code, keyup=True)
        if not ok:
            self._on_log(f"[按键] 释放失败: key={key}")
            return False
        self._held_keys.discard(key)
        return True

    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        """按下指定键，支持多字符序列（如 "hm" 依次按 h、m）。

        【SendInput 流程】
        1. 获取虚拟键码
        2. 激活窗口到前台（SetForegroundWindow）
        3. 获取硬件扫描码（MapVirtualKeyW）
        4. SendInput 发送 KEYDOWN（扫描码模式，扩展键加标志）
        5. sleep(0.03s) 让游戏有时间处理
        6. SendInput 发送 KEYUP

        【PostMessage 流程】
        1. 获取虚拟键码
        2. 投递 WM_KEYDOWN 到目标窗口（后台注入，无需前台）
        3. sleep(0.03s) 让游戏有时间处理
        4. 投递 WM_KEYUP

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
