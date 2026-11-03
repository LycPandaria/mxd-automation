"""鼠标输入控制。

================================================================================
原理
================================================================================

  通过 Windows API PostMessageW() 向目标窗口发送鼠标消息，
  实现后台点击（不占用真实鼠标，不干扰用户操作）。

  发送的消息序列:
    WM_LBUTTONDOWN → WM_LBUTTONUP （一次完整点击）
    WM_MOUSEMOVE → WM_LBUTTONDOWN → WM_LBUTTONUP → WM_MOUSEMOVE （带移动）

================================================================================
lParam 打包
================================================================================

  WM_MOUSEMOVE / WM_LBUTTONDOWN / WM_LBUTTONUP 的 lParam 含义:
    LOWORD(lParam) = x 坐标（窗口客户区内的像素坐标）
    HIWORD(lParam) = y 坐标

  客户区坐标: 相对于窗口内容区域左上角（不含标题栏/边框）。

================================================================================
wParam 标志
================================================================================

  MK_LBUTTON (0x0001): 鼠标左键按下
  MK_MBUTTON (0x0010): 鼠标中键按下
  MK_RBUTTON (0x0002): 鼠标右键按下
  MK_CONTROL (0x0008): Ctrl 键按下
  MK_SHIFT   (0x0004): Shift 键按下

  WM_MOUSEMOVE 时 wParam 指示当前有哪些键被按下。
  WM_LBUTTONDOWN 时 wParam 指示当前修饰键状态。

================================================================================
点击冷却
================================================================================

  为了防止连续多帧重复点击同一位置，_last_click 记录上次点击的坐标和时间。
  如果在 0.5 秒内对同一位置（±2px）重复点击，会被忽略。
"""
import time
import ctypes
from ctypes import wintypes

# ---- Windows API 常量 ----
user32 = ctypes.windll.user32

WM_MOUSEMOVE = 0x0200     # 鼠标移动
WM_LBUTTONDOWN = 0x0201   # 鼠标左键按下
WM_LBUTTONUP = 0x0202     # 鼠标左键抬起
WM_MBUTTONDOWN = 0x0207   # 鼠标中键按下
WM_MBUTTONUP = 0x0208     # 鼠标中键抬起
WM_RBUTTONDOWN = 0x0204   # 鼠标右键按下
WM_RBUTTONUP = 0x0205     # 鼠标右键抬起

MK_LBUTTON = 0x0001  # 左键按下标志
MK_MBUTTON = 0x0010  # 中键按下标志
MK_RBUTTON = 0x0002  # 右键按下标志


def _pack_lparam(x: int, y: int) -> int:
    """将 (x, y) 坐标打包为 lParam。

    lParam = (y << 16) | x
    其中 x 是低 16 位，y 是高 16 位。

    Args:
        x: 窗口客户区内的 x 像素坐标
        y: 窗口客户区内的 y 像素坐标

    Returns:
        打包后的 lParam 值
    """
    return (y << 16) | (x & 0xFFFF)


class MouseController:
    """鼠标控制器。

    通过 PostMessage 向锁定窗口发送鼠标点击/移动消息。
    支持点击冷却（同一位置短时间内不重复点击）。

    用法:
        mc = MouseController()
        mc.set_target_window(hwnd)  # 设置目标窗口
        mc.click(500, 300)          # 点击窗口内 (500, 300) 位置
        mc.move_to(500, 300)        # 移动鼠标到该位置
    """

    def __init__(self):
        self._hwnd = None                    # 目标窗口句柄
        self._last_click = (0, 0, 0.0)      # (x, y, 时间戳) 上次点击记录

    # =========================================================================
    # 窗口绑定
    # =========================================================================

    def set_target_window(self, hwnd):
        """设置目标窗口句柄。

        Args:
            hwnd: Windows 窗口句柄（整数）
        """
        self._hwnd = hwnd

    # =========================================================================
    # 鼠标移动
    # =========================================================================

    def move_to(self, x: int, y: int):
        """移动鼠标到窗口内指定坐标（不点击）。

        发送 WM_MOUSEMOVE 消息，wParam=0 表示没有按键按下。

        Args:
            x: 窗口客户区 x 坐标
            y: 窗口客户区 y 坐标
        """
        if self._hwnd:
            user32.PostMessageW(self._hwnd, WM_MOUSEMOVE, 0, _pack_lparam(x, y))

    # =========================================================================
    # 鼠标点击
    # =========================================================================

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """在窗口内指定坐标点击。

        【点击序列】
        1. WM_MOUSEMOVE  移动鼠标到目标位置
        2. WM_LBUTTONDOWN 按下左键
        3. 短暂延迟 (0.02s)
        4. WM_LBUTTONUP   抬起左键

        【冷却机制】
        同一位置（±2px）在 0.5 秒内不重复点击，
        防止多帧重复触发。

        Args:
            x:      窗口客户区 x 坐标
            y:      窗口客户区 y 坐标
            button: 按键类型 "left" / "right" / "middle"

        Returns:
            True 点击已发送，False 在冷却中
        """
        if not self._hwnd:
            return False

        # 冷却检查：同一位置 0.5 秒内不重复点击
        now = time.time()
        lx, ly, lt = self._last_click
        if abs(x - lx) <= 2 and abs(y - ly) <= 2 and now - lt < 0.5:
            return False

        self._last_click = (x, y, now)

        # 根据按键类型选择消息
        if button == "right":
            down_msg = WM_RBUTTONDOWN
            up_msg = WM_RBUTTONUP
            wparam = MK_RBUTTON
        elif button == "middle":
            down_msg = WM_MBUTTONDOWN
            up_msg = WM_MBUTTONUP
            wparam = MK_MBUTTON
        else:
            down_msg = WM_LBUTTONDOWN
            up_msg = WM_LBUTTONUP
            wparam = MK_LBUTTON

        lparam = _pack_lparam(x, y)

        # 1. 移动鼠标到目标位置
        user32.PostMessageW(self._hwnd, WM_MOUSEMOVE, 0, lparam)

        # 2. 按下
        user32.PostMessageW(self._hwnd, down_msg, wparam, lparam)
        time.sleep(0.02)  # 让游戏有时间处理按下事件

        # 3. 抬起
        user32.PostMessageW(self._hwnd, up_msg, 0, lparam)

        return True

    # =========================================================================
    # 真实鼠标（游戏界面按钮只认系统全局输入）
    # =========================================================================

    def real_move(self, x: int, y: int) -> bool:
        """真实地把光标移到窗口客户区 (x, y)（SendInput 绝对移动，和手动挪鼠标一样有移动事件），不点击。

        游戏窗口不在前台时不动、返回 False（不能在你切到别的窗口时乱挪鼠标）。
        """
        if not self._hwnd or user32.GetForegroundWindow() != self._hwnd:
            return False
        from .keyboard_controller import INPUT, _MOUSEINPUT, SendInput
        pt = wintypes.POINT(int(x), int(y))
        if not user32.ClientToScreen(self._hwnd, ctypes.byref(pt)):
            return False
        vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)      # 虚拟桌面左上角
        vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)      # 虚拟桌面宽高
        inp = INPUT(type=0, mi=_MOUSEINPUT(
            dx=int(round((pt.x - vx) * 65535 / max(1, vw - 1))),
            dy=int(round((pt.y - vy) * 65535 / max(1, vh - 1))), mouseData=0,
            dwFlags=0x0001 | 0x8000 | 0x4000, time=0, dwExtraInfo=0))  # MOVE | ABSOLUTE | VIRTUALDESK
        return SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) == 1

    def real_click(self, x: int, y: int) -> bool:
        """真实鼠标左键点击：把光标移到窗口客户区 (x, y)，用 SendInput 按下/抬起。

        游戏的界面按钮（商店、会员窗口等）不一定认 PostMessage 发来的点击，
        这里和按键一样走系统全局输入。调用前游戏窗口要在前台（ActionExecutor 会先激活）。
        """
        if not self._hwnd:
            return False
        from .keyboard_controller import INPUT, _MOUSEINPUT, SendInput
        pt = wintypes.POINT(int(x), int(y))
        if not user32.ClientToScreen(self._hwnd, ctypes.byref(pt)):
            return False

        def send(flags, dx=0, dy=0):
            inp = INPUT(type=0, mi=_MOUSEINPUT(dx=dx, dy=dy, mouseData=0,
                                               dwFlags=flags, time=0, dwExtraInfo=0))
            SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

        # 用"真实的鼠标移动"把光标移过去：只 SetCursorPos 不产生移动事件，
        # 有的游戏界面收不到悬停、点击无效
        vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
        vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
        MOVE_ABS = 0x0001 | 0x8000 | 0x4000  # MOVE | ABSOLUTE | VIRTUALDESK
        for ox in (2, 0):                    # 先移到旁边 2px 再移到目标，保证有一次移动
            send(MOVE_ABS, int((pt.x + ox - vx) * 65535 / max(1, vw - 1)),
                 int((pt.y - vy) * 65535 / max(1, vh - 1)))
            time.sleep(0.03)
        user32.SetCursorPos(pt.x, pt.y)
        time.sleep(0.15)                     # 停一下，让游戏认到鼠标悬停在按钮上
        send(0x0002)                         # LEFTDOWN
        time.sleep(0.08)
        send(0x0004)                         # LEFTUP
        return True

    def real_double_click(self, x: int, y: int) -> bool:
        """真实鼠标左键双击窗口客户区 (x, y)。

        连着两次 real_click 太慢（每次都要移动、悬停），游戏不一定认成双击。
        """
        if not self.real_click(x, y):
            return False
        from .keyboard_controller import INPUT, _MOUSEINPUT, SendInput
        for flags in (0x0002, 0x0004):       # LEFTDOWN, LEFTUP
            time.sleep(0.06)
            inp = INPUT(type=0, mi=_MOUSEINPUT(dx=0, dy=0, mouseData=0,
                                               dwFlags=flags, time=0, dwExtraInfo=0))
            SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return True

    # =========================================================================
    # 重置
    # =========================================================================

    def reset(self):
        """清空点击冷却记录。"""
        self._last_click = (0, 0, 0.0)