"""动作执行器 - 聚合键盘与鼠标控制。

================================================================================
职责
================================================================================

  ActionExecutor 是执行层的门面（Facade），聚合 KeyboardController
  和 MouseController，对决策层提供统一的动作接口。

  决策层只需要调用:
    executor.press_key("f")      → 按键
    executor.click(500, 300)     → 点击
    executor.set_target_window() → 绑定窗口

  不需要关心底层是 SendInput 还是 PostMessage。

================================================================================
冷却机制
================================================================================

  按键冷却和点击冷却各自独立:
    - KeyboardController 维护 _last_press 字典
    - MouseController 维护 _last_click 三元组

  两者都在各自的控制器中实现，ActionExecutor 只是透传。
"""
from typing import Optional

from .keyboard_controller import KeyboardController
from .mouse_controller import MouseController


class ActionExecutor:
    """动作执行器 - 聚合键盘和鼠标控制器。

    对决策层提供统一的动作接口，隐藏底层实现细节。

    用法:
        executor = ActionExecutor()
        executor.set_target_window(hwnd)  # 绑定窗口
        executor.press_key("f")           # 按 F 键
        executor.click(500, 300)          # 点击 (500, 300)
        executor.reset()                  # 重置所有冷却
    """

    def __init__(self, on_log=None):
        self._kb = KeyboardController(on_log=on_log)  # 键盘控制器
        self._mouse = MouseController()  # 鼠标控制器

    # =========================================================================
    # 窗口绑定
    # =========================================================================

    def set_target_window(self, hwnd: int):
        """设置目标窗口句柄，同时传给键盘和鼠标控制器。

        SendInput 模式需要绑定窗口，以便按键前激活该窗口到前台。

        Args:
            hwnd: Windows 窗口句柄
        """
        self._kb.set_target_window(hwnd)
        self._mouse.set_target_window(hwnd)

    @property
    def locked(self) -> bool:
        """是否已锁定目标窗口（可注入按键/点击）。"""
        return self._kb.locked

    # =========================================================================
    # 按键
    # =========================================================================

    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        """按下指定键（带冷却）。

        Args:
            key:      按键名，如 "f", "1", "tab"
            cooldown: 冷却时间（秒），0 表示无冷却

        Returns:
            True 按键已发送，False 在冷却中或键无效
        """
        return self._kb.press_key(key, cooldown)

    def can_press(self, key: str, cooldown: float = 0.0) -> bool:
        """查询某键当前是否可按下（冷却已过）。

        Args:
            key:      按键名
            cooldown: 冷却时间

        Returns:
            True 表示冷却已过
        """
        return self._kb.can_press(key, cooldown)

    def key_down(self, key: str) -> bool:
        """按住指定键（持续按住直到 key_up / reset）。

        用于持续移动/攀爬：按住期间角色一直移动。

        Args:
            key: 按键名，如 "right", "up"

        Returns:
            True 已按住
        """
        return self._kb.key_down(key)

    def key_up(self, key: str) -> bool:
        """释放指定键（停止移动/攀爬）。

        Args:
            key: 按键名

        Returns:
            True 已释放
        """
        return self._kb.key_up(key)

    # =========================================================================
    # 鼠标
    # =========================================================================

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """在窗口内指定坐标点击。

        Args:
            x:      窗口客户区 x 坐标
            y:      窗口客户区 y 坐标
            button: "left" / "right" / "middle"

        Returns:
            True 点击已发送，False 在冷却中
        """
        return self._mouse.click(x, y, button)

    def move_to(self, x: int, y: int):
        """移动鼠标到窗口内指定坐标（不点击）。

        Args:
            x: 窗口客户区 x 坐标
            y: 窗口客户区 y 坐标
        """
        self._mouse.move_to(x, y)

    # =========================================================================
    # 重置
    # =========================================================================

    def reset(self):
        """重置所有冷却记录（停止/重启时调用）。"""
        self._kb.reset()
        self._mouse.reset()