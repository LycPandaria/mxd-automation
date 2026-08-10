"""动作执行器。

将决策层产出的高层动作指令（"加血"、"加蓝"、"选目标"、"放技能"、
"移动到坐标"）翻译为具体的键鼠操作，并统一管理冷却与窗口绑定。

整合了原 ``app/controller.py`` 的 ``Controller`` 类的全部对外接口，
旧代码可用 ``ActionExecutor`` 无感替换 ``Controller``。
"""
from .keyboard_controller import KeyboardController
from .mouse_controller import MouseController


class ActionExecutor:
    """动作执行器：聚合键盘 + 鼠标控制器。"""

    def __init__(self):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self._hwnd = None

    # ---------------- 窗口绑定 ----------------

    def set_target_window(self, hwnd):
        """同时设置键盘和鼠标控制器的目标窗口。"""
        self._hwnd = hwnd
        self.keyboard.set_target_window(hwnd)
        self.mouse.set_target_window(hwnd)

    # ---------------- 高层动作 ----------------

    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        """按下指定键（带冷却）。委托给 KeyboardController。"""
        return self.keyboard.press_key(key, cooldown)

    def can_press(self, key: str, cooldown: float = 0.0) -> bool:
        return self.keyboard.can_press(key, cooldown)

    def click(self, x: int, y: int, button: str = "left"):
        """点击屏幕坐标。委托给 MouseController。"""
        self.mouse.click(x, y, button)

    def reset(self):
        """清空所有冷却记录。"""
        self.keyboard.reset()


# 向后兼容：旧代码用 Controller 类名
Controller = ActionExecutor
