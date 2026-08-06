"""按键 / 鼠标输入控制。

使用 keyboard / mouse 库做全局输入（游戏无需焦点也能触发）。
带冷却记录，避免技能被狂按。
"""
import time

import keyboard
import mouse


class Controller:
    def __init__(self):
        self._last_press = {}  # key -> 上次触发时间戳

    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        """按键。若在冷却期内则跳过，返回是否真正触发。"""
        if not key:
            return False
        now = time.time()
        if cooldown > 0 and now - self._last_press.get(key, 0) < cooldown:
            return False
        self._last_press[key] = now
        try:
            keyboard.send(key)
        except Exception:
            return False
        return True

    def can_press(self, key: str, cooldown: float = 0.0) -> bool:
        now = time.time()
        return now - self._last_press.get(key, 0) >= cooldown

    def click(self, x: int, y: int, button: str = "left"):
        """在屏幕坐标点击。"""
        try:
            mouse.move(x, y, duration=0)
            mouse.click(button)
        except Exception:
            pass

    def reset(self):
        self._last_press.clear()
