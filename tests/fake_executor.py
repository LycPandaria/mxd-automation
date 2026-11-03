"""测试用的假执行器：记录按键，不碰真实键鼠。

接口按本地 ActionExecutor 当前签名实现（key_up 支持 force），
不依赖 Windows / SendInput，可以在无游戏环境跑决策逻辑。
"""
from typing import List, Tuple


class FakeExecutor:
    def __init__(self):
        self.pressed: List[Tuple[str, float]] = []  # (按键, 传入的冷却)
        self.down: List[str] = []                   # key_down 过的键（按顺序）
        self.held: set = set()                      # 当前按住的方向键
        self.reset_count = 0

    # ---- 点按 ----
    def press_key(self, key: str, cooldown: float = 0.0) -> bool:
        self.pressed.append((key, cooldown))
        return True

    def can_press(self, key: str, cooldown: float = 0.0) -> bool:
        return True  # 测试里不模拟冷却，需要时由用例自己控制

    # ---- 按住 / 释放 ----
    def key_down(self, key: str) -> bool:
        self.down.append(key)
        self.held.add(key)
        return True

    def key_up(self, key: str, force: bool = False) -> bool:
        self.held.discard(key)
        return True

    def reset(self):
        self.reset_count += 1
        self.held.clear()

    # ---- 断言辅助 ----
    def keys(self) -> List[str]:
        """按顺序返回所有点按过的键。"""
        return [k for k, _ in self.pressed]

    def clear(self):
        self.pressed.clear()
        self.down.clear()
        self.held.clear()
