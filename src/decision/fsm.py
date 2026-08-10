"""有限状态机（FSM）占位。

未来用于替换 ``DecisionEngine`` 中的简单 if-else 决策。状态枚举：

  Idle       - 空闲，等待目标
  Attacking  - 攻击中（轮转技能）
  Moving     - 移动到怪位置
  Climbing   - 攀爬/跳跃跨平台（配合 A* 寻路）
  Healing    - 加血
  Recovering - 加蓝

迁移路径：
  1. 在 ``DecisionEngine`` 中暴露状态字段，方便观察
  2. 将 ``decide()`` 改为按状态查表执行
  3. 引入状态转移表替代 if-else 链
"""
from enum import Enum


class State(Enum):
    Idle = "idle"
    Attacking = "attacking"
    Moving = "moving"
    Climbing = "climbing"
    Healing = "healing"
    Recovering = "recovering"


class FSM:
    """有限状态机（占位）。

    TODO:
        - 状态转移表
        - 状态进入/退出钩子
        - 状态超时
    """

    def __init__(self, initial: State = State.Idle):
        self.current = initial

    def transition(self, next_state: State):
        """状态转移（占位）。"""
        # TODO: 校验转移合法性，触发进入/退出钩子
        self.current = next_state
