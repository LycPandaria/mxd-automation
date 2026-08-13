"""有限状态机（FSM）占位。

================================================================================
为什么需要状态机
================================================================================

  当前 DecisionEngine.decide() 使用简单的 if-else 链做决策，
  这在逻辑简单时够用，但复杂场景下（如"正在移动中突然没血了"）
  if-else 链会变得很乱。

  状态机（FSM）把决策逻辑拆分为多个状态，每个状态有明确的:
    - 进入条件: 什么情况下进入这个状态
    - 退出条件: 什么情况下离开这个状态
    - 行为:     在这个状态下每帧做什么

================================================================================
状态定义
================================================================================

  Idle       - 空闲，等待目标
  Attacking  - 攻击中（轮转技能）
  Moving     - 移动到怪位置
  Climbing   - 攀爬/跳跃跨平台（配合 A* 寻路）
  Healing    - 加血
  Recovering - 加蓝

================================================================================
迁移路径
================================================================================

  1. 在 DecisionEngine 中暴露状态字段，方便 UI 观察
  2. 将 decide() 改为按状态查表执行
  3. 引入状态转移表替代 if-else 链

  例如:
    状态转移表:
      Idle      → 有怪 → Attacking
      Idle      → 没血 → Healing
      Attacking → 没怪 → Idle
      Attacking → 没血 → Healing
      Healing   → 血满 → 回到之前的状态
"""
from enum import Enum


class State(Enum):
    """状态枚举。

    每个状态值是一个字符串，方便日志输出和调试。
    """
    Idle = "idle"           # 空闲：没有目标，等待中
    Attacking = "attacking" # 攻击中：轮转释放技能
    Moving = "moving"       # 移动中：走向怪物位置
    Climbing = "climbing"   # 攀爬中：跨平台（绳索/梯子）
    Healing = "healing"     # 加血中：HP 低于阈值
    Recovering = "recovering"  # 加蓝中：MP 低于阈值


class FSM:
    """有限状态机（占位类）。

    当前只是简单的状态容器，后续需要实现:
      - 状态转移表: 定义哪些状态之间可以转移
      - 进入钩子: on_enter(state) 进入状态时执行一次
      - 退出钩子: on_exit(state) 离开状态时执行一次
      - 状态超时: 某个状态持续太久自动跳出

    Args:
        initial: 初始状态，默认 Idle
    """

    def __init__(self, initial: State = State.Idle):
        self.current = initial  # 当前状态

    def transition(self, next_state: State):
        """状态转移（占位方法）。

        TODO:
          - 校验转移合法性（不能从 Attacking 直接跳到 Climbing）
          - 触发 on_exit(current) 和 on_enter(next_state)
          - 记录转移日志

        Args:
            next_state: 目标状态
        """
        # TODO: 校验转移合法性，触发进入/退出钩子
        self.current = next_state