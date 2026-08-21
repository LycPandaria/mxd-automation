"""有限状态机 — 反应式决策的状态管理。

================================================================================
为什么需要状态机
================================================================================

  反应式决策引擎根据当前画面做判断，但"判断"本身有上下文——
  比如"正在追怪"和"正在爬绳"时，对同一画面的反应完全不同。

  状态机把决策逻辑拆分为多个状态，每个状态有明确的:
    - 进入条件: 什么情况下进入这个状态
    - 退出条件: 什么情况下离开这个状态
    - 行为:     在这个状态下每帧做什么

================================================================================
状态定义
================================================================================

  IDLE       - 空闲：画面里没怪，往一个方向走探索
  CHASING    - 追击：同平台有怪，走过去
  ATTACKING  - 攻击：怪在攻击范围内，轮转放技能
  CLIMBING   - 攀爬：怪在上方，找绳索爬上去
  DROPPING   - 下落：怪在下方，找平台边缘跳下去
  HEALING    - 加血：HP 低于阈值
  RECOVERING - 加蓝：MP 低于阈值

================================================================================
状态转移图
================================================================================

                    ┌─────────┐
           ┌──────▶│  IDLE   │◀──────┐
           │       └────┬────┘       │
           │            │ 有怪       │ 怪消失
           │       ┌────▼────┐       │
           │ 同平台│ CHASING │       │
           │       └────┬────┘       │
           │        进入范围         │
           │       ┌────▼────┐       │
           │       │ATTACKING├───────┘
           │       └─────────┘
           │
           │       ┌─────────┐
           │ 怪在上│ CLIMBING│──▶ 到达 → CHASING
           │       └─────────┘
           │
           │       ┌─────────┐
           │ 怪在下│ DROPPING│──▶ 到达 → CHASING
           │       └─────────┘

  任何状态 ──HP低──▶ HEALING ──血满──▶ 回到之前状态
  任何状态 ──MP低──▶ RECOVERING ──蓝满──▶ 回到之前状态
"""
from enum import Enum


class State(Enum):
    """状态枚举。"""
    IDLE = "idle"
    CHASING = "chasing"
    ATTACKING = "attacking"
    CLIMBING = "climbing"
    DROPPING = "dropping"
    HEALING = "healing"
    RECOVERING = "recovering"


class FSM:
    """有限状态机。

    管理状态切换，记录状态持续时间，提供进入/退出钩子。

    Args:
        initial: 初始状态，默认 IDLE
        on_log:  日志回调
    """

    def __init__(self, initial: State = State.IDLE,
                 on_log=None):
        self.current = initial
        self.previous = initial
        self._log = on_log or (lambda m: None)
        self._frame_count = 0
        self._state_entry_frame = 0

    def tick(self):
        """每帧调用，递增帧计数。"""
        self._frame_count += 1

    def transition(self, next_state: State):
        """状态转移。

        记录前一个状态，触发进入/退出日志。

        Args:
            next_state: 目标状态
        """
        if next_state == self.current:
            return
        self.previous = self.current
        old = self.current
        self.current = next_state
        self._state_entry_frame = self._frame_count
        self._log(f"[状态] {old.value} → {next_state.value}")

    @property
    def frames_in_state(self) -> int:
        """当前状态持续的帧数。"""
        return self._frame_count - self._state_entry_frame

    @property
    def state_name(self) -> str:
        """当前状态名（供 UI 显示）。"""
        return self.current.value

    def reset(self):
        """重置到初始状态。"""
        self.current = State.IDLE
        self.previous = State.IDLE
        self._frame_count = 0
        self._state_entry_frame = 0