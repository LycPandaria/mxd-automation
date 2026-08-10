"""决策层：只负责"想"，输出动作指令。

模块：
  - context:   共享上下文（汇总感知层数据，如当前血量、怪物列表）
  - fsm:       有限状态机（占位，未来替换 DecisionEngine）
  - astar:     A* 拓扑寻路（占位，读取 JSON 计算跨平台跳跃路径）
"""
from .context import Context, DecisionEngine  # noqa: F401
