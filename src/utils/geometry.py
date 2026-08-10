"""几何计算工具。

供决策层判断怪物是否在攻击范围内、计算移动距离等。
"""
import math
from typing import Tuple


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点欧氏距离。"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def in_range(point: Tuple[float, float],
             center: Tuple[float, float],
             radius: float) -> bool:
    """判断 point 是否在以 center 为圆心、radius 为半径的圆内。"""
    return distance(point, center) <= radius


def manhattan(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """曼哈顿距离（A* 启发式可用）。"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
