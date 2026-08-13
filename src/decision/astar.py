"""A* 网格寻路 — 基于全局地图的路径规划。

================================================================================
设计理念
================================================================================

  不依赖预定义地图 JSON，而是在探索阶段通过 YOLO 逐帧拼接出 GlobalMap，
  然后在 GlobalMap 上运行 A* 搜索最短路径。

  与之前"每帧重建网格"方案的区别:
    之前: 每帧根据当前屏幕的 floors/ropes 构建临时网格 → A*
    现在: 根据已探索的 GlobalMap 直接 A*，路径可以跨屏幕

================================================================================
输入
================================================================================

  global_map:     GlobalMap 实例（已探索的地图）
  start_global:   起点全局坐标 (global_x, global_y) 像素
  goal_global:    终点全局坐标 (global_x, global_y) 像素

================================================================================
输出
================================================================================

  路径点列表（全局像素坐标），空列表表示无路径。
  决策层根据路径点序列生成方向键指令。
"""
import heapq
from typing import List, Tuple

from .global_map import GlobalMap, UNEXPLORED, AIR, FLOOR, LADDER, \
    pixel_to_grid, grid_to_pixel

# 8 方向: 上下左右 + 4 个对角
DIRECTIONS = [
    (-1, 0), (1, 0),    # 左、右
    (0, -1), (0, 1),    # 上、下
    (-1, -1), (1, -1),  # 左上、右上
    (-1, 1), (1, 1),    # 左下、右下
]


class AStarPathfinder:
    """A* 寻路器 — 在 GlobalMap 上搜索最短路径。

    用法:
        pf = AStarPathfinder(global_map)
        path = pf.find_path(start_global_px, goal_global_px)
        if path:
            for pt in path:
                print(f"  → ({pt[0]}, {pt[1]})")
    """

    def __init__(self, global_map: GlobalMap):
        self._map = global_map

    def find_path(self, start_px: Tuple[int, int],
                  goal_px: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* 搜索从 start 到 goal 的最短路径。

        Args:
            start_px: 起点全局像素坐标 (x, y)
            goal_px:  终点全局像素坐标 (x, y)

        Returns:
            路径点列表（全局像素坐标），空列表表示无路径。
        """
        start = (pixel_to_grid(start_px[0]), pixel_to_grid(start_px[1]))
        goal = (pixel_to_grid(goal_px[0]), pixel_to_grid(goal_px[1]))

        gm = self._map

        if not gm.is_walkable(start[0], start[1]):
            return []
        if not gm.is_walkable(goal[0], goal[1]):
            return []

        # A* 数据结构
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            cx, cy = current

            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                if neighbor in closed:
                    continue
                if not gm.is_walkable(nx, ny):
                    continue

                # 对角移动检查: 不能穿过障碍物
                if dx != 0 and dy != 0:
                    if not gm.is_walkable(cx + dx, cy) and \
                       not gm.is_walkable(cx, cy + dy):
                        continue

                # 移动代价: 直线 1.0，对角 1.414
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0

                # 攀爬代价稍高
                if gm.is_ladder(cx, cy) or gm.is_ladder(nx, ny):
                    move_cost *= 1.5

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    h = abs(nx - goal[0]) + abs(ny - goal[1])
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current

        return []

    def _reconstruct_path(self, came_from: dict,
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """回溯 A* 路径，返回全局像素坐标列表。"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [(grid_to_pixel(gx), grid_to_pixel(gy)) for gx, gy in path]


# =============================================================================
# 便捷函数: 从路径生成方向键指令
# =============================================================================

def path_to_directions(path: List[Tuple[int, int]],
                       current_pos: Tuple[int, int],
                       lookahead: int = 3) -> List[str]:
    """将 A* 路径转换为方向键指令序列。

    取路径的前 lookahead 个点，根据当前位置判断需要按哪些键。

    Args:
        path:        A* 返回的路径点列表（全局像素坐标）
        current_pos: 当前自身全局像素坐标
        lookahead:   向前看几个路径点（用于判断是否需要跳跃）

    Returns:
        方向键指令列表，如 ["left", "jump"], ["right"], ["up"] 等
    """
    if not path:
        return []

    commands = []
    cx, cy = current_pos

    next_idx = min(lookahead, len(path) - 1)
    tx, ty = path[next_idx]

    dx = tx - cx
    dy = cy - ty  # y 轴向下

    if abs(dx) > 5:
        commands.append("right" if dx > 0 else "left")

    if dy > 20:
        commands.append("jump")
    elif dy < -10:
        commands.append("down")

    return commands