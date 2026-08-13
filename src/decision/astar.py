"""A* 网格寻路 — 基于实时视觉感知的动态路径规划。

================================================================================
设计理念
================================================================================

  不依赖预定义地图 JSON，而是每帧根据 YOLO 检测结果动态构建网格，
  在网格上运行 A* 搜索最短路径。

================================================================================
输入（来自感知层）
================================================================================

  floors:         YOLO 检测到的地板列表 [Detection, ...]
  ropes:          YOLO 检测到的绳索列表 [Detection, ...]
  self_position:  自身脚底坐标 (cx, cy)
  target_position: 目标坐标（怪物脚底）(cx, cy)
  frame_width:    画面宽度（像素）
  frame_height:   画面高度（像素）

================================================================================
网格构建
================================================================================

  画面被划分为 CELL_SIZE × CELL_SIZE 的网格（默认 10px）。
  每个格子有三种状态:
    AIR (0):      不可行走（空中）
    FLOOR (1):    可站立/行走（地板区域及上方 2 格）
    LADDER (2):   可攀爬（绳索区域）

  构建规则:
    1. 所有格子初始为 AIR
    2. 地板检测框内的格子 → FLOOR
    3. 地板检测框上方 2 格 → FLOOR（角色可以站在地板上）
    4. 绳索检测框内的格子 → LADDER（可上下攀爬）

================================================================================
移动规则（2D 横版平台跳跃）
================================================================================

  从格子 (cx, cy) 可以移动到:

    1. 左右移动:  (cx±1, cy)   目标格子必须是 FLOOR
    2. 攀爬上下:  (cx, cy±1)   当前格和目标格都是 LADDER，或当前格是 FLOOR 且目标格是 LADDER
    3. 跳跃:      (cx±dx, cy-dy)  dx≤JUMP_WIDTH, dy≤JUMP_HEIGHT
        - 起点必须是 FLOOR（站在地上才能跳）
        - 终点必须是 FLOOR
        - 路径中间不能穿过 FLOOR 格子（不能穿过平台）
    4. 下落:      (cx±dx, cy+dy)  dx≤1, 任意 dy（重力下落）
        - 终点必须是 FLOOR
        - 路径中间不能穿过 FLOOR 格子

================================================================================
输出
================================================================================

  find_path() 返回路径点列表 [(x, y), ...]，每个点的间隔可能 >1 格。
  决策层根据路径点序列生成方向键指令（← → ↑ ↓ + 跳跃键）。

  如果 A* 找不到路径，返回空列表，决策层回退到"按 Tab 选同平台怪"。
"""
import heapq
import math
from typing import List, Optional, Tuple, Set

from ..perception.yolo_detector import Detection

# ---- 网格定义 ----
# 每个格子的像素大小。10px 在 1366×768 下产生 137×77 ≈ 1 万格，A* 很快。
CELL_SIZE = 10

# 格子类型
AIR = 0     # 不可行走
FLOOR = 1   # 可站立/行走
LADDER = 2  # 可攀爬

# 跳跃参数
JUMP_WIDTH = 8   # 水平跳跃距离（格数），8 格 × 10px = 80px
JUMP_HEIGHT = 6  # 垂直跳跃高度（格数），6 格 × 10px = 60px

# 角色站在地板上方的高度（格数）
FLOOR_STAND_HEIGHT = 2  # 地板检测框上方 2 格 = 20px 留给角色站立

# 方向向量（用于 A* 展开邻居）
# 8 方向: 上下左右 + 4 个对角
DIRECTIONS = [
    (-1, 0), (1, 0),   # 左、右
    (0, -1), (0, 1),   # 上、下
    (-1, -1), (1, -1), # 左上、右上
    (-1, 1), (1, 1),   # 左下、右下
]


def _grid_pos(px: int, py: int) -> Tuple[int, int]:
    """像素坐标 → 网格坐标。

    Args:
        px: 像素 x 坐标
        py: 像素 y 坐标

    Returns:
        (grid_x, grid_y) 网格坐标
    """
    return (px // CELL_SIZE, py // CELL_SIZE)


def _pixel_pos(gx: int, gy: int) -> Tuple[int, int]:
    """网格坐标 → 像素坐标（格子中心）。

    Args:
        gx: 网格 x 坐标
        gy: 网格 y 坐标

    Returns:
        (pixel_x, pixel_y) 像素坐标
    """
    return (gx * CELL_SIZE + CELL_SIZE // 2, gy * CELL_SIZE + CELL_SIZE // 2)


class GridMap:
    """基于 YOLO 检测结果动态构建的网格地图。

    每帧重建（或 YOLO 检测结果变化时重建），不是持久化的。

    Attributes:
        grid:    二维数组 grid[y][x]，值为 AIR/FLOOR/LADDER
        width:   网格宽度（列数）
        height:  网格高度（行数）
    """

    def __init__(self):
        self.grid = []
        self.width = 0
        self.height = 0

    def build(self, floors: List[Detection], ropes: List[Detection],
              frame_width: int, frame_height: int):
        """根据 YOLO 检测结果构建网格。

        Args:
            floors:       地板检测结果列表
            ropes:        绳索检测结果列表
            frame_width:  画面宽度
            frame_height: 画面高度
        """
        self.width = frame_width // CELL_SIZE
        self.height = frame_height // CELL_SIZE

        # 初始化所有格子为 AIR（不可行走）
        self.grid = [[AIR for _ in range(self.width)] for _ in range(self.height)]

        # 标记地板: 地板框内的格子 + 上方站立区域
        for d in floors:
            x1, y1 = _grid_pos(d.x, d.y)
            x2, y2 = _grid_pos(d.x + d.w, d.y + d.h)

            # 限制在网格范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width - 1, x2)
            y2 = min(self.height - 1, y2)

            for gy in range(y1, y2 + 1):
                for gx in range(x1, x2 + 1):
                    self.grid[gy][gx] = FLOOR

            # 地板检测框上方 FLOOR_STAND_HEIGHT 格也标记为 FLOOR
            # 这样角色可以"站在"地板上
            stand_y1 = max(0, y1 - FLOOR_STAND_HEIGHT)
            for gy in range(stand_y1, y1):
                for gx in range(x1, x2 + 1):
                    if self.grid[gy][gx] == AIR:
                        self.grid[gy][gx] = FLOOR

        # 标记绳索: 绳索框内的格子标记为 LADDER
        for d in ropes:
            x1, y1 = _grid_pos(d.x, d.y)
            x2, y2 = _grid_pos(d.x + d.w, d.y + d.h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width - 1, x2)
            y2 = min(self.height - 1, y2)

            for gy in range(y1, y2 + 1):
                for gx in range(x1, x2 + 1):
                    # 绳索区域覆盖为 LADDER（即使原本是 FLOOR 也覆盖）
                    self.grid[gy][gx] = LADDER

    def is_walkable(self, gx: int, gy: int) -> bool:
        """判断格子是否可行走。

        FLOOR 和 LADDER 都算可行走。
        FLOOR 格可以左右走，LADDER 格可以上下爬。

        Args:
            gx: 网格 x 坐标
            gy: 网格 y 坐标

        Returns:
            True 表示可以站在这个格子上
        """
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy][gx] != AIR

    def is_ladder(self, gx: int, gy: int) -> bool:
        """判断格子是否是绳索（可攀爬）。"""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy][gx] == LADDER

    def is_floor(self, gx: int, gy: int) -> bool:
        """判断格子是否是地板（可站立）。"""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy][gx] == FLOOR

    def is_standable(self, gx: int, gy: int) -> bool:
        """判断格子是否可站立（FLOOR 或 LADDER 顶部）。

        FLOOR 格可以直接站，LADDER 格也可以站（绳索顶端）。
        """
        if not self.is_walkable(gx, gy):
            return False
        # 下方格子也必须是可行走的（不能悬空）
        below = gy + 1
        if below >= self.height:
            return False  # 画面底部不算可站立（除非是地面）
        return self.is_walkable(gx, below)


class AStarPathfinder:
    """A* 网格寻路器。

    每帧根据 YOLO 检测结果构建网格，然后搜索从自身到目标的路径。

    用法:
        pf = AStarPathfinder()
        pf.build_grid(floors, ropes, frame_width, frame_height)
        path = pf.find_path(self_pos, target_pos)
        if path:
            next_step = path[0]  # 下一步走到的像素坐标
            # 决策层根据 next_step 和 self_pos 生成方向键指令
    """

    def __init__(self):
        self.grid_map = GridMap()

    def build_grid(self, floors: List[Detection], ropes: List[Detection],
                   frame_width: int, frame_height: int):
        """根据 YOLO 检测结果重建网格。

        每帧调用一次（YOLO 检测结果变化时才需要重建）。
        """
        self.grid_map.build(floors, ropes, frame_width, frame_height)

    def find_path(self, start_px: Tuple[int, int],
                  goal_px: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* 搜索从 start 到 goal 的最短路径。

        算法: 标准 A*，使用曼哈顿距离作为启发式，8 方向移动。

        Args:
            start_px: 起点像素坐标 (x, y)
            goal_px:  终点像素坐标 (x, y)

        Returns:
            路径点列表（像素坐标），从起点到终点，空列表表示无路径。
            第一个点是起点，最后一个点是终点。
        """
        start = _grid_pos(start_px[0], start_px[1])
        goal = _grid_pos(goal_px[0], goal_px[1])

        gm = self.grid_map

        # 起点或终点不可行走 → 无路径
        if not gm.is_walkable(start[0], start[1]):
            return []
        if not gm.is_walkable(goal[0], goal[1]):
            return []

        # A* 数据结构
        # open_set: 优先队列 (f_score, g_score, node)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        # came_from: node → 前驱节点（用于回溯路径）
        came_from = {}

        # g_score: 从起点到 node 的实际代价
        g_score = {start: 0}

        # 已访问集合（closed_set）
        closed = set()

        # A* 主循环
        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # 找到路径，回溯
                return self._reconstruct_path(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            cx, cy = current

            # 展开邻居（8 方向）
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                if neighbor in closed:
                    continue

                if not gm.is_walkable(nx, ny):
                    continue

                # 对角移动检查: 不能穿过障碍物
                if dx != 0 and dy != 0:
                    # 对角移动需要两个相邻方向都可行走
                    if not gm.is_walkable(cx + dx, cy) and not gm.is_walkable(cx, cy + dy):
                        continue

                # 移动代价: 直线 1.0，对角 1.414
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0

                # 攀爬代价: 在 LADDER 上移动代价稍高
                if gm.is_ladder(cx, cy) or gm.is_ladder(nx, ny):
                    move_cost *= 1.5

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    # 启发式: 曼哈顿距离
                    h = abs(nx - goal[0]) + abs(ny - goal[1])
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current

        # open_set 为空，无路径
        return []

    def _reconstruct_path(self, came_from: dict,
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """从 A* 搜索结果回溯路径。

        Args:
            came_from: 前驱节点字典
            current:   终点网格坐标

        Returns:
            路径点列表（像素坐标），从起点到终点
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # 转换为像素坐标
        return [_pixel_pos(gx, gy) for gx, gy in path]


# =============================================================================
# 便捷函数: 从路径生成方向键指令
# =============================================================================

def path_to_directions(path: List[Tuple[int, int]],
                       current_pos: Tuple[int, int],
                       lookahead: int = 3) -> List[str]:
    """将 A* 路径转换为方向键指令序列。

    取路径的前 lookahead 个点，根据当前位置判断需要按哪些键。

    Args:
        path:        A* 返回的路径点列表（像素坐标）
        current_pos: 当前自身像素坐标
        lookahead:   向前看几个路径点（用于判断是否需要跳跃）

    Returns:
        方向键指令列表，如 ["left", "jump"], ["right"], ["up"] 等
    """
    if not path:
        return []

    commands = []
    cx, cy = current_pos

    # 取路径上的下一个点
    next_idx = min(lookahead, len(path) - 1)
    tx, ty = path[next_idx]

    dx = tx - cx
    dy = cy - ty  # 注意: y 轴向下，所以 dy = cy - ty

    # 水平方向
    if abs(dx) > 5:
        commands.append("right" if dx > 0 else "left")

    # 垂直方向: 如果目标在上方且距离较远，需要跳跃
    if dy > 20:
        commands.append("jump")
    elif dy < -10:
        # 目标在下方，可能需要按↓从平台下落
        commands.append("down")

    return commands