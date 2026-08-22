"""人物-怪物 路径距离推算器。

================================================================================
推算规则
================================================================================

  输入: 人物脚底点 (px, py)、怪物脚底点 (mx, my)、平台列表、绳索列表

  注意: 两个坐标点都必须是"脚底/站立点"语义，不能用角色中心与怪物
  框中心混比——角色中心 = 脚底 - 角色半高，怪物框中心 = 框几何中点，
  YOLO 怪物框通常偏上，两者中心 Y 差可达 80px+，会把同平台怪物
  误判为跨层（角色只在原地左右跑、从不攻击）。

  1. 同层判定:
     若 |人物脚底y - 怪物脚底y| <= SAME_LEVEL_Y_TOLERANCE(30px) → 同一层
     路径距离 = |mx - px|（只算 x 坐标差）

  2. 不同层（y 差 > 30px）:
     a. 附近有绳索且绳底端人物够得着 → 绳索路径
        距离 = |mx - px|（x 坐标差绝对值）+ 绳索长度（rope.h）
        绳底端（rope.y + rope.h）若远高于人物脚底
        （差 > ROPE_REACH_Y）→ 绳挂在高处，人物够不着，不算可达
     b. 附近无绳索 → 平台跳跃路径
        按平台"最上面的 y 坐标"（top_y = floor.y）逐层跳跃:
          - 每次跳跃高度差 <= JUMP_HEIGHT(80px) 才可达
          - 路径距离 = 累加水平距离（x 差）+ 垂直距离（top_y 差）
     c. 无法通过跳跃到达 → 不可达

注意:
  平台高度一律取平台检测框最上面的 y（floor.y），
  不使用平台中心点 y（floor.center[1]）。

================================================================================
坐标系约定
================================================================================

  Y 轴向下（图像坐标）: top_y 越小 = 平台越高 / 位置越靠上
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..perception.yolo_detector import Detection


# =============================================================================
# 距离推算常量
# =============================================================================

SAME_LEVEL_Y_TOLERANCE = 30
"""同层判定：人物中心与怪物中心的纵坐标差小于等于此值视为同一层（像素）"""

JUMP_HEIGHT = 80
"""人物跳跃高度（像素）：跳跃到不同平台时，平台 top_y 差必须小于等于此值"""

PLATFORM_JUMP_GAP_X = 50
"""平台间跳跃允许的最大水平间隔（像素）：两平台水平间隔超过此值视为跳不过去"""

ROPE_NEAR_X = 150
"""绳索"附近"判定：绳索中心 x 距人物 x 小于此值视为附近（像素）"""

ROPE_REACH_Y = 90
"""绳索可达判定：绳底端（rope.y + rope.h）最多高出人物脚底此值（像素）。
绳底远高于人物（如挂在半空/画面顶部的高绳）→ 人物跳抓不到，不算可达。"""


@dataclass
class PathEstimate:
    """一次路径距离推算的结果。

    Attributes:
        path_type:  路径类型
                    "same_level" 同层直线
                    "rope"       绳索路径
                    "jump"       平台跳跃路径
                    "unreachable" 不可达
        distance:   路径总距离（像素），不可达时为 -1
        horizontal: 水平距离分量（x 差绝对值）
        vertical:   垂直距离分量（y 差绝对值）
        rope_length: 绳索路径时的绳索长度（像素）
        climb_rope: 绳索路径时要攀爬的绳索（Detection）
        jump_count:  平台跳跃路径时的跳跃次数
        path_floors: 平台跳跃路径时"人物起始平台 → 目标平台"的
                     平台序列（BFS 规划结果，供逐层跳跃执行）
        reachable:  是否可达
    """
    path_type: str
    distance: int
    horizontal: int
    vertical: int
    rope_length: int = 0
    climb_rope: Optional[Detection] = None
    jump_count: int = 0
    path_floors: List[Detection] = field(default_factory=list)
    reachable: bool = True


def estimate_path_distance(player_center: Tuple[int, int],
                          monster_center: Tuple[int, int],
                          floors: List[Detection],
                          ropes: List[Detection],
                          same_level_tolerance: int = SAME_LEVEL_Y_TOLERANCE
                          ) -> PathEstimate:
    """推算人物中心点到怪物中心点的可达路径距离。

    Args:
        player_center:  人物脚底/站立点 (px, py)
        monster_center: 怪物脚底点 (mx, my)（bbox 底部，不是框中心）
        floors:         平台（地板）检测结果列表
        ropes:          绳索检测结果列表
        same_level_tolerance: 同层判定的垂直容差(px)。默认 30，
            但应传入 exe 界面配置的 attack_range_y（"垂直容差px"），
            否则 30~容差值 之间的怪物会被误判为跨层，导致既不追击也不攻击。

    Returns:
        PathEstimate 推算结果
    """
    px, py = player_center
    mx, my = monster_center
    dx = abs(mx - px)
    dy = abs(my - py)

    # ---- 1. 同层: 纵坐标差 <= 容差，直接按 x 坐标差 ----
    if dy <= same_level_tolerance:
        return PathEstimate(path_type="same_level", distance=dx,
                            horizontal=dx, vertical=dy)

    # ---- 2. 不同层: 先找附近绳索（有绳索优先爬绳）----
    rope = _find_rope_between(player_center, monster_center, ropes)
    if rope is not None:
        # 路径 = x 坐标差绝对值 + 绳索长度（绳索承担纵向部分）
        rope_len = max(rope.h, dy)
        dist = dx + rope_len
        return PathEstimate(path_type="rope", distance=dist,
                            horizontal=dx, vertical=dy,
                            rope_length=rope_len, climb_rope=rope)

    # ---- 3. 不同层且无绳索: 平台跳跃路径 ----
    plan = _plan_platform_jumps(player_center, monster_center, floors)
    if plan is not None:
        return PathEstimate(path_type="jump", distance=plan["distance"],
                            horizontal=plan["horizontal"],
                            vertical=plan["vertical"],
                            jump_count=plan["jump_count"],
                            path_floors=plan["path_floors"])

    # ---- 4. 不可达 ----
    return PathEstimate(path_type="unreachable", distance=-1,
                        horizontal=dx, vertical=dy, reachable=False)


# =============================================================================
# 绳索查找
# =============================================================================

def _find_rope_between(player_center: Tuple[int, int],
                       monster_center: Tuple[int, int],
                       ropes: List[Detection]) -> Optional[Detection]:
    """查找位于人物与怪物之间（水平范围）且离人物最近的绳索。

    绳索必须水平夹在人物与怪物之间（允许 ROPE_NEAR_X 的放宽），
    否则爬了也到不了怪物那边。
    """
    px, _ = player_center
    mx, _ = monster_center
    x_min = min(px, mx) - ROPE_NEAR_X
    x_max = max(px, mx) + ROPE_NEAR_X

    _, py = player_center
    best = None
    best_dist = float("inf")
    for r in ropes:
        rx = r.center[0]
        if not (x_min <= rx <= x_max):
            continue
        # 垂直可达：不再用"绳底够不着"过滤——YOLO 绳框常只覆盖
        # 绳子上段，绳底远高于人物脚底是检测框不完整，地图绳索
        # 通常垂到地面都能爬。只要水平夹在人与怪之间即视为可用。
        d = abs(rx - px)
        if d < best_dist:
            best_dist = d
            best = r
    return best


# =============================================================================
# 平台跳跃规划
# =============================================================================

def _plan_platform_jumps(player_center: Tuple[int, int],
                         monster_center: Tuple[int, int],
                         floors: List[Detection]):
    """规划从人物所在平台逐层跳跃到怪物所在平台的路径。

    平台高度一律使用平台检测框"最上面的 y"（floor.y），
    不使用平台中心点 y。

    返回 dict 或 None（不可达）:
      {
        "distance":   路径总距离（水平 + 垂直）
        "horizontal": 水平距离分量
        "vertical":   垂直距离分量
        "jump_count": 跳跃次数
      }
    """
    px, py = player_center
    mx, my = monster_center

    start = _find_floor_under(floors, px, py)
    goal = _find_floor_under(floors, mx, my)
    if start is None or goal is None:
        return None
    if start is goal:
        return {"distance": abs(mx - px), "horizontal": abs(mx - px),
                "vertical": 0, "jump_count": 0, "path_floors": [start]}

    # BFS: 平台为节点，高度差 <= JUMP_HEIGHT 且水平间隔 <= PLATFORM_JUMP_GAP_X 可跳
    from collections import deque
    queue = deque()
    queue.append((start, [start]))
    visited = {id(start)}
    best_path = None
    while queue:
        cur, path = queue.popleft()
        if cur is goal:
            best_path = path
            break
        for f in floors:
            if id(f) in visited:
                continue
            if not _can_jump(cur, f):
                continue
            visited.add(id(f))
            queue.append((f, path + [f]))

    if best_path is None:
        return None

    # 累加路径距离: 水平 x 差 + 垂直 top_y 差
    total_h = 0
    total_v = 0
    cur_x = px
    cur_y = py
    for f in best_path[1:]:
        # 跳到平台 f: 水平走到平台中心，垂直从当前高度跳到平台 top_y
        total_h += abs(f.center[0] - cur_x)
        total_v += abs(f.y - cur_y)
        cur_x = f.center[0]
        cur_y = f.y
    # 最后一段: 从最后平台到怪物
    total_h += abs(mx - cur_x)
    total_v += abs(my - cur_y)

    return {"distance": total_h + total_v,
            "horizontal": total_h,
            "vertical": total_v,
            "jump_count": len(best_path) - 1,
            "path_floors": best_path}


def _find_floor_under(floors: List[Detection], x: int, y: int) -> Optional[Detection]:
    """找到覆盖 x 坐标、且 top_y 与 y 最接近的平台（y 站在该平台上）。

    平台范围 [floor.y, floor.y + floor.h]，人物/怪物站立点应落在
    平台顶部附近或平台高度范围内。
    """
    best = None
    best_key = float("inf")
    for f in floors:
        if not (f.x <= x <= f.x + f.w):
            continue
        if f.y - 30 <= y <= f.y + f.h + 30:
            d = abs(f.y - y)
            if d < best_key:
                best_key = d
                best = f
    return best


def _can_jump(a: Detection, b: Detection) -> bool:
    """判断能否从平台 a 跳到平台 b。

    条件:
      1. 垂直: |top_y 差| <= JUMP_HEIGHT（跳跃高度内）
      2. 水平: 平台水平范围重叠，或间隔 <= PLATFORM_JUMP_GAP_X
    """
    if abs(a.y - b.y) > JUMP_HEIGHT:
        return False
    a_lo, a_hi = a.x, a.x + a.w
    b_lo, b_hi = b.x, b.x + b.w
    gap = max(a_lo, b_lo) - min(a_hi, b_hi)  # 负数 = 水平重叠
    return gap <= PLATFORM_JUMP_GAP_X
