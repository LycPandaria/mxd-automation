"""决策上下文与反应式决策引擎。

================================================================================
设计理念
================================================================================

  不再建地图、不做 A* 寻路。每一帧只看 YOLO 检测到的画面内容，
  像人类玩家一样"看到什么就做什么反应"。

  画面里有什么 → 就应该做什么:
    - 看到怪 → 判断同平台还是跨平台，走过去或爬绳跳下去
    - 看到地板 → 知道哪里能站
    - 看到绳索 → 知道哪里能爬
    - 没看到怪 → 往一个方向走探索
    - HP/MP 低 → 加血加蓝

================================================================================
架构
================================================================================

  ┌─────────────┐    ┌─────────────────────┐    ┌──────────────┐
  │  感知层      │ →  │  DecisionEngine     │ →  │ ActionExec   │
  │ YOLO/OCR/HP │    │  (反应式决策 + FSM)  │    │ (方向键/技能) │
  └─────────────┘    └─────────────────────┘    └──────────────┘

================================================================================
决策流程（优先级从高到低）
================================================================================

  1. HP 低于阈值 → 加血键
  2. MP 低于阈值 → 加蓝键
  3. 检测到怪物:
     a. 同平台 → 走过去 + 进入范围后攻击
     b. 怪在上方 + 有绳索 → 爬绳
     c. 怪在下方 → 找边缘跳下
     d. 都不满足 → Tab 选怪 + 原地攻击
  4. 没怪 → 探索（往一个方向走，遇坑跳）

================================================================================
Context 字段说明
================================================================================

  monsters:       YOLO 检测到的怪物列表
  floors:         地板列表
  ropes:          绳索列表
  self_position:  自身脚底坐标 (cx, cy) 或 None
  self_center:    自身角色中心点坐标 (cx, cy) 或 None（OCR 定位时记录）
  hp_ratio:       血量比例 0.0~1.0
  mp_ratio:       蓝量比例 0.0~1.0
  detections:     全部 YOLO 检测结果（含所有类别）
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

from ..perception.yolo_detector import Detection
from ..execution.action_executor import ActionExecutor
from ..utils.config_loader import Config
from .fsm import FSM, State
from .distance import (
    estimate_path_distance,
    JUMP_HEIGHT,
)


# =============================================================================
# 反应式决策的阈值常量
# =============================================================================

ATTACK_RANGE_X = 250
"""攻击范围：自身与怪物 X 坐标差小于此值视为进入攻击范围（像素）"""

ROPE_SEARCH_RANGE_X = 200
"""搜索绳索的水平范围（像素）"""

STUCK_FRAMES = 60
"""卡住判定帧数：持续此帧数位置不变则视为卡住"""

EXPLORE_DIRECTION_SWITCH_FRAMES = 180
"""探索方向切换帧数：探索状态下持续此帧数没遇到怪就换方向"""

DISTANCE_LOG_FRAMES = 60
"""距离推算日志输出间隔帧数（避免刷屏）"""


@dataclass
class Context:
    """感知层 → 决策层的数据载体（每帧一份）。

    self_position 由 OCR 识别得到（窗口内坐标，脚底）。
    self_center 为角色中心点（名字中心 + 人物高度一半），用于距离推算。
    """
    monsters: List[Detection] = field(default_factory=list)
    floors: List[Detection] = field(default_factory=list)
    ropes: List[Detection] = field(default_factory=list)
    self_position: Optional[Tuple[int, int]] = None
    self_center: Optional[Tuple[int, int]] = None
    hp_ratio: Optional[float] = None
    mp_ratio: Optional[float] = None
    detections: List[Detection] = field(default_factory=list)


class DecisionEngine:
    """反应式决策引擎：根据画面实时内容决定下一步动作。

    【核心理念】
    不做地图、不做全局规划。每一帧只看 YOLO 检测结果，
    模拟人类玩家的反应模式。

    【状态机】
    使用 FSM 管理 7 个状态：
      IDLE → CHASING → ATTACKING  （同平台追击）
      IDLE → CLIMBING              （爬绳追怪）
      IDLE → DROPPING              （跳下追怪）
      任意 → HEALING / RECOVERING  （生存优先）

    Args:
        config:   全局配置
        executor: 动作执行器
        on_log:   日志回调
    """

    def __init__(self, config: Config, executor: ActionExecutor,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.executor = executor
        self._log = on_log or (lambda m: None)
        self._skill_index = 0

        self._fsm = FSM(on_log=self._log)

        self._target_monster: Optional[Detection] = None
        self._explore_direction = "right"
        self._last_self_pos: Optional[Tuple[int, int]] = None
        self._stuck_counter = 0
        self._explore_frame_count = 0
        self._distance_log_frame_count = 0

    def update_config(self, config: Config):
        self.config = config

    def reset(self):
        self._skill_index = 0
        self._target_monster = None
        self._explore_direction = "right"
        self._last_self_pos = None
        self._stuck_counter = 0
        self._explore_frame_count = 0
        self._distance_log_frame_count = 0
        self._fsm.reset()
        self.executor.reset()

    @property
    def state_name(self) -> str:
        """当前状态名（供 UI 显示）。"""
        return self._fsm.state_name

    # =========================================================================
    # 决策主入口
    # =========================================================================

    def decide(self, ctx: Context):
        """每帧调用一次，根据画面内容执行动作。

        Args:
            ctx: 当前帧的感知数据
        """
        self._fsm.tick()

        # 检测卡住
        if ctx.self_position:
            if self._last_self_pos and self._last_self_pos == ctx.self_position:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
            self._last_self_pos = ctx.self_position

        # ---- 优先级 1: 没血加血 ----
        if ctx.hp_ratio is not None and ctx.hp_ratio < self.config.hp_threshold:
            # 满状态（>=95%）不触发，防止刚加完又按
            if ctx.hp_ratio < 0.95:
                self._fsm.transition(State.HEALING)
                if self.executor.press_key(self.config.hp_key, cooldown=1.5):
                    self._log(
                        f"[加血] HP={ctx.hp_ratio:.0%} < {self.config.hp_threshold:.0%}，"
                        f"按下 {self.config.hp_key}"
                    )
                    return

        # ---- 优先级 2: 没蓝加蓝 ----
        if ctx.mp_ratio is not None and ctx.mp_ratio < self.config.mp_threshold:
            if ctx.mp_ratio < 0.95:
                self._fsm.transition(State.RECOVERING)
                if self.executor.press_key(self.config.mp_key, cooldown=1.5):
                    self._log(
                        f"[加蓝] MP={ctx.mp_ratio:.0%} < {self.config.mp_threshold:.0%}，"
                        f"按下 {self.config.mp_key}"
                    )
                    return

        # ---- 优先级 3: 检测到怪物 ----
        if ctx.monsters:
            self._handle_monsters(ctx)
        else:
            self._fsm.transition(State.IDLE)
            self._explore(ctx)

    # =========================================================================
    # 怪物处理
    # =========================================================================

    def _handle_monsters(self, ctx: Context):
        """处理画面中的怪物。"""
        target = self._pick_best_target(ctx)
        self._target_monster = target

        tx, ty = target.center

        # 人物中心点（优先用角色中心，回退脚底）
        player = ctx.self_center or ctx.self_position
        if player is None:
            self._tab_attack(ctx)
            return

        sx, sy = player

        # ---- 距离推算 ----
        est = estimate_path_distance(player, (tx, ty), ctx.floors, ctx.ropes)

        # 距离日志（限频输出，避免刷屏）
        self._distance_log_frame_count += 1
        if self._distance_log_frame_count >= DISTANCE_LOG_FRAMES:
            self._distance_log_frame_count = 0
            self._log(
                f"[距离] 人物({sx},{sy}) → 怪({tx},{ty}) "
                f"路径={est.path_type} 距离={est.distance}px "
                f"(水平={est.horizontal} 垂直={est.vertical}"
                + (f" 绳长={est.rope_length}" if est.path_type == "rope" else "")
                + (f" 跳数={est.jump_count}" if est.path_type == "jump" else "")
                + ")"
            )

        # 同平台（纵坐标差 <= 30px）→ 追击或攻击
        if est.path_type == "same_level":
            if self._in_attack_range(sx, tx):
                self._fsm.transition(State.ATTACKING)
                self._attack(ctx, target)
            else:
                self._fsm.transition(State.CHASING)
                self._chase(ctx, target)
            return

        # 不同层 + 有绳索 → 爬绳
        if est.path_type == "rope":
            self._fsm.transition(State.CLIMBING)
            if not self._try_climb(ctx, target):
                self._tab_attack(ctx)
            return

        # 不同层 + 无绳索 → 平台跳跃追击
        if est.path_type == "jump":
            self._fsm.transition(State.CHASING)
            self._jump_chase(ctx, target)
            return

        # 兜底：Tab 选怪 + 原地攻击
        self._tab_attack(ctx)

    # =========================================================================
    # 目标选择
    # =========================================================================

    def _pick_best_target(self, ctx: Context) -> Detection:
        """选择最佳目标怪物。

        按路径距离（同层/绳索/跳跃）选择距离最近的怪物；
        无法定位自身时回退为画面中最大的怪物。
        """
        monsters = ctx.monsters
        if not monsters:
            return None
        player = ctx.self_center or ctx.self_position
        if player is None:
            return max(monsters, key=lambda d: d.w * d.h)

        best = None
        best_dist = float("inf")
        for m in monsters:
            est = estimate_path_distance(player, m.center, ctx.floors, ctx.ropes)
            if est.reachable and est.distance < best_dist:
                best = m
                best_dist = est.distance
        return best if best is not None else max(monsters, key=lambda d: d.w * d.h)

    # =========================================================================
    # 地板判定
    # =========================================================================

    def _has_floor_under(self, ctx: Context, pos: Tuple[int, int]) -> bool:
        """检查指定位置下方是否有地板。

        判断: 地板检测框的 Y 范围是否覆盖了该位置的 Y 坐标附近。
        """
        px, py = pos
        for f in ctx.floors:
            if f.x <= px <= f.x + f.w:
                if f.y - 10 <= py <= f.y + f.h + 10:
                    return True
        return True  # 没检测到地板时默认认为可以站（宽容处理）

    # =========================================================================
    # 攻击范围判定
    # =========================================================================

    def _in_attack_range(self, sx: int, tx: int) -> bool:
        """判断自身是否在攻击范围内。"""
        return abs(sx - tx) < ATTACK_RANGE_X

    # =========================================================================
    # 追击（同平台）
    # =========================================================================

    def _chase(self, ctx: Context, target: Detection):
        """走向怪物。"""
        if ctx.self_position is None:
            return
        sx = ctx.self_position[0]
        tx = target.center[0]

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[追击] 卡住了，尝试跳跃")
            self.executor.press_key("alt", cooldown=1.0)
            self._stuck_counter = 0
            return

        if tx > sx + 10:
            self.executor.press_key("right", cooldown=0.05)
        elif tx < sx - 10:
            self.executor.press_key("left", cooldown=0.05)

    # =========================================================================
    # 攻击
    # =========================================================================

    def _attack(self, ctx: Context, target: Detection):
        """在攻击范围内，面向怪物并释放技能。

        先调整面向（确保攻击方向正确），再轮转放技能。
        """
        if ctx.self_position is None:
            self._cast_skill()
            return
        sx = ctx.self_position[0]
        tx = target.center[0]

        # 调整面向
        if tx > sx + 5:
            self.executor.press_key("right", cooldown=0.05)
        elif tx < sx - 5:
            self.executor.press_key("left", cooldown=0.05)

        # 轮转释放技能
        self._cast_skill()

    def _tab_attack(self, ctx: Context):
        """Tab 选怪 + 原地攻击（兜底方案）。"""
        self.executor.press_key(self.config.target_key, cooldown=0.8)
        self._cast_skill()

    # =========================================================================
    # 攀爬
    # =========================================================================

    def _try_climb(self, ctx: Context, target: Detection) -> bool:
        """尝试找绳索爬上去追怪。

        返回 True 表示找到了绳索并执行了动作，False 表示没找到。
        """
        if ctx.self_position is None:
            return False
        sx = ctx.self_position[0]

        # 找最近的绳索
        nearest_rope = None
        min_dist = float("inf")
        for r in ctx.ropes:
            rx = r.center[0]
            dist = abs(rx - sx)
            if dist < ROPE_SEARCH_RANGE_X and dist < min_dist:
                nearest_rope = r
                min_dist = dist

        if nearest_rope is None:
            return False

        rx = nearest_rope.center[0]
        ry = nearest_rope.center[1]

        # 如果已经接近绳索
        if abs(rx - sx) < 30:
            # 攀爬：一直按上键（直到接近目标高度）
            target_top = target.y
            if ctx.self_position[1] > target_top + 50:
                self.executor.press_key("up", cooldown=0.05)
                self._log("[攀爬] 沿绳索向上")
            else:
                self._log("[攀爬] 已到达目标高度")
                self.executor.press_key("right" if target.center[0] > sx else "left",
                                        cooldown=0.1)
            return True
        else:
            # 走向绳索
            if rx > sx:
                self.executor.press_key("right", cooldown=0.05)
            else:
                self.executor.press_key("left", cooldown=0.05)
            return True

    # =========================================================================
    # 跨层跳跃追击（无绳索时）
    # =========================================================================

    def _jump_chase(self, ctx: Context, target: Detection):
        """无绳索时，通过跳跃不同平台追击怪物。

        平台高度按"最上面的 y 坐标"（floor.y，top_y）计算，不是平台中心 y:
          - 怪物所在平台 top_y 与人物高度差 <= 跳跃高度(80px) → 起跳
          - 脚下没地板（在平台边缘/空中）→ 下落
          - 否则继续往怪物方向走
        """
        if ctx.self_position is None:
            return
        sx, sy = ctx.self_position
        tx, ty = target.center[0], target.center[1]

        # 怪物所在平台的 top_y（最上面的 y）
        target_top = self._find_floor_top(ctx, tx, ty)

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[跳跃追击] 卡住了，起跳")
            self.executor.press_key("alt", cooldown=1.0)
            self._stuck_counter = 0
            return

        # 水平方向走向怪物
        if tx > sx + 10:
            self.executor.press_key("right", cooldown=0.05)
        elif tx < sx - 10:
            self.executor.press_key("left", cooldown=0.05)

        # 脚下有地板
        on_floor = self._has_floor_under(ctx, ctx.self_position)

        if on_floor and target_top is not None:
            # 怪在上层：高度差在跳跃高度内就起跳
            if ty < sy and (sy - target_top) <= JUMP_HEIGHT:
                self._log(
                    f"[跳跃追击] 目标平台 top_y={target_top}，"
                    f"高度差 {sy - target_top}px <= {JUMP_HEIGHT}px，起跳"
                )
                self.executor.press_key("alt", cooldown=1.0)
                return
            # 高度差超出跳跃高度 → 走不过去，只能继续往怪物方向走
            self._log(
                f"[跳跃追击] 目标平台 top_y={target_top}，"
                f"高度差 {sy - target_top}px > {JUMP_HEIGHT}px，无法直接跳上，继续走"
            )
            return

        # 脚下没地板（在平台边缘/空中）→ 下落
        if not on_floor:
            self.executor.press_key("down", cooldown=0.3)
            self.executor.press_key("alt", cooldown=1.0)
            self._log("[跳跃追击] 脚下没地板，下落")
            return

    def _find_floor_top(self, ctx: Context, x: int, y: int) -> Optional[int]:
        """找到覆盖 (x, y) 的平台，返回其"最上面的 y"（top_y）。

        找不到返回 None。
        """
        best = None
        best_key = float("inf")
        for f in ctx.floors:
            if f.x <= x <= f.x + f.w:
                if f.y - 30 <= y <= f.y + f.h + 30:
                    d = abs(f.y - y)
                    if d < best_key:
                        best_key = d
                        best = f.y
        return best

    # =========================================================================
    # 探索
    # =========================================================================

    def _explore(self, ctx: Context):
        """画面里没怪时，往一个方向走探索。

        行为:
          - 往探索方向走
          - 遇到平台边缘（脚下没地板）就跳
          - 卡住时反向走
          - 长时间没遇到怪就换方向
        """
        self._explore_frame_count += 1

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[探索] 卡住了，跳跃并反向")
            self._explore_direction = "left" if self._explore_direction == "right" else "right"
            self.executor.press_key("alt", cooldown=1.0)
            self._stuck_counter = 0
            return

        # 长时间探索没遇到怪，换方向
        if self._explore_frame_count >= EXPLORE_DIRECTION_SWITCH_FRAMES:
            self._explore_direction = "left" if self._explore_direction == "right" else "right"
            self._explore_frame_count = 0
            self._log(f"[探索] 换方向 → {self._explore_direction}")

        # 检测脚下是否有地板
        if ctx.self_position and not self._has_floor_under(ctx, ctx.self_position):
            self._log("[探索] 脚下没地板，跳跃")
            self.executor.press_key("alt", cooldown=1.0)
            return

        # 往前走
        self.executor.press_key(self._explore_direction, cooldown=0.05)

    # =========================================================================
    # 技能释放
    # =========================================================================

    def _cast_skill(self):
        """轮转释放技能。"""
        skills = self.config.skills
        if not skills:
            return
        for _ in range(len(skills)):
            skill = skills[self._skill_index % len(skills)]
            self._skill_index += 1
            if self.executor.press_key(skill["key"], skill["cooldown"]):
                self._log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                break