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
     a. 同平台 → 按住方向键走过去，进入 200px 攻击范围后停止移动原地攻击
     b. 怪在上方 + 有绳索 → 走到绳索正下方，跳跃 + 按住上键爬绳
     c. 怪在下方/跨平台 → 按住方向键移动 + 按需跳跃
     d. 都不满足 → Tab 选怪 + 原地攻击
  4. 没怪 → 探索（按住方向键往一个方向走，遇坑跳）

【移动方式】所有移动（追击/攀爬/探索）都是"按住方向键不松手"，
  进入攻击范围或攻击时才释放方向键，攻击期间完全不移动。

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
    SAME_LEVEL_Y_TOLERANCE,
)


# =============================================================================
# 反应式决策的阈值常量
# =============================================================================

ATTACK_RANGE_X = 200
"""攻击范围：自身与怪物 X 坐标差小于此值（约 200px 附近）才开始攻击（像素）"""

ROPE_SEARCH_RANGE_X = 200
"""搜索绳索的水平范围（像素）"""

CLIMB_ALIGN_TOLERANCE = 5
"""攀爬对准容差：人物中心与绳索中心 X 差小于此值（±5px）
视为在同一竖直轴线（绳索正下方），才允许抓绳攀爬（像素）"""

CLIMB_EXIT_FRAMES = 45
"""爬绳结束后横向走出绳索的帧数（约 2 秒 @20fps），期间不重新抓绳"""

ATTACK_STALE_FRAMES = 90
"""锁定同一目标持续攻击的最大帧数（约 4.5 秒 @20fps）。

怪物死亡后 YOLO 仍可能把尸体/消失残影检测为 monster，
位置匹配会一直锁定这个残影，导致角色原地打空气、不换下一只。
超过该帧数目标仍未消失（画面仍检测到）→ 判定为残影/无敌，
立即解除锁定重新选目标。"""

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

        # 移动键按住状态（持续移动/攀爬）
        self._held_key: Optional[str] = None      # 当前按住的键（left/right/up/down）
        self._climbing = False                    # 是否正在沿绳索攀爬
        self._climb_exit_frames = 0               # 脱离绳索后横向走出的剩余帧数
        self._climb_log_count = 0                 # 攀爬日志限频计数
        self._attack_stale_counter = 0            # 锁定同一目标持续攻击的帧数（残影检测）

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
        self._attack_stale_counter = 0
        self.release_keys()
        self._fsm.reset()
        self.executor.reset()

    def release_keys(self):
        """释放所有按住的移动键（停止时调用，防止方向键卡住）。"""
        self._release_move()
        self._climbing = False
        self._climb_exit_frames = 0
        self._climb_log_count = 0

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
                self._release_move()  # 加血时站住不动
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
                self._release_move()  # 加蓝时站住不动
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
            # 画面中已没有怪物：立即解除锁定并清理攀爬等残留状态，
            # 防止"上帧还锁着怪/在爬绳"的状态影响后续探索与重新选怪
            self._target_monster = None
            self._attack_stale_counter = 0
            self._climbing = False
            self._climb_exit_frames = 0
            self._explore(ctx)

    # =========================================================================
    # 怪物处理
    # =========================================================================

    def _handle_monsters(self, ctx: Context):
        """处理画面中的怪物。

        【锁定机制】找到怪物后优先持续攻击同一只，直到它消失：
          1. 若已锁定目标且仍能在画面中匹配到（位置接近的同一只怪）
             → 继续攻击它，不去换别的怪
          2. 若锁定目标已消失（匹配不到）→ 立即清除锁定，重新选最近目标
          3. 【残影防护】持续攻击同一目标超过 ATTACK_STALE_FRAMES 帧
             仍未击杀（画面仍检测到）→ 判定为死尸残影/无敌，
             解除锁定重新选目标，避免原地打空气半天不动
        """
        target = self._resolve_locked_target(ctx)

        # ---- 攻击超时检测（残影防护）----
        # 只有"一直锁定同一只 + 处于攻击状态"才累计；换目标/追击中不计
        if self._target_monster is not None and self._is_same_monster(
                self._target_monster, target):
            if self._fsm.current == State.ATTACKING:
                self._attack_stale_counter += 1
            else:
                self._attack_stale_counter = 0
        else:
            self._attack_stale_counter = 0

        # 同一只怪攻击过久仍没死 → 很可能是尸体残影/无敌，强制换目标
        if self._attack_stale_counter >= ATTACK_STALE_FRAMES:
            self._log("[换目标] 持续攻击无效果(疑似残影/已死)，重新锁定最近目标")
            self._target_monster = None
            self._attack_stale_counter = 0
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
    # 按住移动
    # =========================================================================

    def _hold_move(self, direction: str):
        """按住方向键持续移动。

        切换方向时先释放旧键再按住新键，避免两个方向键同时按下。
        移动期间不松手，直到调用 _release_move() 停止。
        """
        if direction not in ("left", "right", "up", "down"):
            return
        if self._held_key == direction:
            return
        if self._held_key:
            self.executor.key_up(self._held_key)
        self.executor.key_down(direction)
        self._held_key = direction

    def _release_move(self):
        """释放当前按住的移动键（停止移动/攀爬）。"""
        if self._held_key:
            self.executor.key_up(self._held_key)
            self._held_key = None

    # =========================================================================
    # 目标选择
    # =========================================================================

    def _resolve_locked_target(self, ctx: Context) -> Detection:
        """解析当前应攻击的目标怪物（含锁定逻辑）。

        优先保持已锁定的目标（直到它消失），否则重新选择最近目标。

        Returns:
            当前应攻击的怪物；若画面中有怪，必定返回一个。
        """
        # 已锁定目标：尝试在当前画面中继续匹配（位置接近的同一只怪）
        if self._target_monster is not None:
            matched = self._match_monster(ctx.monsters, self._target_monster)
            if matched is not None:
                return matched  # 继续锁定同一只
            # 锁定目标已从画面消失 → 立即重新选择，不等不拖
            self._log("[锁定] 目标已消失，立即重新选择最近目标")

        # 锁定目标已消失或尚未锁定 → 重新选择最近目标
        return self._pick_best_target(ctx)

    def _is_same_monster(self, a: Detection, b: Detection) -> bool:
        """按中心点距离判断两个检测是否可能是同一只怪。

        容差与锁定匹配一致（目标 bbox 宽度的 1.5 倍，保底 40px），
        用于攻击超时统计。
        """
        if a is None or b is None:
            return False
        tolerance = max(int(a.w * 1.5), 40)
        d = abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])
        return d <= tolerance

    def _match_monster(self, monsters: List[Detection],
                       locked: Detection) -> Optional[Detection]:
        """在当前怪物列表中匹配已锁定的目标。

        用中心点距离匹配：锁定时记录目标中心，之后每帧找
        与之位置最接近的怪物。若两者距离小于锁定容差
        （目标 bbox 宽度的 1.5 倍，保底 40px），视为同一只怪。

        Args:
            monsters: 当前帧检测到的怪物列表
            locked:   已锁定的目标怪物

        Returns:
            匹配到的当前怪物；匹配不到返回 None（视为已消失）
        """
        if not monsters:
            return None
        lx, ly = locked.center
        tolerance = max(int(locked.w * 1.5), 40)
        best = None
        best_dist = float("inf")
        for m in monsters:
            mx, my = m.center
            d = abs(mx - lx) + abs(my - ly)
            if d < best_dist:
                best, best_dist = m, d
        if best_dist <= tolerance:
            return best
        return None

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
        """按住方向键持续走向怪物。

        到达攻击范围前不松手（按住方向键移动），进入攻击范围后
        由上层切换为攻击状态并释放方向键。
        """
        if ctx.self_position is None:
            self._release_move()
            return
        sx = ctx.self_position[0]
        tx = target.center[0]

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[追击] 卡住了，尝试跳跃")
            self._release_move()
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        if tx > sx + 10:
            self._hold_move("right")
        elif tx < sx - 10:
            self._hold_move("left")
        else:
            self._release_move()

    # =========================================================================
    # 攻击
    # =========================================================================

    def _attack(self, ctx: Context, target: Detection):
        """在攻击范围内原地释放技能（攻击时不移动）。

        进入攻击范围后停止移动（释放方向键），原地轮转放技能。
        """
        self._release_move()  # 攻击时保持不动
        self._cast_skill()

    def _tab_attack(self, ctx: Context):
        """Tab 选怪 + 原地攻击（兜底方案，不移动）。"""
        self._release_move()
        self.executor.press_key(self.config.target_key, cooldown=0.8)
        self._cast_skill()

    # =========================================================================
    # 攀爬
    # =========================================================================

    def _try_climb(self, ctx: Context, target: Detection) -> bool:
        """攀爬追怪：走到绳索正下方 → 跳跃 + 按住上键爬绳。

        流程:
          1. 脱离绳索后的横向走出阶段（不重新抓绳）
          2. 找最近绳索（水平 ROPE_SEARCH_RANGE_X 内）
          3. 【对准判定】人物中心与绳索中心是否在同一竖直轴线
             （X 差 <= 5px）→ 否，先水平移动到绳索正下方
          4. 已对准 → 按跳跃 + 按住上键沿绳索向上爬
          5. 爬到怪物所在高度（人物中心与怪物中心 Y 差 <= 30px）
             或爬到绳顶 → 停止爬绳，横向走出绳索继续追击

        返回 True 表示找到了绳索并执行了动作，False 表示没找到。
        """
        # 用人物中心点（不是脚底），回退脚底
        player = ctx.self_center or ctx.self_position
        if player is None:
            self._release_move()
            return False
        sx, sy = player
        tx = target.center[0]

        # 刚爬完绳，横向走出绳索（此阶段不重新抓绳）
        if self._climb_exit_frames > 0:
            self._climb_exit_frames -= 1
            if tx > sx:
                self._hold_move("right")
            else:
                self._hold_move("left")
            return True

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
            self._climbing = False
            self._release_move()
            return False

        rx, ry = nearest_rope.center

        # ---- 阶段 1: 对准判定（人物中心与绳索中心同一竖直轴线）----
        if not self._climbing:
            if abs(rx - sx) > CLIMB_ALIGN_TOLERANCE:
                # 不在绳索正下方 → 水平移动对准
                if rx > sx:
                    self._hold_move("right")
                else:
                    self._hold_move("left")
                return True
            # 人物中心与绳索中心在同一竖直轴线（±5px）→ 跳跃 + 按住上键爬绳
            self._release_move()
            self._climbing = True
            self._log("[攀爬] 对准绳索，跳跃并开始攀爬")
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._hold_move("up")
            return True

        # ---- 阶段 2: 正在爬绳 ----
        ty = target.center[1]

        # 到达条件: 人物中心与怪物中心 Y 差 <= 30px（已爬到怪物所在层）
        if abs(sy - ty) <= SAME_LEVEL_Y_TOLERANCE:
            self._climbing = False
            self._climb_exit_frames = CLIMB_EXIT_FRAMES
            self._release_move()
            self._log("[攀爬] 已到达怪物所在高度，脱离绳索")
            return True

        # 爬到绳顶（人物中心已接近绳索顶部）仍没到怪物高度 → 停止，横向走出
        if sy <= ry - (nearest_rope.h / 2) + 10:
            self._climbing = False
            self._climb_exit_frames = CLIMB_EXIT_FRAMES
            self._release_move()
            self._log("[攀爬] 已到绳顶仍追不上，脱离绳索")
            return True

        # 还没到 → 继续按住上键向上爬（日志限频，防刷屏）
        self._hold_move("up")
        self._climb_log_count += 1
        if self._climb_log_count % 15 == 1:
            self._log("[攀爬] 沿绳索向上")
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
            self._release_move()
            return
        sx, sy = ctx.self_position
        tx, ty = target.center[0], target.center[1]

        # 怪物所在平台的 top_y（最上面的 y）
        target_top = self._find_floor_top(ctx, tx, ty)

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[跳跃追击] 卡住了，起跳")
            self._release_move()
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # 水平方向按住走向怪物（跳跃时保持按住，跳得更远）
        if tx > sx + 10:
            self._hold_move("right")
        elif tx < sx - 10:
            self._hold_move("left")
        else:
            self._release_move()

        # 脚下有地板
        on_floor = self._has_floor_under(ctx, ctx.self_position)

        if on_floor and target_top is not None:
            # 怪在上层：高度差在跳跃高度内就起跳
            if ty < sy and (sy - target_top) <= JUMP_HEIGHT:
                self._log(
                    f"[跳跃追击] 目标平台 top_y={target_top}，"
                    f"高度差 {sy - target_top}px <= {JUMP_HEIGHT}px，起跳"
                )
                self.executor.press_key(self.config.jump_key, cooldown=1.0)
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
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
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
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
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
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            return

        # 按住方向键持续往前走
        self._hold_move(self._explore_direction)

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