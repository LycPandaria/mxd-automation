"""决策上下文与决策引擎。

================================================================================
职责
================================================================================

  Context:      感知层 → 决策层的数据载体（每帧一份）
  DecisionEngine: 根据 Context + Config 决定下一步动作并执行

================================================================================
整体架构
================================================================================

  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │  感知层      │ →  │  MapExplorer  │ →  │  GlobalMap   │
  │ YOLO/OCR/HP │    │  (建图)       │    │  (持久化)    │
  └─────────────┘    └──────────────┘    └──────┬───────┘
                                                │
  ┌─────────────┐    ┌──────────────┐           │
  │  ActionExec │ ←  │ DecisionEngine│ ← A*Pathfinder │
  │  (方向键)   │    │  (决策)       │    (寻路)      │
  └─────────────┘    └──────────────┘    └──────────────┘

  两种运行模式:
    探索模式: 边走边建图，同时打怪（建图和战斗可以同时进行）
    战斗模式: 地图已加载，直接用 GlobalMap 规划路径

================================================================================
决策流程（优先级从高到低）
================================================================================

  1. HP 低于阈值 → 按加血键
  2. MP 低于阈值 → 按加蓝键
  3. 检测到怪物:
     a. 若 move_to_monster=True → A* 在全局地图上寻路 + 方向键移动
     b. 否则 → 按选目标键 + 轮转释放技能
  4. 没怪 → 按选目标键自动寻找目标

================================================================================
寻路机制（move_to_monster=True 时）
================================================================================

  冒险岛是 2D 横版游戏，移动靠方向键（← → ↑ ↓ + 跳跃），不能用鼠标点击。

  每帧:
    1. MapExplorer.update() 把当前帧的 floors/ropes 拼接到 GlobalMap
    2. 用 AStarPathfinder 在 GlobalMap 上搜索从自身到怪物的路径
    3. 将路径转换为方向键指令并执行
    4. 寻路失败 → 回退到 Tab 选同平台怪

================================================================================
技能轮转机制
================================================================================

  DecisionEngine 维护一个 _skill_index 计数器，每次释放技能时自增。
  轮转时跳过冷却中的技能，直到找到一个冷却已好的。

================================================================================
Context 字段说明
================================================================================

  monsters:       YOLO 检测到的怪物列表（已过滤类别）
  floors:         地板列表
  ropes:          绳索列表
  players:        其他玩家列表
  self_position:  自身脚底坐标 (cx, cy) 或 None
  hp_ratio:       血量比例 0.0~1.0
  mp_ratio:       蓝量比例 0.0~1.0
  detections:     全部 YOLO 检测结果（含所有类别）
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

from ..perception.yolo_detector import Detection
from ..execution.action_executor import ActionExecutor
from ..utils.config_loader import Config
from .global_map import GlobalMap
from .map_explorer import MapExplorer
from .astar import AStarPathfinder, path_to_directions


@dataclass
class Context:
    """感知层 → 决策层的数据载体（每帧一份）。

    这是一个纯数据类（dataclass），没有任何行为逻辑。
    DecisionEngine 读取 Context 的字段来决定下一步动作。

    self_position 由 HP 条区域推算或 OCR 识别得到（窗口内坐标），
    不依赖 YOLO 检测自身的类别。
    """
    monsters: List[Detection] = field(default_factory=list)
    """YOLO 检测到的怪物"""

    floors: List[Detection] = field(default_factory=list)
    """地板检测结果"""

    ropes: List[Detection] = field(default_factory=list)
    """绳索检测结果"""

    players: List[Detection] = field(default_factory=list)
    """其他玩家检测结果"""

    self_position: Optional[Tuple[int, int]] = None
    """自身脚底坐标 (x, y) 窗口内坐标，None 表示无法定位"""

    hp_ratio: Optional[float] = None
    """血量比例 (0.0 ~ 1.0)，None 表示未检测到"""

    mp_ratio: Optional[float] = None
    """蓝量比例 (0.0 ~ 1.0)，None 表示未检测到"""

    detections: List[Detection] = field(default_factory=list)
    """全部 YOLO 检测结果（含所有类别，供调试/日志用）"""


class DecisionEngine:
    """决策引擎：根据 Context + Config 决定下一步动作并执行。

    【职责】
    读 Context（感知数据），按 Config（配置），通过 ActionExecutor（执行层）触发动作。

    【决策优先级】
      1. HP 低 → 加血键
      2. MP 低 → 加蓝键
      3. 检测到怪物：
         - 若 move_to_monster=True → A* 寻路 + 方向键移动到怪物平台
         - 否则 → 按选目标键 + 轮转释放技能
      4. 没怪 → 按选目标键自动寻找目标

    【探索 + 建图】
      MapExplorer 每帧接收 YOLO 检测到的 floors/ropes，逐帧拼接成 GlobalMap。
      探索和战斗可以同时进行：一边打怪一边建图。

    【寻路】
      move_to_monster=True 时，AStarPathfinder 在 GlobalMap 上搜索路径。
      路径缓存到 _current_path，多帧持续执行直到到达目标。

    Args:
        config:     全局配置
        executor:   动作执行器
        map_explorer: 地图探索器（可选，不传则自动创建）
        on_log:     日志回调
    """

    def __init__(self, config: Config, executor: ActionExecutor,
                 capture=None,
                 map_explorer: Optional[MapExplorer] = None,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.executor = executor
        self._log = on_log or (lambda m: None)
        self._skill_index = 0

        # 地图探索器: 逐帧拼接 GlobalMap
        self._explorer = map_explorer or MapExplorer(on_log=self._log)

        # A* 寻路器: 在 GlobalMap 上搜索路径（每次 GlobalMap 更新后重建）
        self._pathfinder: Optional[AStarPathfinder] = None

        # 当前寻路路径缓存: 多帧持续执行移动
        self._current_path: List[Tuple[int, int]] = []
        self._path_target: Optional[Tuple[int, int]] = None  # 路径的目标全局坐标

    def update_config(self, config: Config):
        """更新配置引用（UI 修改配置后调用）。"""
        self.config = config

    def reset(self):
        """重置状态：清空技能轮转索引、按键冷却记录和寻路缓存。"""
        self._skill_index = 0
        self._current_path = []
        self._path_target = None
        self.executor.reset()

    @property
    def explorer(self) -> MapExplorer:
        """获取地图探索器（供外部查询探索状态）。"""
        return self._explorer

    # =========================================================================
    # 类别名解析
    # =========================================================================

    def _monster_classes(self) -> List[str]:
        return [c.strip() for c in self.config.monster_classes.split(",") if c.strip()]

    def _floor_classes(self) -> List[str]:
        return [c.strip() for c in self.config.floor_classes.split(",") if c.strip()]

    def _rope_classes(self) -> List[str]:
        return [c.strip() for c in self.config.rope_classes.split(",") if c.strip()]

    def _player_classes(self) -> List[str]:
        return [c.strip() for c in self.config.player_classes.split(",") if c.strip()]

    # =========================================================================
    # 决策主逻辑
    # =========================================================================

    def decide(self, ctx: Context):
        """根据上下文执行决策。

        每帧调用一次，按优先级判断是否需要执行动作。
        无论是否寻路，都会先更新 MapExplorer（建图）。

        寻路模式 (move_to_monster=True):
          1. MapExplorer.update() 把当前帧拼接到 GlobalMap
          2. A* 在 GlobalMap 上搜索从自身到怪物的路径
          3. 路径存在 → 转换为方向键指令并执行
          4. 路径不存在 → 回退到 Tab 选怪 + 技能

        Args:
            ctx: 当前帧的感知数据
        """
        # ---- 始终更新地图探索器（建图） ----
        # 探索和战斗可以同时进行，每帧都拼接 floors/ropes
        self._update_map(ctx)

        # ---- 优先级 1: 没血加血 ----
        if ctx.hp_ratio is not None and ctx.hp_ratio < self.config.hp_threshold:
            if self.executor.press_key(self.config.hp_key, cooldown=1.5):
                self._log(
                    f"[加血] HP={ctx.hp_ratio:.0%} < {self.config.hp_threshold:.0%}，"
                    f"按下 {self.config.hp_key}"
                )
                return

        # ---- 优先级 2: 没蓝加蓝 ----
        if ctx.mp_ratio is not None and ctx.mp_ratio < self.config.mp_threshold:
            if self.executor.press_key(self.config.mp_key, cooldown=1.5):
                self._log(
                    f"[加蓝] MP={ctx.mp_ratio:.0%} < {self.config.mp_threshold:.0%}，"
                    f"按下 {self.config.mp_key}"
                )
                return

        # ---- 优先级 3: 检测到怪物 ----
        if ctx.monsters:
            target = max(ctx.monsters, key=lambda d: d.w * d.h)
            cx, cy = target.center
            self._log(
                f"[检测] {target.cls_name} conf={target.confidence:.2f} @ ({cx},{cy})"
            )

            # 3a: 寻路模式 — 在 GlobalMap 上 A* 寻路 + 方向键移动
            if self.config.move_to_monster and ctx.self_position:
                if self._try_navigate(ctx, target):
                    return
                # 寻路失败，回退到简单模式

            # 3b: 简单模式 — 按选目标键选中怪物
            self.executor.press_key(self.config.target_key, cooldown=0.8)

            # 3c: 轮转释放技能
            self._cast_skill()
        else:
            # ---- 优先级 4: 没怪，按选目标键自动寻找目标 ----
            self.executor.press_key(self.config.target_key, cooldown=1.5)

    # =========================================================================
    # 地图更新
    # =========================================================================

    def _update_map(self, ctx: Context):
        """将当前帧的 YOLO 检测结果拼接到 GlobalMap。

        每帧调用，无论是否在战斗中。
        """
        self._explorer.update(
            ctx.floors, ctx.ropes,
            ctx.self_position,
            frame_width=1366,  # TODO: 从 capture 获取实际帧尺寸
            frame_height=768,
        )

    # =========================================================================
    # 寻路与导航
    # =========================================================================

    def _try_navigate(self, ctx: Context, target: Detection) -> bool:
        """尝试在 GlobalMap 上寻路并执行移动。

        返回 True 表示成功执行了移动指令，False 表示寻路失败。

        Args:
            ctx:    当前帧感知数据
            target: 目标怪物

        Returns:
            True 表示已执行移动（可以 return），False 表示需要回退
        """
        global_map = self._explorer.current_map
        if global_map is None or global_map.explored_count == 0:
            return False

        # 转换坐标: 窗口坐标 → 全局坐标
        self_global = self._explorer.window_to_global(
            ctx.self_position[0], ctx.self_position[1]
        )
        target_global = self._explorer.window_to_global(
            target.center[0], target.center[1]
        )

        # 检查目标是否与当前路径目标相同（避免每帧重新寻路）
        if self._path_target == target_global and self._current_path:
            # 路径还在，继续执行移动
            self._execute_move(self._current_path, self_global)
            return True

        # 重建 A* 寻路器（GlobalMap 可能已更新）
        if self._pathfinder is None or self._pathfinder._map is not global_map:
            self._pathfinder = AStarPathfinder(global_map)

        # A* 搜索
        path = self._pathfinder.find_path(self_global, target_global)

        if path:
            self._log(f"[寻路] 找到路径，共 {len(path)} 个路径点")
            self._current_path = path
            self._path_target = target_global
            self._execute_move(path, self_global)
            return True
        else:
            self._log(f"[寻路] 无法到达目标，回退到 Tab 选怪")
            self._current_path = []
            self._path_target = None
            return False

    def _execute_move(self, path: List[Tuple[int, int]],
                      self_pos: Tuple[int, int]):
        """根据路径生成方向键指令并执行。

        取路径的下一个点，判断相对当前位置的方向，按对应的方向键。

        Args:
            path:     A* 返回的路径点列表（全局像素坐标）
            self_pos: 当前自身全局坐标
        """
        commands = path_to_directions(path, self_pos)
        for cmd in commands:
            if cmd == "jump":
                if len(path) > 1:
                    next_pt = path[1]
                    if next_pt[0] > self_pos[0]:
                        self.executor.press_key("right", cooldown=0.1)
                    elif next_pt[0] < self_pos[0]:
                        self.executor.press_key("left", cooldown=0.1)
                self.executor.press_key("alt", cooldown=1.0)
            elif cmd == "down":
                self.executor.press_key("down", cooldown=0.3)
            elif cmd in ("left", "right"):
                self.executor.press_key(cmd, cooldown=0.1)

    # =========================================================================
    # 技能释放
    # =========================================================================

    def _cast_skill(self):
        """轮转释放技能。

        遍历技能列表，跳过冷却中的技能，释放第一个可用的。
        """
        skills = self.config.skills
        for _ in range(len(skills)):
            skill = skills[self._skill_index % len(skills)]
            self._skill_index += 1
            if self.executor.press_key(skill["key"], skill["cooldown"]):
                self._log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                break