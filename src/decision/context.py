"""决策上下文与决策引擎。

================================================================================
职责
================================================================================

  Context:      感知层 → 决策层的数据载体（每帧一份）
  DecisionEngine: 根据 Context + Config 决定下一步动作并执行

================================================================================
决策流程（优先级从高到低）
================================================================================

  1. HP 低于阈值 → 按加血键
  2. MP 低于阈值 → 按加蓝键
  3. 检测到怪物:
     a. 若 move_to_monster=True → 点击怪位置（屏幕坐标）
     b. 否则 → 按选目标键 + 轮转释放技能
  4. 没怪 → 按选目标键自动寻找目标

================================================================================
技能轮转机制
================================================================================

  DecisionEngine 维护一个 _skill_index 计数器，每次释放技能时自增。
  轮转时跳过冷却中的技能，直到找到一个冷却已好的。

  例如技能列表: [技能1(CD=1s), 技能2(CD=3s), 技能3(CD=8s)]
  轮转顺序:    1→2→3→1→2→3→1→2→3→...
  但 2 在冷却中时会跳过，直接到 3。

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
from typing import List, Optional, Tuple, Callable, Any

from ..perception.yolo_detector import Detection
from ..perception.screen_capture import ScreenCapture
from ..execution.action_executor import ActionExecutor
from ..utils.config_loader import Config


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
         - 若 move_to_monster=True → 点击怪位置
         - 否则 → 按选目标键 + 轮转释放技能
      4. 没怪 → 按选目标键自动寻找目标

    【为什么需要 capture 引用】
    当 move_to_monster=True 时，需要把怪物的窗口内坐标转成屏幕坐标，
    才能通过 PostMessage 发送鼠标点击。没有 capture 引用就做不到这个转换。

    【技能轮转】
    维护 _skill_index 计数器，每次释放技能后自增。
    轮转时跳过冷却中的技能，直到找到一个可用的。

    Args:
        config:   全局配置
        executor: 动作执行器
        capture:  截图器（用于坐标转换），None 表示不支持点击移动
        on_log:   日志回调
    """

    def __init__(self, config: Config, executor: ActionExecutor,
                 capture: Optional[ScreenCapture] = None,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.executor = executor
        self.capture = capture
        self._log = on_log or (lambda m: None)
        self._skill_index = 0  # 技能轮转索引，每次释放后自增

    def update_config(self, config: Config):
        """更新配置引用（UI 修改配置后调用）。"""
        self.config = config

    def update_capture(self, capture: ScreenCapture):
        """更新截图器引用（窗口切换后调用）。"""
        self.capture = capture

    def reset(self):
        """重置状态：清空技能轮转索引和按键冷却记录。"""
        self._skill_index = 0
        self.executor.reset()

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
        一旦某个条件命中并执行了动作，立即 return 不再继续。

        注意：目前只处理了 monsters 字段，floors/ropes/players/self_position
        预留供后续扩展（寻路、避障、跟随等）。

        Args:
            ctx: 当前帧的感知数据
        """
        # ---- 优先级 1: 没血加血 ----
        # hp_ratio < hp_threshold 时触发
        # cooldown=1.5: 加血后 1.5 秒内不重复触发
        if ctx.hp_ratio is not None and ctx.hp_ratio < self.config.hp_threshold:
            if self.executor.press_key(self.config.hp_key, cooldown=1.5):
                self._log(
                    f"[加血] HP={ctx.hp_ratio:.0%} < {self.config.hp_threshold:.0%}，"
                    f"按下 {self.config.hp_key}"
                )
                return  # 执行了加血就返回，不再往下走

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
            # 选面积最大的怪物作为目标（通常是离得最近或威胁最大的）
            target = max(ctx.monsters, key=lambda d: d.w * d.h)
            cx, cy = target.center
            self._log(
                f"[检测] {target.cls_name} conf={target.confidence:.2f} @ ({cx},{cy})"
            )

            # 3a: 可选 — 点击移动到怪物位置
            # move_to_monster=True 时，把窗口内坐标转成屏幕坐标再点击
            if self.config.move_to_monster and self.capture is not None:
                # get_rect() 返回 (left, top, width, height)
                # 窗口内坐标 (cx, cy) + 窗口左上角屏幕坐标 (left, top) = 屏幕坐标
                rect = self.capture.get_rect()
                self.executor.click(rect[0] + cx, rect[1] + cy)
                return

            # 3b: 按选目标键选中怪物
            self.executor.press_key(self.config.target_key, cooldown=0.8)

            # 3c: 轮转释放技能
            skills = self.config.skills
            # 遍历技能列表，最多尝试一轮（len(skills) 次）
            for _ in range(len(skills)):
                # 取当前轮转位置的技能
                skill = skills[self._skill_index % len(skills)]
                self._skill_index += 1  # 索引自增，下一帧用下一个技能
                # can_press 检查 + press_key 发送（带冷却）
                if self.executor.press_key(skill["key"], skill["cooldown"]):
                    self._log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                    break  # 成功释放一个技能就退出
        else:
            # ---- 优先级 4: 没怪，按选目标键自动寻找目标 ----
            self.executor.press_key(self.config.target_key, cooldown=1.5)