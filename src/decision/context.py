"""决策上下文与决策引擎。

``Context`` 汇总感知层每帧产出的数据（HP/MP 比例、怪物列表），
``DecisionEngine`` 根据 Context + Config 产出并执行动作指令。

当前实现沿用原 ``app/automation.py`` 中 ``_decide()`` 的简单 if-else 决策
逻辑（加血 → 加蓝 → 选目标/技能）。后续可替换为 ``fsm.py`` 中的有限状态机。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any

from ..perception.yolo_detector import Detection
from ..perception.screen_capture import ScreenCapture
from ..execution.action_executor import ActionExecutor
from ..utils.config_loader import Config


@dataclass
class Context:
    """感知层 → 决策层的数据载体（每帧一份）。"""
    monsters: List[Detection] = field(default_factory=list)
    hp_ratio: Optional[float] = None
    mp_ratio: Optional[float] = None
    detections: List[Detection] = field(default_factory=list)  # 全部检测（含非怪物）


class DecisionEngine:
    """决策引擎：根据 Context + Config 决定下一步动作并执行。

    决策优先级（沿用原 ``_decide()`` 逻辑）：
      1. HP 低 → 加血键
      2. MP 低 → 加蓝键
      3. 检测到怪物：
         - 若 ``move_to_monster=True``，点击怪位置（屏幕坐标）
         - 否则按选目标键 + 轮转释放技能
      4. 没怪：按选目标键自动寻找目标
    """

    def __init__(self, config: Config, executor: ActionExecutor,
                 capture: Optional[ScreenCapture] = None,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.executor = executor
        self.capture = capture
        self._log = on_log or (lambda m: None)
        self._skill_index = 0  # 技能轮转索引

    def update_config(self, config: Config):
        self.config = config

    def update_capture(self, capture: ScreenCapture):
        self.capture = capture

    def reset(self):
        self._skill_index = 0
        self.executor.reset()

    def _monster_classes(self) -> List[str]:
        return [c.strip() for c in self.config.monster_classes.split(",") if c.strip()]

    def decide(self, ctx: Context):
        """根据上下文执行决策。返回 None（动作通过 executor 直接触发）。"""
        # 1) 没血优先加血
        if ctx.hp_ratio is not None and ctx.hp_ratio < self.config.hp_threshold:
            if self.executor.press_key(self.config.hp_key, cooldown=1.5):
                self._log(
                    f"[加血] HP={ctx.hp_ratio:.0%} < {self.config.hp_threshold:.0%}，"
                    f"按下 {self.config.hp_key}"
                )
                return

        # 2) 没蓝加蓝
        if ctx.mp_ratio is not None and ctx.mp_ratio < self.config.mp_threshold:
            if self.executor.press_key(self.config.mp_key, cooldown=1.5):
                self._log(
                    f"[加蓝] MP={ctx.mp_ratio:.0%} < {self.config.mp_threshold:.0%}，"
                    f"按下 {self.config.mp_key}"
                )
                return

        # 3) 检测到怪物
        if ctx.monsters:
            target = max(ctx.monsters, key=lambda d: d.w * d.h)
            cx, cy = target.center
            self._log(
                f"[检测] {target.cls_name} conf={target.confidence:.2f} @ ({cx},{cy})"
            )

            # 可选：点击移动到怪位置（窗口内坐标转屏幕坐标）
            if self.config.move_to_monster and self.capture is not None:
                rect = self.capture.get_rect()
                self.executor.click(rect[0] + cx, rect[1] + cy)
                return

            # 选中目标
            self.executor.press_key(self.config.target_key, cooldown=0.8)

            # 轮转释放技能（挑一个冷却好的）
            skills = self.config.skills
            for _ in range(len(skills)):
                skill = skills[self._skill_index % len(skills)]
                self._skill_index += 1
                if self.executor.press_key(skill["key"], skill["cooldown"]):
                    self._log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                    break
        else:
            # 没怪，按选中键自动寻找目标
            self.executor.press_key(self.config.target_key, cooldown=1.5)
