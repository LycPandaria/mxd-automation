"""决策引擎冒烟测试（本地主干）。

目的不是覆盖全部行为，而是先证明：
  1. 决策引擎可以在无 Windows / 无游戏 / 无模型的环境里被"喂帧"跑起来；
  2. 三条最基本的分支（加血、在射程内出手、没怪探索）不会回归。

逐功能的细粒度用例（跳跃攻击、活动范围、残影、风筝、站桩回位等）
在 share-baseline 分支上，等对应功能移植过来时再逐个搬（见 tests/README.md）。
"""
import time
import unittest

from src.decision.context import Context, DecisionEngine
from src.decision.fsm import State
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config

from .fake_executor import FakeExecutor

GROUND = 600  # 脚底 y


def monster(cx, foot=GROUND, w=60, h=60):
    return Detection(cls_name="monster", confidence=0.9,
                     x=cx - w // 2, y=foot - h, w=w, h=h)


def engine(**overrides):
    cfg = Config(overrides={
        "attack_type": "long", "attack_range": 400, "attack_range_y": 60,
        "turn_mode": "tap", "aoe_mode": "burst",
        "stand_mode": False, "fps": 20,
        "hp_threshold": 0.01, "mp_threshold": 0.01,
        "skills": [{"name": "连击", "key": "a", "cooldown": 0.5}],
        "buff_skills": [],
        "knockback_skill": {},
        "stand_micro_move_enabled": False,
        "pickup_enabled": False,
        # 活动范围钉死为不限制：Config 会合并本机 user.yaml，
        # 不钉的话本机设了范围（如 0~80）会泄漏进来，探索/打怪用例会被拦
        "range_start": 0.0, "range_end": 100.0,
        **overrides,
    })
    ex = FakeExecutor()
    e = DecisionEngine(cfg, ex)
    # 朝向已确认且近期不重申：让用例只考察"攻击判定 → 出手"这一段
    e._face_dir = "right"
    e._face_next_assert = time.time() + 1000
    return e


class SmokeTests(unittest.TestCase):
    def test_casts_when_target_in_range(self):
        e = engine()
        ctx = Context(monsters=[monster(1200)], self_position=(1000, GROUND))
        e.decide(ctx)
        self.assertIn("a", e.executor.keys())
        self.assertEqual(e._fsm.current, State.ATTACKING)

    def test_heals_after_react_delay(self):
        e = engine(hp_threshold=0.5, hp_key="end")
        for _ in range(15):  # ~0.5s 反应延迟 @20fps
            e.decide(Context(self_position=(1000, GROUND), hp_ratio=0.3))
        self.assertIn("end", e.executor.keys())

    def test_explores_when_no_monster(self):
        e = engine()
        e.decide(Context(self_position=(1000, GROUND)))
        self.assertIn("right", e.executor.held)

    def test_empty_context_does_not_crash(self):
        e = engine()
        e.decide(Context())  # 什么都没检测到（启动第一帧）
        self.assertEqual(e.executor.held, {"right"})

    def test_same_level_check_uses_foot_y(self):
        e = engine()
        # 怪在上一层（脚底 y 差 200px）→ 不应当被选中
        ctx = Context(monsters=[monster(1200, foot=GROUND - 200)],
                      self_position=(1000, GROUND))
        e.decide(ctx)
        self.assertNotIn("a", e.executor.keys())


if __name__ == "__main__":
    unittest.main()
