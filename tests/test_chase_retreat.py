"""追击节奏 / 近身后撤突围。

对应第 3 批第 3、5 点：
  3. 刚出过手先等怪过来；追击卡住两次放弃并记残影
  5. 贴脸判定看最近同层威胁；撤不动（没地板/超时）就跳过怪物突围

（第 6 点"定位丢失左右走动"已按实机反馈移除：角色会在一直定位不到的地方
来回走，反而更糟。）
"""
import unittest

from src.decision import context as ctxmod
from src.decision.context import (
    Context, DecisionEngine, State, STUCK_FRAMES,
)
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config

from .fake_executor import FakeExecutor

GROUND = 600


def monster(cx, foot=GROUND, w=60, h=60):
    return Detection(cls_name="monster", confidence=0.9,
                     x=cx - w // 2, y=foot - h, w=w, h=h)


def floor(x, w=500, y=GROUND, h=20):
    return Detection(cls_name="floor", confidence=1.0, x=x, y=y, w=w, h=h)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Base(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self._orig = ctxmod.time
        ctxmod.time = self.clock
        self.addCleanup(self._restore)
        self.logs = []

    def _restore(self):
        ctxmod.time = self._orig

    def engine(self, **overrides):
        cfg = Config(overrides={
            "attack_type": "long", "attack_range": 400, "attack_range_y": 60,
            "attack_min_range": 0, "stand_mode": False, "pixel_base_height": 768,
            "jump_key": "alt", "ghost_casts": 6, "fps": 20,
            "skills": [{"name": "攻击", "key": "a", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = 1.0
        return e

    def logged(self, text):
        return any(text in m for m in self.logs)


class ChasePatienceTests(Base):
    def ctx(self, *monsters, sx=1000):
        return Context(monsters=list(monsters), self_position=(sx, GROUND),
                       frame_width=1920, frame_height=1080)

    def test_monster_nearby_only_counts_same_level_within_reach(self):
        e = self.engine()
        c = self.ctx(monster(1440))          # 440px：攻击距离 400 + 50 以内
        self.assertTrue(e._monster_nearby(c, 1000, GROUND))
        c = self.ctx(monster(1500), monster(1200, foot=GROUND - 200))
        self.assertFalse(e._monster_nearby(c, 1000, GROUND))

    def test_waits_after_a_cast_when_a_monster_is_close_by(self):
        e = self.engine()
        e._last_cast_at = self.clock.t
        e._target_monster = monster(1440)
        e._fsm.transition(State.CHASING)
        e._handle_monsters(self.ctx(monster(1440)))
        self.assertEqual(e.executor.held, set())   # 原地等，不追
        self.assertNotIn("a", e.executor.keys())    # 也不出手

    def test_walks_to_far_monster_right_after_a_cast_when_nothing_is_near(self):
        e = self.engine()
        e._last_cast_at = self.clock.t
        e._handle_monsters(self.ctx(monster(1800)))
        self.assertIn("right", e.executor.held)

    def test_gives_up_after_two_chase_stucks(self):
        e = self.engine()
        m = monster(1800)
        e._target_monster = m
        e._fsm.transition(State.CHASING)
        ctx = self.ctx(m)

        e._stuck_counter = STUCK_FRAMES
        e._chase(ctx, m)                       # 第一次：跳一下，目标保留
        self.assertIn("alt", e.executor.keys())
        self.assertIs(e._target_monster, m)
        self.assertFalse(e._in_ghost_zone(m))

        e._stuck_counter = STUCK_FRAMES
        e._chase(ctx, m)                       # 第二次：放弃并记残影
        self.assertIsNone(e._target_monster)
        self.assertTrue(e._in_ghost_zone(m))
        self.assertTrue(self.logged("追击卡住"))


class RetreatTests(Base):
    def test_threat_is_nearest_same_level_monster(self):
        e = self.engine()
        far, near = monster(1500), monster(1050)
        self.assertIs(e._nearest_threat(self.ctx([far, near]), far), near)

    def ctx(self, monsters, floors=(), sx=1000):
        return Context(monsters=list(monsters), floors=list(floors),
                       self_position=(sx, GROUND), frame_width=1920, frame_height=1080)

    def test_retreats_from_nearest_threat_not_locked_target(self):
        e = self.engine(attack_min_range=100)
        far, near = monster(1500), monster(1050)
        e._target_monster = far
        e._fsm.transition(State.ATTACKING)
        e._handle_monsters(self.ctx([far, near]))
        self.assertIn("left", e.executor.held)   # 躲的是贴脸的 near（在右边 → 往左退）
        self.assertTrue(e._retreating)

    def test_escape_jumps_over_monster_when_no_floor_behind(self):
        e = self.engine(attack_min_range=100)
        target = monster(1050)
        # 身后（x=955）没地板 → 不能往左退 → 朝怪方向跳过它
        ctx = self.ctx([target], floors=[floor(1000)])
        e._retreat(ctx, target, reason="测试")
        self.assertEqual(e._escape_dir, "right")
        self.assertIn("right", e.executor.held)
        self.assertIn("alt", e.executor.keys())
        self.assertTrue(self.logged("跳过怪物到另一侧"))

    def test_escape_ends_after_passing_the_monster(self):
        e = self.engine(attack_min_range=100)
        target = monster(1050)
        e._retreat(self.ctx([target], floors=[floor(1000)]), target, reason="测试")
        # 跳到怪物另一侧（角色 1200 > 怪 1050）且已拉开到最小射程外
        self.clock.t += 0.3
        e._retreat(self.ctx([target], floors=[floor(800, 800)], sx=1200),
                   target, direction=None, reason="测试")
        self.assertEqual(e._escape_started, 0.0)
        self.assertFalse(e.executor.held)

    def test_retreat_timeout_switches_to_escape(self):
        e = self.engine(attack_min_range=100)
        target = monster(1050)
        ctx = self.ctx([target], floors=[floor(800, 800)])  # 身后有地板，能正常退
        e._retreat(ctx, target, reason="测试")
        self.assertTrue(e._retreating)
        self.assertEqual(e._escape_started, 0.0)

        self.clock.t += 2.0                      # 超过 RETREAT_MAX_SECONDS(1.5)
        e._retreat(ctx, target, reason="测试")
        self.assertGreater(e._escape_started, 0)
        self.assertIn("alt", e.executor.keys())
        self.assertTrue(self.logged("后撤超时仍被贴脸"))


class SelfLostTests(Base):
    def test_long_loss_does_not_walk_back_and_forth(self):
        """定位长时间丢失不再左右来回走：方向不会每 0.4 秒翻一次。"""
        e = self.engine()
        e._last_self_pos = (1000, GROUND)          # 曾经定位到过
        held = []
        for _ in range(6):
            e._self_pos_stale_frames = e._frames(10)   # 远超 2.5 秒阈值
            self.clock.t += 0.5
            e.decide(Context(monsters=[monster(1200)],
                             frame_width=1920, frame_height=1080))
            held.append(tuple(e.executor.held))
        self.assertEqual(len(set(held)), 1)        # 始终同一方向，不来回切换
        self.assertFalse(self.logged("认不出自己"))


if __name__ == "__main__":
    unittest.main()
