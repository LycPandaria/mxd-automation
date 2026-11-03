"""残影判定：出手次数 + 静止判定 + 位置记忆。

对应分享包 context.py 的 GHOST_* 逻辑，替换本地原来"连续攻击帧数 90 帧"的实现。
"""
import unittest

from src.decision import context as ctxmod
from src.decision.context import Context, DecisionEngine, State, GHOST_ZONE_SECONDS
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config

from .fake_executor import FakeExecutor

GROUND = 600


def monster(cx, foot=GROUND, w=60, h=60):
    return Detection(cls_name="monster", confidence=0.9,
                     x=cx - w // 2, y=foot - h, w=w, h=h)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class GhostTestCase(unittest.TestCase):
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
            "stand_mode": False, "pixel_base_height": 768, "ghost_casts": 6,
            "skills": [{"name": "攻击", "key": "a", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = 1.0
        return e

    def logged(self, text):
        return any(text in m for m in self.logs)


class GhostZoneTests(GhostTestCase):
    def test_first_mark_is_8s_then_doubles(self):
        e = self.engine()
        m = monster(1200)
        s1, c1 = e._mark_ghost(m)
        self.assertEqual((s1, c1), (GHOST_ZONE_SECONDS, 1))
        s2, c2 = e._mark_ghost(m)
        self.assertEqual((s2, c2), (GHOST_ZONE_SECONDS * 2, 2))
        for _ in range(10):
            s, c = e._mark_ghost(m)
        self.assertEqual(s, 300.0)  # 上限 GHOST_ZONE_MAX_SECONDS
        self.assertEqual(c, 12)

    def test_zone_radius_and_expiry(self):
        e = self.engine()
        m = monster(1200)
        e._mark_ghost(m)
        self.assertTrue(e._in_ghost_zone(m))
        self.assertFalse(e._in_ghost_zone(monster(1300)))  # 100px > 半径 60
        self.clock.t += GHOST_ZONE_SECONDS + 1
        self.assertFalse(e._in_ghost_zone(m))

    def test_pick_best_target_skips_ghost_zone(self):
        e = self.engine()
        ghost = monster(1050)
        real = monster(1300)
        e._mark_ghost(ghost)
        ctx = Context(monsters=[ghost, real], self_position=(1000, GROUND))
        target = e._pick_best_target(ctx)
        self.assertIsNotNone(target)
        self.assertEqual(target.center[0], 1300)

    def test_resolve_locked_target_does_not_relock_ghost(self):
        e = self.engine()
        m = monster(1200)
        e._target_monster = m
        e._fsm.transition(State.ATTACKING)
        e._mark_ghost(m)
        ctx = Context(monsters=[m], self_position=(1000, GROUND))
        self.assertIsNone(e._resolve_locked_target(ctx))
        self.assertIsNone(e._target_monster)  # 不能走"漏检沿用最后位置"再打回去


class GhostTriggerTests(GhostTestCase):
    def test_static_ghost_marked_after_casts(self):
        e = self.engine()
        m = monster(1200)
        e._target_monster = m
        e._fsm.transition(State.ATTACKING)
        ctx = Context(monsters=[m], self_position=(1000, GROUND))

        e._handle_monsters(ctx)          # 第一帧：记下位置和时间
        self.assertFalse(e._in_ghost_zone(m))

        self.clock.t += 3.5              # 超过 GHOST_MIN_SECONDS(3s)
        e._casts_on_target = 6           # 打满 ghost_casts 次
        e._handle_monsters(ctx)

        self.assertTrue(e._in_ghost_zone(m))
        self.assertIsNone(e._target_monster)
        self.assertTrue(self.logged("疑似残影"))

    def test_moving_target_needs_double(self):
        e = self.engine()
        m = monster(1200)
        e._target_monster = m
        e._fsm.transition(State.ATTACKING)
        e._handle_monsters(Context(monsters=[m], self_position=(1000, GROUND)))

        self.clock.t += 3.5
        e._casts_on_target = 6
        moved = monster(1260)            # 动了 60px > GHOST_STATIC_PX(20)
        ctx = Context(monsters=[moved], self_position=(1000, GROUND))
        e._handle_monsters(ctx)
        self.assertFalse(e._in_ghost_zone(moved))   # 活怪：6 次不够

        self.clock.t += 3.5              # 累计 7s > 2×3s
        e._casts_on_target = 12
        e._handle_monsters(ctx)
        self.assertTrue(e._in_ghost_zone(moved))    # 12 次 + 6 秒才判"打不到"

    def test_disabled_when_zero(self):
        e = self.engine(ghost_casts=0)
        m = monster(1200)
        e._target_monster = m
        e._fsm.transition(State.ATTACKING)
        ctx = Context(monsters=[m], self_position=(1000, GROUND))
        e._handle_monsters(ctx)
        self.clock.t += 100
        e._casts_on_target = 999
        e._handle_monsters(ctx)
        self.assertFalse(e._in_ghost_zone(m))

    def test_cast_skill_counts_on_target(self):
        e = self.engine()
        self.assertEqual(e._casts_on_target, 0)
        self.assertTrue(e._cast_skill(force_index=0))
        self.assertEqual(e._casts_on_target, 1)

    def test_switching_target_resets_count(self):
        e = self.engine()
        old, new = monster(1200), monster(1400)
        e._target_monster = old
        e._fsm.transition(State.ATTACKING)
        e._casts_on_target = 5
        e._target_miss_frames = 999  # 漏检保持已到期，不再沿用旧目标
        ctx = Context(monsters=[new], self_position=(1000, GROUND))
        e._handle_monsters(ctx)
        self.assertEqual(e._target_monster.center[0], 1400)
        self.assertEqual(e._casts_on_target, 0)


if __name__ == "__main__":
    unittest.main()
