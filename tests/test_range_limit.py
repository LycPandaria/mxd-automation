"""活动范围：只在地图 start%~end% 区间内活动（0~100 = 不限制）。

镜头跟着角色走（角色在画面里的 x 不变，地图在动），所以用一个假世界：
小地图黄点 = 世界 x / 比例。只测范围本身的过滤 / 走回 / 未知位置处理，
不模拟完整打怪（本地转向有"生效自检"，需要真实的位移反馈，另由实机验证）。
"""
import unittest

from src.decision import context as ctxmod
from src.decision.context import Context, DecisionEngine, State
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config, _defaults, _migrate_legacy

from .fake_executor import FakeExecutor

GROUND = 600
FRAME_W, FRAME_H = 1920, 1080
SCREEN_X = FRAME_W // 2       # 镜头跟着走：角色一直在画面中间
SCALE = 16                    # 小地图 1px = 画面 16px
MAP_W = 3200                  # 地图宽（画面 px）→ 小地图宽 200px
MINIMAP_W = MAP_W // SCALE
LINE = MAP_W // 2             # 地图中点（世界 x）


def pct(p):
    """地图百分比 → 世界 x。"""
    return MAP_W * p / 100.0


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class World:
    """假世界：角色世界 x + 小地图黄点，按世界坐标生成画面里的怪。"""

    def __init__(self, x, monsters=()):
        self.x = float(x)
        self.monsters = list(monsters)
        self.dot_visible = True

    def ctx(self):
        monsters = []
        for wx in self.monsters:
            sx = int(SCREEN_X + wx - self.x)
            if 0 <= sx <= FRAME_W:
                monsters.append(Detection(cls_name="monster", confidence=0.9,
                                          x=sx - 30, y=GROUND - 60, w=60, h=60))
        return Context(monsters=monsters, self_position=(SCREEN_X, GROUND),
                       frame_height=FRAME_H, frame_width=FRAME_W,
                       minimap_x=self.x / SCALE if self.dot_visible else None,
                       minimap_width=MINIMAP_W)


class RangeTestCase(unittest.TestCase):
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
            "attack_type": "long", "attack_range": 300, "attack_range_y": 60,
            "turn_mode": "tap", "aoe_mode": "burst",
            "stand_mode": False, "pixel_base_height": 768,
            "skills": [{"name": "攻击", "key": "a", "cooldown": 0.6}],
            "hp_threshold": 0, "mp_threshold": 0,
            # 默认 0~50 = 老的"只在地图左半边"，方便逐条对比旧行为
            "range_start": 0.0, "range_end": 50.0,
            "minimap_region": [0.0, 0.05, 0.1, 0.1], "minimap_scale": SCALE,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = FRAME_H / float(cfg.pixel_base_height)
        return e

    def logged(self, text):
        return any(text in m for m in self.logs)


class FilterTests(RangeTestCase):
    def test_full_range_keeps_every_monster(self):
        w = World(LINE - 100, [LINE + 100])
        e = self.engine(range_start=0, range_end=100)
        c = w.ctx()
        self.assertFalse(e._range_step(c))
        self.assertEqual(len(c.monsters), 1)
        self.assertIsNone(e._range_start_x)
        self.assertIsNone(e._range_end_x)
        self.assertEqual(self.logs, [])
        self.assertFalse(e.range_enabled())

    def test_config_defaults_to_no_limit(self):
        self.assertEqual(_defaults()["range_start"], 0.0)
        self.assertEqual(_defaults()["range_end"], 100.0)

    def test_left_half_drops_monsters_past_the_mid(self):
        w = World(LINE - 300, [LINE - 100, LINE + 200])
        e = self.engine()                       # 0~50
        c = w.ctx()
        e._range_step(c)
        self.assertEqual([m.center[0] for m in c.monsters], [SCREEN_X + 200])
        self.assertAlmostEqual(e._range_end_x, SCREEN_X + 300, delta=1)

    def test_right_half_is_the_mirror(self):
        w = World(LINE + 300, [LINE + 100, LINE - 200])
        e = self.engine(range_start=50, range_end=100)
        c = w.ctx()
        e._range_step(c)
        self.assertEqual([m.center[0] for m in c.monsters], [SCREEN_X - 200])
        self.assertAlmostEqual(e._range_start_x, SCREEN_X - 300, delta=1)

    def test_middle_band_drops_both_sides(self):
        w = World(pct(50), [pct(5), pct(30), pct(70), pct(95)])
        e = self.engine(range_start=10, range_end=90)
        c = w.ctx()
        e._range_step(c)
        self.assertEqual([m.center[0] for m in c.monsters],
                         [SCREEN_X + (pct(30) - pct(50)),
                          SCREEN_X + (pct(70) - pct(50))])

    def test_percentages_move_the_boundaries(self):
        w = World(pct(25), [pct(25)])
        e = self.engine(range_start=25, range_end=100)
        e._range_step(w.ctx())
        self.assertAlmostEqual(e._range_start_x, SCREEN_X, delta=1)
        e = self.engine(range_start=0, range_end=25)
        e._range_step(w.ctx())
        self.assertAlmostEqual(e._range_end_x, SCREEN_X, delta=1)

    def test_monster_near_the_boundary_still_counts(self):
        w = World(LINE - 300, [LINE + 10])
        e = self.engine()
        c = w.ctx()
        e._range_step(c)
        self.assertEqual(len(c.monsters), 1)

    def test_locked_monster_walking_out_is_released_at_once(self):
        w = World(LINE - 300, [LINE - 5])
        e = self.engine()
        c = w.ctx()
        e._range_step(c)
        e._target_monster = c.monsters[0]
        w.monsters = [LINE + 40]  # 走到范围外（还认得出是同一只）
        e._range_step(w.ctx())
        self.assertIsNone(e._target_monster)
        self.assertTrue(self.logged("锁定的怪走到范围外"))

    def test_reversed_config_is_ordered_not_ignored(self):
        w = World(pct(50), [pct(5), pct(95)])
        e = self.engine(range_start=90, range_end=10)  # 配反了
        c = w.ctx()
        e._range_step(c)
        self.assertEqual([m.center[0] for m in c.monsters], [])
        self.assertTrue(e.range_enabled())


class WalkBackTests(RangeTestCase):
    def test_knocked_past_the_end_walks_back_then_keeps_fighting(self):
        w = World(LINE + 300, [LINE - 100])
        e = self.engine()
        ex = e.executor
        # 前两帧只是确认出界（松键不打），第三帧开始往左走
        for _ in range(3):
            self.clock.t += 0.08
            self.assertTrue(e._range_step(w.ctx()))
        self.assertIn("left", ex.held)
        # 走回范围内
        w.x = LINE - 100
        self.clock.t += 0.08
        self.assertFalse(e._range_step(w.ctx()))
        self.assertFalse(ex.held)
        self.assertTrue(self.logged("回到范围内，继续打怪"))

    def test_knocked_past_the_start_walks_back_right(self):
        w = World(pct(5), [pct(30)])
        e = self.engine(range_start=10, range_end=90)
        ex = e.executor
        for _ in range(3):
            self.clock.t += 0.08
            self.assertTrue(e._range_step(w.ctx()))
        self.assertIn("right", ex.held)
        w.x = pct(20)
        self.clock.t += 0.08
        self.assertFalse(e._range_step(w.ctx()))
        self.assertFalse(ex.held)
        self.assertTrue(self.logged("回到范围内，继续打怪"))

    def test_small_overshoot_does_not_trigger_a_walk_back(self):
        w = World(LINE + 30, [LINE - 100])  # 出界 30px < RANGE_RETURN_START(60)
        e = self.engine()
        self.assertFalse(e._range_step(w.ctx()))
        self.assertFalse(e.executor.held)

    def test_stand_mode_filters_but_never_walks(self):
        w = World(pct(5), [pct(30), pct(95)])
        e = self.engine(stand_mode=True, range_start=10, range_end=90)
        c = w.ctx()
        self.assertFalse(e._range_step(c))
        self.assertEqual(len(c.monsters), 1)
        self.assertFalse(e.executor.held)

    def test_no_minimap_stands_still_and_hits_only_what_is_in_reach(self):
        w = World(LINE - 100, [LINE - 50, LINE - 2000])
        w.dot_visible = False
        e = self.engine()
        c = w.ctx()
        self.assertFalse(e._range_step(c))
        # 只留够得着的：LINE-50 在画面里 50px，LINE-2000 已经出画面
        self.assertEqual(len(c.monsters), 1)
        self.assertFalse(e.executor.held)
        self.assertTrue(self.logged("还没框选小地图") or self.logged("找不到自己的黄点"))

    def test_blocked_walk_back_gives_up_and_retries(self):
        w = World(LINE + 300, [LINE - 100])
        e = self.engine()
        for _ in range(3):
            self.clock.t += 0.08
            e._range_step(w.ctx())
        # 一直走不动：每次"卡住"计一次，3 次后先放弃
        for _ in range(3):
            e._stuck_counter = 999
            self.clock.t += 0.08
            e._range_step(w.ctx())
        self.assertTrue(self.logged("走不回去"))
        self.assertFalse(e.executor.held)
        self.assertEqual(e._fsm.current, State.CHASING)

    def test_at_range_line_detects_both_boundaries(self):
        e = self.engine(range_start=10, range_end=90)
        w = World(pct(90))
        e._range_step(w.ctx())
        self.assertTrue(e._at_range_line(w.ctx(), "right"))
        self.assertFalse(e._at_range_line(w.ctx(), "left"))
        w = World(pct(10))
        e._range_step(w.ctx())
        self.assertTrue(e._at_range_line(w.ctx(), "left"))
        self.assertFalse(e._at_range_line(w.ctx(), "right"))


class LegacyMigrationTests(unittest.TestCase):
    """旧配置 range_limit + range_split → range_start/range_end。"""

    def test_left(self):
        d = _migrate_legacy({"range_limit": "left", "range_split": 0.8})
        self.assertEqual((d["range_start"], d["range_end"]), (0.0, 80.0))
        self.assertNotIn("range_limit", d)
        self.assertNotIn("range_split", d)

    def test_right(self):
        d = _migrate_legacy({"range_limit": "right", "range_split": 0.6})
        self.assertEqual((d["range_start"], d["range_end"]), (60.0, 100.0))

    def test_none(self):
        d = _migrate_legacy({"range_limit": "none", "range_split": 0.5})
        self.assertEqual((d["range_start"], d["range_end"]), (0.0, 100.0))

    def test_new_keys_win(self):
        d = _migrate_legacy({"range_limit": "left", "range_start": 10, "range_end": 20})
        self.assertEqual((d["range_start"], d["range_end"]), (10, 20))

    def test_no_legacy_keys_untouched(self):
        d = _migrate_legacy({"range_start": 10, "range_end": 90})
        self.assertEqual((d["range_start"], d["range_end"]), (10, 90))


if __name__ == "__main__":
    unittest.main()
