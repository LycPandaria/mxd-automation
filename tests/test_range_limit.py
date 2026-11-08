"""活动范围：只在地图左半边/右半边活动（移植自分享包 tests/test_range_limit.py）。

镜头跟着角色走（角色在画面里的 x 不变，地图在动），所以用一个假世界：
小地图黄点 = 世界 x / 比例。只测范围本身的过滤 / 走回 / 未知位置处理，
不模拟完整打怪（本地转向有"生效自检"，需要真实的位移反馈，另由实机验证）。
"""
import unittest

from src.decision import context as ctxmod
from src.decision.context import Context, DecisionEngine, State
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config, _defaults

from .fake_executor import FakeExecutor

GROUND = 600
FRAME_W, FRAME_H = 1920, 1080
SCREEN_X = FRAME_W // 2       # 镜头跟着走：角色一直在画面中间
SCALE = 16                    # 小地图 1px = 画面 16px
MAP_W = 3200                  # 地图宽（画面 px）→ 小地图宽 200px
MINIMAP_W = MAP_W // SCALE
LINE = MAP_W // 2             # 分界线（世界 x）


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
            "range_limit": "left", "range_split": 0.5,
            "minimap_region": [0.0, 0.05, 0.1, 0.1], "minimap_scale": SCALE,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = FRAME_H / float(cfg.pixel_base_height)
        return e

    def logged(self, text):
        return any(text in m for m in self.logs)


class FilterTests(RangeTestCase):
    def test_default_is_no_limit_and_keeps_every_monster(self):
        w = World(LINE - 100, [LINE + 100])
        e = self.engine(range_limit="none")
        c = w.ctx()
        self.assertFalse(e._range_step(c))
        self.assertEqual(len(c.monsters), 1)
        self.assertIsNone(e._range_line_x)
        self.assertEqual(self.logs, [])

    def test_config_defaults_to_no_limit(self):
        self.assertEqual(_defaults()["range_limit"], "none")
        self.assertEqual(_defaults()["range_split"], 0.5)

    def test_left_limit_drops_monsters_past_the_line(self):
        w = World(LINE - 300, [LINE - 100, LINE + 200])
        e = self.engine()
        c = w.ctx()
        e._range_step(c)
        self.assertEqual([m.center[0] for m in c.monsters], [SCREEN_X + 200])
        self.assertEqual(e._range_line_x, SCREEN_X + 300)

    def test_right_limit_is_the_mirror(self):
        w = World(LINE + 300, [LINE + 100, LINE - 200])
        e = self.engine(range_limit="right")
        c = w.ctx()
        e._range_step(c)
        self.assertEqual([m.center[0] for m in c.monsters], [SCREEN_X - 200])
        self.assertEqual(e._range_line_x, SCREEN_X - 300)

    def test_split_moves_the_line(self):
        w = World(MAP_W // 4)
        e = self.engine(range_split=0.25)
        c = w.ctx()
        e._range_step(c)
        self.assertEqual(e._range_line_x, SCREEN_X)

    def test_monster_on_the_line_still_counts(self):
        w = World(LINE - 300, [LINE + 10])
        e = self.engine()
        c = w.ctx()
        e._range_step(c)
        self.assertEqual(len(c.monsters), 1)

    def test_locked_monster_walking_past_the_line_is_released_at_once(self):
        w = World(LINE - 300, [LINE - 5])
        e = self.engine()
        c = w.ctx()
        e._range_step(c)
        e._target_monster = c.monsters[0]
        w.monsters = [LINE + 40]  # 走到线外（还认得出是同一只）
        e._range_step(w.ctx())
        self.assertIsNone(e._target_monster)
        self.assertTrue(self.logged("锁定的怪走到分界线外"))


class WalkBackTests(RangeTestCase):
    def test_knocked_past_the_line_walks_back_then_keeps_fighting(self):
        w = World(LINE + 300, [LINE - 100])
        e = self.engine()
        ex = e.executor
        # 前两帧只是确认过线（松键不打），第三帧开始往左走
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

    def test_small_overshoot_does_not_trigger_a_walk_back(self):
        w = World(LINE + 30, [LINE - 100])  # 过线 30px < RANGE_RETURN_START(60)
        e = self.engine()
        self.assertFalse(e._range_step(w.ctx()))
        self.assertFalse(e.executor.held)

    def test_stand_mode_filters_but_never_walks(self):
        w = World(LINE + 300, [LINE - 100, LINE + 400])
        e = self.engine(stand_mode=True)
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


if __name__ == "__main__":
    unittest.main()
