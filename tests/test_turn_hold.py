"""按住转向（turn_mode="hold"）：按住方向键走出确认位移再松键出手。

对比点按模式（tap）："一次确定性按住"代替"点按→自检→重试"循环，
消除小碎步。默认仍是 tap，本文件只测 hold。
"""
import time
import unittest

from src.decision import context as ctxmod
from src.decision.context import (
    Context, DecisionEngine, State,
    TURN_HOLD_CONFIRM_PX, TURN_HOLD_MAX_SECONDS, TURN_HOLD_MIN_SECONDS,
)
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config

from .fake_executor import FakeExecutor

GROUND = 600


def monster(cx, foot=GROUND, w=60, h=60):
    return Detection(cls_name="monster", confidence=0.9,
                     x=cx - w // 2, y=foot - h, w=w, h=h)


def floor(x, w=600, y=GROUND, h=20):
    return Detection(cls_name="floor", confidence=1.0, x=x, y=y, w=w, h=h)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class HoldTestCase(unittest.TestCase):
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
            "turn_mode": "tap", "aoe_mode": "burst",
            "attack_min_range": 0, "stand_mode": False, "pixel_base_height": 768,
            "turn_mode": "hold", "fps": 20,
            "knockback_skill": {"name": "击退", "key": "D", "cooldown": 0.3, "range": 240},
            "skills": [{"name": "攻击", "key": "a", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = 1.0
        return e

    def primed(self, e, side):
        """预置换边防抖：目标侧已稳定多帧（真实运行里每帧都在累计）。"""
        e._face_need_side = side
        e._face_need_frames = 99
        return e

    def logged(self, text):
        return any(text in m for m in self.logs)


class HoldBasicTests(HoldTestCase):
    def test_default_mode_is_tap(self):
        """代码内置默认是 tap（_defaults 层面）；本地 user.yaml 可能已开 hold，
        所以运行值用显式覆盖断言，不依赖本机配置。"""
        from src.utils.config_loader import _defaults
        self.assertEqual(_defaults()["turn_mode"], "tap")
        cfg = Config(overrides={
            "attack_type": "long", "attack_range": 400, "attack_range_y": 60,
            "turn_mode": "tap",
            "skills": [{"name": "攻击", "key": "a", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
        })
        e = DecisionEngine(cfg, FakeExecutor())
        self.assertFalse(e._turn_mode_hold())

    def ctx(self, *monsters, sx=1000, floors=()):
        return Context(monsters=list(monsters), floors=list(floors),
                       self_position=(sx, GROUND),
                       frame_width=1920, frame_height=1080)

    def test_hold_starts_and_confirms_on_displacement(self):
        """发起转向 → 按住 keydown；走出确认位移 → 确认朝向并松键。"""
        e = self.engine()
        e._face_dir = "left"
        m = monster(1300)
        self.primed(e, "right")
        e._face_step(self.ctx(m), m)
        self.assertEqual(e._turn_hold_dir, "right")
        self.assertIn("right", e.executor.held)
        self.assertEqual(e._face_dir, "left")   # 未确认前保留旧朝向

        # 走出 40px（角色移到 1040）且过了最短时长 → 确认
        self.clock.t += TURN_HOLD_MIN_SECONDS + 0.05
        e._face_step(self.ctx(m, sx=1040), m)
        self.assertIsNone(e._turn_hold_dir)
        self.assertEqual(e._face_dir, "right")
        self.assertFalse(e.executor.held)       # 已松键，下一帧可以出手
        self.assertTrue(self.logged("按住转向 right 已确认"))

    def test_hold_blocks_casting_until_confirmed(self):
        """按住期间 _face_step 返回 True（不出手），确认后放行。"""
        e = self.engine()
        e._face_dir = "left"
        m = monster(1300)
        self.primed(e, "right")
        self.assertTrue(e._face_step(self.ctx(m), m))    # 发起按住，本帧不出手
        e._turn_hold_planned = 0.5
        self.clock.t += 0.1
        self.assertTrue(e._face_step(self.ctx(m, sx=1010), m))  # 位移不够，继续按住
        self.clock.t += TURN_HOLD_MIN_SECONDS
        self.assertFalse(e._face_step(self.ctx(m, sx=1040), m))  # 达标，放行施法

    def test_hold_min_duration_before_confirm(self):
        """位移够了但不足最短时长 → 再等一帧（避免一帧抖动瞬间确认）。"""
        e = self.engine()
        e._face_dir = None
        m = monster(1500)   # 离得远：不触发击退距离的提前松键
        self.primed(e, "right")
        e._face_step(self.ctx(m), m)
        start = e._turn_hold_started
        self.clock.t = start + TURN_HOLD_MIN_SECONDS - 0.05
        e._face_step(self.ctx(m, sx=1100), m)   # 位移 100px 但时间不够
        self.assertIsNotNone(e._turn_hold_dir)
        self.clock.t = start + TURN_HOLD_MIN_SECONDS + 0.01
        e._face_step(self.ctx(m, sx=1100), m)
        self.assertEqual(e._face_dir, "right")

    def test_timeout_confirms_optimistically(self):
        """被怪群挡住走不动（零位移）→ 按住到期仍乐观确认，绝不无限站着等。"""
        e = self.engine()
        e._face_dir = "left"
        m = monster(1300)
        self.primed(e, "right")
        e._face_step(self.ctx(m, floors=[floor(1000)]), m)
        self.assertEqual(e._turn_hold_dir, "right")
        e._turn_hold_planned = 0.4
        self.clock.t += 0.5
        e._face_step(self.ctx(m, sx=1010, floors=[floor(1000)]), m)  # 只挪了 10px
        self.assertIsNone(e._turn_hold_dir)
        self.assertEqual(e._face_dir, "right")  # 乐观确认：按住已横跨多个游戏帧
        self.assertFalse(e.executor.held)
        self.assertTrue(self.logged("按住到期"))


class HoldSafetyTests(HoldTestCase):
    def ctx(self, *monsters, sx=1000, floors=()):
        return Context(monsters=list(monsters), floors=list(floors),
                       self_position=(sx, GROUND),
                       frame_width=1920, frame_height=1080)

    def test_releases_at_ledge(self):
        """前方没地板 → 提前松键；按住已过最短时长 → 朝向照常确认。"""
        e = self.engine()
        e._face_dir = "left"
        m = monster(1300)
        self.primed(e, "right")
        e._face_step(self.ctx(m, floors=[floor(950, 120)]), m)  # 地板到 1070 为止
        self.clock.t += 0.3
        e._face_step(self.ctx(m, sx=1025, floors=[floor(950, 120)]), m)  # 前瞻越过崖边
        self.assertIsNone(e._turn_hold_dir)
        self.assertFalse(e.executor.held)
        self.assertEqual(e._face_dir, "right")

    def test_releases_at_frame_edge(self):
        """走到画面右缘 → 提前松键。"""
        e = self.engine()
        e._face_dir = "left"
        m = monster(1300)
        self.primed(e, "right")
        e._face_step(self.ctx(m), m)
        self.clock.t += 0.3
        # EDGE_MARGIN=80：x ≥ 1840 就算到边
        e._face_step(self.ctx(m, sx=1850), m)
        self.assertIsNone(e._turn_hold_dir)

    def test_releases_when_min_range_reached(self):
        """长手按住走向目标、进入最小射程 → 提前松键（不贴脸）。"""
        e = self.engine(attack_min_range=120)
        m = monster(1400)   # 目标在 400px 外
        self.primed(e, "right")
        e._face_step(self.ctx(m), m)
        self.clock.t += 0.3
        # 走到 1300：距离 100px < min_range 120
        e._face_step(self.ctx(m, sx=1300), m)
        self.assertIsNone(e._turn_hold_dir)
        self.assertFalse(e.executor.held)
        self.assertEqual(e._face_dir, "right")  # 按住 ≥ 最短时长 → 确认

    def test_close_monster_does_not_abort_turn(self):
        """按住方向上有近身怪 → 不中止转向（怪群里本来就要转身打它，
        2026-11-08 日志里"怪已近身提前松键"反复杀死转向 → 站着不动）。"""
        e = self.engine()
        e._face_dir = "left"
        target = monster(1300)
        close = monster(1150)   # 同侧 150px < 击退距离 240
        self.primed(e, "right")
        e._face_step(self.ctx(target, close), target)
        e._turn_hold_planned = 0.4
        self.clock.t += 0.5
        e._face_step(self.ctx(target, close, sx=1040), target)
        self.assertIsNone(e._turn_hold_dir)
        self.assertEqual(e._face_dir, "right")  # 到点乐观确认
        self.assertFalse(self.logged("怪已近身"))

    def test_target_flip_aborts_hold(self):
        """按住 right 时目标换到左边（按住不足最短时长）→ 松键、朝向不变、
        本帧不许施法（face != need）。"""
        e = self.engine()
        e._face_dir = "left"
        right = monster(1300)
        self.primed(e, "right")
        e._face_step(self.ctx(right), right)
        self.assertEqual(e._turn_hold_dir, "right")
        left = monster(800)
        self.primed(e, "left")
        ret = e._face_step(self.ctx(left), left)
        self.assertIsNone(e._turn_hold_dir)
        self.assertFalse(e.executor.held)
        self.assertEqual(e._face_dir, "left")   # 按住太短，朝向记忆不动
        self.assertFalse(ret)                    # face(left) != need(left)？need=left…
        self.assertEqual(e.executor.keys(), []) # 没出手

    def test_flip_after_min_hold_confirms_held_side(self):
        """按住 ≥ 最短时长才换边 → 角色确实转到了按住方向，照实记录。"""
        e = self.engine()
        e._face_dir = "left"
        right = monster(1300)
        self.primed(e, "right")
        e._face_step(self.ctx(right), right)
        self.clock.t += 0.35                    # 超过最短时长
        left = monster(800)
        self.primed(e, "left")
        e._face_step(self.ctx(left), left)
        self.assertIsNone(e._turn_hold_dir)
        self.assertEqual(e._face_dir, "right")  # 物理事实：确实转到了 right

    def test_stop_releases_held_key(self):
        """运行中停止：release_keys 清掉按住转向并松键。"""
        e = self.engine()
        e._face_dir = "left"
        m = monster(1300)
        self.primed(e, "right")
        e._face_step(self.ctx(m), m)
        self.assertIn("right", e.executor.held)
        e.release_keys()
        self.assertIsNone(e._turn_hold_dir)
        self.assertFalse(e.executor.held)


class HoldReassertTests(HoldTestCase):
    def ctx(self, *monsters, sx=1000, floors=()):
        return Context(monsters=list(monsters), floors=list(floors),
                       self_position=(sx, GROUND),
                       frame_width=1920, frame_height=1080)

    def test_reassert_is_a_short_hold(self):
        """到点重申 = 一次短按住，走出确认位移就松（不是点按）。"""
        e = self.engine()
        m = monster(1300)
        e._face_dir = "right"
        e._face_next_assert = self.clock.t - 1   # 到重申时间
        self.primed(e, "right")
        e._face_step(self.ctx(m), m)
        e._turn_hold_planned = 0.5
        self.clock.t += 0.4
        e._face_step(self.ctx(m, sx=1040), m)
        self.assertEqual(e._face_dir, "right")
        self.assertGreaterEqual(e._face_next_assert, self.clock.t + 7.0)  # 8~12s


if __name__ == "__main__":
    unittest.main()
