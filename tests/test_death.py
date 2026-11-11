"""角色死亡判定：HP 连续为 0 且自身坐标几乎不动超过 60 秒 → 停机。

背景：角色死亡后游戏会忽略所有按键，程序只看画面会继续加血/放技能，
既打不到怪也是明显的脚本特征（2026-11-09 17:18:19 死亡后原地按了 10 分钟）。
"""
import unittest

from src.decision import context as ctxmod
from src.decision.context import (
    Context, DEATH_STOP_SECONDS, DecisionEngine,
)

from src.utils.config_loader import Config

from .fake_executor import FakeExecutor

GROUND = 600
POS = (700, GROUND)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class DeathTestCase(unittest.TestCase):
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
            "stand_mode": False, "pixel_base_height": 768,
            "skills": [{"name": "攻击", "key": "a", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = 1.0
        return e

    def step(self, e, hp=0.0, pos=POS):
        e.decide(Context(self_position=pos, hp_ratio=hp))


class DeathStopTests(DeathTestCase):
    def test_dead_after_60s_still(self):
        e = self.engine()
        self.step(e)                       # 开始计时
        self.clock.t += DEATH_STOP_SECONDS - 1
        self.step(e)
        self.assertFalse(e.dead, "不到 60 秒不该判死")
        self.clock.t += 2
        self.step(e)
        self.assertTrue(e.dead)
        self.assertTrue(any("角色死亡" in m for m in self.logs))

    def test_yolo_jitter_does_not_reset_timer(self):
        """玩家框的 20~40px 抖动不能被当成"在动"。"""
        e = self.engine()
        for i in range(8):
            self.step(e, pos=(POS[0] + (i % 3) * 12, POS[1] + (i % 2) * 8))
            self.clock.t += 9
        self.step(e)
        self.assertTrue(e.dead)

    def test_moving_character_is_not_dead(self):
        e = self.engine()
        x = POS[0]
        for _ in range(10):
            self.step(e, pos=(x, POS[1]))
            self.clock.t += 10
            x += 100                      # 每次都有显著位移 → 计时重置
        self.assertFalse(e.dead)

    def test_hp_recover_resets(self):
        e = self.engine()
        self.step(e)
        self.clock.t += 50
        self.step(e)
        self.step(e, hp=0.5)              # 血回来了 → 清零
        self.step(e)
        self.clock.t += DEATH_STOP_SECONDS - 1
        self.step(e)
        self.assertFalse(e.dead)

    def test_hp_unknown_keeps_timer(self):
        """血量读数丢失的帧不推进也不打断计时。"""
        e = self.engine()
        self.step(e)
        self.clock.t += 30
        self.step(e)
        self.clock.t += 10
        self.step(e, hp=None)
        self.clock.t += DEATH_STOP_SECONDS - 40 - 1
        self.step(e)
        self.assertFalse(e.dead)
        self.clock.t += 2
        self.step(e)
        self.assertTrue(e.dead)

    def test_dead_engine_stops_pressing(self):
        e = self.engine()
        self.step(e)
        self.clock.t += DEATH_STOP_SECONDS + 1
        self.step(e)
        self.assertTrue(e.dead)

        e.executor.clear()
        n_logs = len(self.logs)
        for _ in range(10):
            self.clock.t += 1
            self.step(e)
        self.assertEqual(e.executor.pressed, [], "判定死亡后不应再按任何键")
        self.assertFalse(
            any("角色死亡" in m for m in self.logs[n_logs:]),
            "死亡日志只打一次",
        )


if __name__ == "__main__":
    unittest.main()
