"""目标横跳冷却 + 影子位置（漏检沿用）限一击。

2026-11-08 日志/视频实测：怪群夹住时"更近的可见怪优先"30 秒触发 14 次，
每次换目标都换边等待 + 转向 + AOE 松键；漏检沿用期间对着最后位置连放
（影子目标空放）。本文件覆盖对应的抑制措施。
"""
import unittest

from src.decision import context as ctxmod
from src.decision.context import Context, DecisionEngine, State
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
            "turn_mode": "tap", "aoe_mode": "hold", "fps": 20,
            "knockback_skill": {"name": "击退", "key": "", "cooldown": 0.3, "range": 240},
            "skills": [{"name": "连击", "key": "a", "cooldown": 0.5},
                       {"name": "爆炸箭", "key": "space", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = 1.0
        e._face_dir = "right"
        e._face_need_side = "right"
        e._face_need_frames = 99
        e._face_next_assert = self.clock.t + 1000
        return e

    def ctx(self, *monsters, sx=1000):
        return Context(monsters=list(monsters), self_position=(sx, GROUND),
                       frame_width=1920, frame_height=1080)

    def logged(self, text):
        return any(text in m for m in self.logs)


class SwitchCooldownTests(Base):
    def test_no_steal_within_cooldown(self):
        """刚换过目标 → 冷却期内反侧更近的怪也不抢。"""
        e = self.engine()
        e._last_target_switch_at = self.clock.t   # 刚锁定的
        locked = monster(1300)
        behind = monster(900)                      # 反侧且远
        near_behind = monster(950)                 # 反侧近
        e._target_monster = locked
        out = None
        for _ in range(3):
            out = e._pick_nearer_visible(self.ctx(locked, near_behind), locked)
        self.assertIsNone(out)
        self.assertEqual(e._target_monster.center[0], 1300)

    def test_steal_allowed_after_cooldown(self):
        """冷却过后 → 正常按抢占规则换目标。"""
        e = self.engine()
        e._last_target_switch_at = self.clock.t
        locked = monster(1300)
        e._target_monster = locked
        front = monster(1400)
        self.clock.t += 1.6                        # 超过 TARGET_SWITCH_MIN_GAP_SECONDS
        out = None
        for _ in range(3):
            out = e._pick_nearer_visible(self.ctx(locked, front), locked)
        # 同侧更近？front(400px) 比 locked(300px) 远 → 不换
        self.assertIsNone(out)
        closer = monster(1150)                     # 同侧 150px
        for _ in range(3):
            out = e._pick_nearer_visible(self.ctx(locked, closer), locked)
        self.assertIsNotNone(out)
        self.assertEqual(out.center[0], 1150)

    def test_real_switch_records_time(self):
        """_set_locked_target 真正换目标时记录时间（冷却的起点）。"""
        e = self.engine()
        e._set_locked_target(monster(1300))
        self.assertEqual(e._last_target_switch_at, self.clock.t)
        self.clock.t += 2.0
        e._set_locked_target(monster(1305))        # 同一只（抖动）：不重置
        self.assertEqual(e._last_target_switch_at, self.clock.t - 2.0)
        e._set_locked_target(monster(1600))        # 真换：重置
        self.assertEqual(e._last_target_switch_at, self.clock.t)


class RetainSuppressTests(Base):
    def test_single_cast_per_retain_segment(self):
        """漏检沿用段：第一击放行，之后停手直到目标重新可见。"""
        e = self.engine()
        m = monster(1300)
        e._target_monster = m
        e._target_miss_frames = 2                  # 正处于沿用段
        e._retain_casted = False
        e._attack(self.ctx(m), m)                  # 第一击
        self.assertIn("a", e.executor.keys())
        self.assertTrue(e._retain_casted)
        n = len(e.executor.keys())
        self.clock.t += 0.5
        e._attack(self.ctx(m), m)                  # 仍看不见目标：停手
        self.assertEqual(len(e.executor.keys()), n)
        self.assertFalse(self.logged("[AOE]"))

    def test_flag_resets_on_new_segment(self):
        """新沿用段（miss_frames==1）重置标记：再给一击机会。"""
        e = self.engine()
        m = monster(1300)
        e._target_monster = m
        e._fsm.transition(State.ATTACKING)         # _in_combat 要求战斗状态
        e._target_miss_frames = 2
        e._retain_casted = True
        # 目标重新可见 → miss_frames 清 0；再次漏检 → 新段重置标记
        e._target_miss_frames = 0
        e._retain_missed_target(self.ctx(m))
        self.assertEqual(e._target_miss_frames, 1)
        self.assertFalse(e._retain_casted)

    def test_no_aoe_hold_on_phantom(self):
        """漏检沿用期间：不对影子位置发起 2 秒按住。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1450)
        e._target_monster = m1
        e._target_miss_frames = 2
        e._retain_casted = False
        e._attack(self.ctx(m1, m2), m1)
        self.assertIsNone(e._skill_hold_key)       # 没按住
        self.assertIn("a", e.executor.keys())      # 只放了一击技能1
        self.assertTrue(e._retain_casted)

    def test_casts_resume_when_visible_again(self):
        """目标重新可见（miss_frames=0）→ 恢复正常攻击（AOE hold 可发起）。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1450)
        e._target_monster = m1
        e._target_miss_frames = 0
        e._retain_casted = True                    # 上一段打过：可见后应解除
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")


if __name__ == "__main__":
    unittest.main()
