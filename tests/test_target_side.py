"""选怪优先级：同侧射程内 > 反侧最近 > 同侧射程外。

规则（长手 + 朝向已知）：
  ① 面朝侧且在攻击距离内 → 一定打它，反侧再近也不换；
  ② 没有同侧射程内的怪 → 反侧最近的一只；
  ③ 两边都没有射程内的 → 同侧最近的一只，不无故转身。
反侧（背后）的近身怪不参与威胁/击退：不转身去处理。
"""
import unittest

from src.decision.context import Context, DecisionEngine, State
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config

from .fake_executor import FakeExecutor

GROUND = 600


def monster(cx, foot=GROUND, w=60, h=60):
    return Detection(cls_name="monster", confidence=0.9,
                     x=cx - w // 2, y=foot - h, w=w, h=h)


class Base(unittest.TestCase):
    def engine(self, **overrides):
        cfg = Config(overrides={
            "attack_type": "long", "attack_range": 300, "attack_range_y": 60,
            "turn_mode": "tap", "aoe_mode": "burst",
            "attack_min_range": 0, "stand_mode": False, "pixel_base_height": 768,
            "jump_key": "alt", "ghost_casts": 6, "fps": 20,
            "knockback_skill": {"name": "击退", "key": "D", "cooldown": 0.3, "range": 240},
            "skills": [{"name": "攻击", "key": "a", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=lambda m: None)
        e._scale = 1.0
        return e

    def ctx(self, *monsters, sx=1000):
        return Context(monsters=list(monsters), self_position=(sx, GROUND),
                       frame_width=1920, frame_height=1080)


class PickTargetTests(Base):
    def test_same_side_in_range_wins_even_if_cross_side_closer(self):
        e = self.engine()
        e._face_dir = "right"
        near_behind = monster(950)       # 反侧 50px（很近）
        same_front = monster(1200)       # 同侧 200px（射程内）
        target = e._pick_best_target(self.ctx(near_behind, same_front))
        self.assertEqual(target.center[0], 1200)

    def test_cross_side_nearest_when_no_same_side_in_range(self):
        e = self.engine()
        e._face_dir = "right"
        same_far = monster(1400)         # 同侧 400px（射程外）
        behind = monster(900)            # 反侧 100px
        target = e._pick_best_target(self.ctx(same_far, behind))
        self.assertEqual(target.center[0], 900)

    def test_same_side_out_of_range_when_no_cross_side(self):
        e = self.engine()
        e._face_dir = "right"
        target = e._pick_best_target(self.ctx(monster(1400), monster(1600)))
        self.assertEqual(target.center[0], 1400)

    def test_unknown_facing_falls_back_to_nearest(self):
        e = self.engine()
        e._face_dir = None
        target = e._pick_best_target(self.ctx(monster(1200), monster(1450)))
        self.assertEqual(target.center[0], 1200)

    def test_short_attack_type_ignores_side(self):
        e = self.engine(attack_type="short")
        e._face_dir = "right"
        target = e._pick_best_target(self.ctx(monster(900), monster(1300)))
        self.assertEqual(target.center[0], 900)   # 近战不按侧优先


class RetainTargetTests(Base):
    """锁定目标已存在时，_pick_nearer_visible 的抢占规则。"""

    def pick_nearer(self, e, ctx, locked, frames=3):
        out = None
        for _ in range(frames):
            out = e._pick_nearer_visible(ctx, locked)
        return out

    def test_cross_side_never_steals_from_same_side_in_range(self):
        e = self.engine()
        e._face_dir = "right"
        locked = monster(1200)           # 同侧射程内（200px）
        behind = monster(950)            # 反侧 50px
        ctx = self.ctx(locked, behind)
        self.assertIsNone(self.pick_nearer(e, ctx, locked))

    def test_same_side_in_range_steals_when_locked_is_behind(self):
        e = self.engine()
        e._face_dir = "right"
        locked = monster(800)            # 反侧（背后）
        front = monster(1150)            # 同侧射程内
        ctx = self.ctx(locked, front)
        target = self.pick_nearer(e, ctx, locked)
        self.assertIsNotNone(target)
        self.assertEqual(target.center[0], 1150)

    def test_cross_side_in_range_steals_from_same_side_out_of_range(self):
        e = self.engine()
        e._face_dir = "right"
        locked = monster(1400)           # 同侧但射程外
        behind = monster(900)            # 反侧且射程内
        ctx = self.ctx(locked, behind)
        target = self.pick_nearer(e, ctx, locked)
        self.assertIsNotNone(target)
        self.assertEqual(target.center[0], 900)


class ThreatTests(Base):
    def test_nearest_threat_ignores_opposite_side(self):
        e = self.engine()
        e._face_dir = "right"
        behind = monster(950)            # 背后 50px
        front = monster(1250)            # 同侧 250px
        threat = e._nearest_threat(self.ctx(behind, front), front)
        self.assertIsNotNone(threat)
        self.assertEqual(threat.center[0], 1250)

    def test_no_same_side_monster_means_no_threat(self):
        e = self.engine()
        e._face_dir = "right"
        behind = monster(950)
        self.assertIsNone(e._nearest_threat(self.ctx(behind), None))

    def test_knockback_threat_scan_ignores_opposite_side(self):
        e = self.engine()
        e._face_dir = "right"
        behind = monster(980)            # 背后 20px，在击退距离内
        ctx = self.ctx(behind)
        self.assertIsNone(e._nearest_knockback_threat(ctx, (1000, GROUND)))
        self.assertFalse(e._try_knockback(behind, 1000))

    def test_knockback_still_works_on_same_side(self):
        e = self.engine()
        e._face_dir = "right"
        front = monster(1100)
        ctx = self.ctx(front)
        self.assertIsNotNone(e._nearest_knockback_threat(ctx, (1000, GROUND)))
        self.assertTrue(e._try_knockback(front, 1000))


if __name__ == "__main__":
    unittest.main()
