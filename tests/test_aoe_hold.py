"""AOE 按住连发（aoe_mode="hold"）：触发 → 按住技能键 → 游戏自动连发（跳过后摇）→ 松键。

对比 burst 模式（3 次点按 + 技能1兜底）：一次长按出手更密。
默认仍是 burst，本文件只测 hold；所有用例显式钉死模式，不依赖本机 user.yaml。
"""
import unittest

from src.decision import context as ctxmod
from src.decision.context import Context, DecisionEngine, State, AOE_HOLD_SECONDS
from src.perception.yolo_detector import Detection
from src.utils.config_loader import Config, _defaults

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


class AoeHoldTestCase(unittest.TestCase):
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
            # AOE/动画锁参数显式钉死：Config 会合并本机 user.yaml，
            # 不钉的话本机改配置（如 aoe_hold_cooldown: 0）会泄漏进测试
            "aoe_hold_seconds": 2.0, "aoe_hold_min_seconds": 0.5,
            "aoe_hold_cooldown": 3.0, "attack_lock_seconds": 0.6,
            "knockback_skill": {"name": "击退", "key": "", "cooldown": 0.3, "range": 240},
            "skills": [{"name": "连击", "key": "a", "cooldown": 0.5},
                       {"name": "爆炸箭", "key": "space", "cooldown": 0.5}],
            "hp_threshold": 0, "mp_threshold": 0,
            **overrides,
        })
        e = DecisionEngine(cfg, FakeExecutor(), on_log=self.logs.append)
        e._scale = 1.0
        # 朝向已确认且近期不重申，让用例聚焦"AOE 触发 → 按住 → 松键"
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

    def buff_window(self, e, frames=10):
        e.request_buff_window(frames)


class TriggerTests(AoeHoldTestCase):
    def test_default_mode_is_burst(self):
        """代码内置默认 burst（本地 user.yaml 可能已开 hold，用 _defaults 断言）。"""
        self.assertEqual(_defaults()["aoe_mode"], "burst")

    def test_two_monsters_start_hold(self):
        """攻击范围两侧合计 ≥2 只怪 → key_down 技能2，不放技能1。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")
        self.assertIn("space", e.executor.held)
        self.assertNotIn("a", e.executor.keys())

    def test_monsters_on_both_sides_start_hold(self):
        """左侧 + 右侧各省一只（都在攻击距离内）→ 合计 2 只也触发。

        AOE 从两侧都造成伤害，不再要求怪在朝向正前方。
        attack_range=400：sx=1000，左侧 600(dx=400 边界) + 右侧 1300(dx=300)。
        """
        e = self.engine()
        right, left = monster(1300), monster(600)
        e._target_monster = right
        e._attack(self.ctx(right, left), right)
        self.assertEqual(e._skill_hold_key, "space")
        self.assertIn("space", e.executor.held)
        self.assertTrue(self.logged("攻击范围两侧 2 只怪"))

    def test_monster_outside_attack_range_not_counted(self):
        """一侧在射程内、另一侧超出 attack_range → 只算 1 只，不触发 AOE。

        attack_range=400：右侧 1300(dx=300) 计入；右侧 1500(dx=500) 不计入。
        """
        e = self.engine()
        near, far = monster(1300), monster(1500)
        e._target_monster = near
        e._attack(self.ctx(near, far), near)
        self.assertIsNone(e._skill_hold_key)
        self.assertIn("a", e.executor.keys())

    def test_single_monster_keeps_burst_rotation(self):
        """只有 1 只怪 → 走原轮转（点按技能1），不按住。"""
        e = self.engine()
        m = monster(1300)
        e._target_monster = m
        e._attack(self.ctx(m), m)
        self.assertIsNone(e._skill_hold_key)
        self.assertIn("a", e.executor.keys())

    def test_hold_blocks_other_casts_while_active(self):
        """按住进行中：每帧只推进 hold，不放技能1/击退。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")
        ex = e.executor
        n_keys = len(ex.keys())
        self.clock.t += 0.3
        e._attack(self.ctx(m1, m2), m1)     # 仍在按住中：无新按键
        self.assertEqual(len(ex.keys()), n_keys)
        self.assertIn("space", ex.held)     # 技能键还按着


class ReleaseTests(AoeHoldTestCase):
    def test_full_duration_release(self):
        """按住满 2s → key_up、关间隔门（松键不再开动画锁）。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += AOE_HOLD_SECONDS + 0.05
        e._attack(self.ctx(m1, m2), m1)
        self.assertIsNone(e._skill_hold_key)
        self.assertFalse(e.executor.held)
        self.assertTrue(self.logged("连发结束"))
        # 间隔门：刚松手，下一帧不会立刻重新按住
        self.assertGreater(e._skill_hold_next_at, self.clock.t)

    def test_hold_continues_on_knockback_threat(self):
        """按住中出现近身怪 → 不松键（2026-11-10：威胁不再打断按住）。"""
        e = self.engine(knockback_skill={"name": "击退", "key": "D", "cooldown": 0.3, "range": 240})
        far, far2 = monster(1300), monster(1350)   # 都在击退距离外，先正常发起按住
        e._target_monster = far
        e._attack(self.ctx(far, far2), far)
        self.assertEqual(e._skill_hold_key, "space")
        near = monster(1150)  # 按住中途出现：击退距离 240 内
        self.clock.t += 0.3
        e._attack(self.ctx(far, far2, near), far)
        self.assertEqual(e._skill_hold_key, "space")
        self.assertFalse(self.logged("近身威胁"))

    def test_hold_continues_on_target_flip(self):
        """目标换到背后 → 不松键（2026-11-10：换边不再打断按住）。

        代价是按住方向暂时不对；松键后由 _face_step 纠正朝向。
        """
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")
        behind = monster(800)
        e._target_monster = behind
        self.clock.t += 0.3
        e._attack(self.ctx(behind), behind)        # need=left != face_dir=right
        self.assertEqual(e._skill_hold_key, "space")
        self.assertFalse(self.logged("目标换边"))

    def test_release_when_monsters_cleared(self):
        """攻击范围内怪清空且超过最短按住时长 → 提前松键。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += 0.8                        # 超过 aoe_hold_min_seconds(0.5)
        e._attack(self.ctx(), m1)                  # 怪全没了
        self.assertIsNone(e._skill_hold_key)
        self.assertTrue(self.logged("怪清空"))

    def test_hold_continues_with_monster_only_behind(self):
        """按住中前方怪没了、但背后（另一侧）射程内还有怪 → 继续按住。

        AOE 两侧都命中，另一侧的怪仍是有效目标，不能按"攻击方向清空"松键。
        """
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")
        behind = monster(800)                      # 背后 dx=200，仍在 attack_range 内
        self.clock.t += 0.8
        e._attack(self.ctx(behind), behind)
        self.assertEqual(e._skill_hold_key, "space")
        self.assertFalse(self.logged("怪清空"))

    def test_no_early_release_before_min(self):
        """怪清空但按住不足 0.5s → 继续按住（太短吃不到免后摇收益）。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += 0.2
        e._attack(self.ctx(), m1)                  # 怪没了但按住才 0.2s
        self.assertEqual(e._skill_hold_key, "space")

    def test_release_does_not_open_attack_lock(self):
        """松键不开动画锁（2026-11-10）：发起时那次锁已过期，松键后 busy 保持 False。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += 0.8                        # 发起锁(0.6s)过期、超过 min 0.5
        e._skill_hold_release("测试松键")           # 直接松键，不混入后续施法的开锁
        self.assertIsNone(e._skill_hold_key)
        self.assertFalse(e.executor.held)
        self.assertFalse(e._attack_busy(), "松键不应开动画锁")

    def test_turn_can_start_in_release_frame(self):
        """松键当帧即可发起转向（旧逻辑被松键动画锁挡住，要等 0.6s）。"""
        e = self.engine(turn_mode="hold")          # 与本机生产配置一致
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)            # 面朝 right 按住
        behind = monster(800)                      # 按住期间目标换到背后
        e._target_monster = behind
        e._face_need_side = "left"
        e._face_need_frames = 99                   # 新侧已稳定，免换边防抖
        self.clock.t += AOE_HOLD_SECONDS + 0.05
        e._attack(self.ctx(behind), behind)        # 松手 → 同帧走到转向分支
        self.assertIsNone(e._skill_hold_key)
        self.assertEqual(e._held_key, "left")      # 转向按住已发起
        self.assertTrue(self.logged("怪物在左"))


    def test_zero_gate_reholds_immediately(self):
        """aoe_hold_cooldown=0：满时长松键当帧即可重新按住（≈连续按住）。"""
        e = self.engine(aoe_hold_cooldown=0)
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += AOE_HOLD_SECONDS + 0.05
        e._attack(self.ctx(m1, m2), m1)            # 松手 + 同帧重按
        self.assertEqual(e._skill_hold_key, "space")
        self.assertIn("space", e.executor.held)
        self.assertTrue(self.logged("连发结束"))

    def test_release_for_buff_window(self):
        """buff 施法窗口被请求 → 立刻松键让路（不能等下一帧的攻击流程）。

        按住状态下技能键一直是 down，施法窗口里不松键游戏会继续自动连发，
        buff 键会被攻击动画吞掉。
        """
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")
        self.buff_window(e)
        self.assertIsNone(e._skill_hold_key)      # 请求当下就松开
        self.assertFalse(e.executor.held)
        self.assertTrue(self.logged("buff 施法窗口"))

    def test_buff_window_reports_hold_release(self):
        """request_buff_window 返回 True = 这次顺手松开了按住的技能键。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertTrue(e.request_buff_window(10))
        self.assertFalse(e.request_buff_window(10))   # 已经松开，再请求返回 False


class GateTests(AoeHoldTestCase):
    def test_no_retrigger_within_cooldown_gate(self):
        """松手后间隔门内：怪还多也不重新按住，走普通轮转。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += AOE_HOLD_SECONDS + 0.05
        e._attack(self.ctx(m1, m2), m1)            # 满时长松手
        self.assertIsNone(e._skill_hold_key)
        e.executor.clear()
        self.clock.t += 0.7                        # 间隔门(3s)内
        e._attack(self.ctx(m1, m2), m1)
        self.assertIsNone(e._skill_hold_key)
        self.assertIn("a", e.executor.keys())      # 普通轮转（技能1点按）

    def test_retrigger_after_gate(self):
        """间隔门过后、怪还多 → 再次按住。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += AOE_HOLD_SECONDS + 0.05
        e._attack(self.ctx(m1, m2), m1)
        self.clock.t += 3.1                        # 门(3s)过
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")

    def test_stop_releases_skill_key(self):
        """运行中停止：release_keys 松掉按住的技能键。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertIn("space", e.executor.held)
        e.release_keys()
        self.assertIsNone(e._skill_hold_key)
        self.assertFalse(e.executor.held)

    def test_no_monster_cleanup_releases(self):
        """画面没怪清锁路径：按住的技能键同步松开。"""
        e = self.engine()
        m1, m2 = monster(1300), monster(1350)
        e._target_monster = m1
        e._attack(self.ctx(m1, m2), m1)
        self.assertEqual(e._skill_hold_key, "space")
        e._skill_hold_release("测试清锁")
        self.assertIsNone(e._skill_hold_key)


if __name__ == "__main__":
    unittest.main()
