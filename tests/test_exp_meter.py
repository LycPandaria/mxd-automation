"""经验统计：界面显示「近 10 分钟经验（万）」的窗口计算与格式。

不碰 OCR，只直接喂 ExpTracker/ExpMeter 读数，验证：
  1. 还没有可信读数时返回占位符；
  2. 运行超过 10 分钟后，只统计最近 10 分钟（而不是开局以来的总量）；
  3. 界面文案用「万」为单位。
"""
import unittest

from src.perception.exp_meter import ExpMeter, ExpTracker


def feed_readings(tracker, count: int, step: float = 5.0,
                  exp: int = 100000, pct: float = 50.0,
                  gain: int = 1000, pct_step: float = 0.5):
    """按固定节奏喂读数：每次 exp +gain、pct +pct_step（同一级内总经验不变）。"""
    for i in range(count):
        tracker.feed(i * step, exp, pct)
        exp += gain
        pct += pct_step


class ExpTrackerTests(unittest.TestCase):
    def test_recent_gained_none_before_first_reading(self):
        t = ExpTracker()
        t.reset(0.0)
        self.assertIsNone(t.recent_gained())

    def test_recent_gained_keeps_only_last_10_minutes(self):
        t = ExpTracker()
        t.reset(0.0)
        # 15 分钟，每 5 秒 +1000 经验 → 最后 10 分钟应为 120000
        feed_readings(t, 181)
        self.assertEqual(t.recent_gained(), 120000)

    def test_recent_gained_equals_total_when_run_under_10_minutes(self):
        t = ExpTracker()
        t.reset(0.0)
        feed_readings(t, 61)  # 5 分钟
        self.assertEqual(t.recent_gained(), t.gained)


class ExpSummaryTests(unittest.TestCase):
    def test_summary_placeholder_then_wan(self):
        m = ExpMeter()
        m.start(0.0)
        self.assertEqual(m.summary(), "近10分钟经验: -")

        feed_readings(m.tracker, 13)  # 1 分钟，累计 +11000
        self.assertEqual(m.tracker.recent_gained(), 11000)
        self.assertEqual(m.summary(), "近10分钟经验: 1.10万")


if __name__ == "__main__":
    unittest.main()
