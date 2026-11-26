"""边界门：上一步离画板四边都很远时，外圈 60px 的高分峰不许赢。

20261120_023051 等四局都是淡出目标在画板内部、程序位置一步跳到右下角 (694,448)；
这里用平画板 + 人为两个峰复现：关掉门会跳过去，开着门留在原处。
"""
import unittest
from unittest import mock

import numpy as np

from src.perception import lie_shadow

BOARD = np.full((500, 750, 3), 128, np.uint8)


def make_finder():
    f = lie_shadow.ShapeFinder(lie_shadow.TargetFinder(None, "single"))
    f.shape.template = np.zeros((40, 40), np.uint8)
    f.shape.template[10:30, 10:30] = 255
    f.shape.taken_at = 0.
    f.pos = np.array([200., 200.])
    f.score = np.zeros((333, 500), np.float32)  # 半分辨率分数图（750×500 / 1.5）
    f.score[133, 133] = 60.                     # 当前位置附近（全分辨率 200,200）
    f.score[299, 463] = 80.                     # 右下角诱饵（全分辨率 694,448）
    return f


class BorderGateTest(unittest.TestCase):
    def test_border_peak_blocked_when_prev_interior(self):
        new = make_finder().step(BOARD, None, 1.)
        self.assertLess(float(np.linalg.norm(new - [200., 200.])), 40.)

    def test_gate_is_per_edge(self):
        """目标贴着下边时，右下角的诱饵在右边外圈，仍然要拦（不能因为贴下边就整圈放行）。"""
        f = make_finder()
        f.pos = np.array([200., 450.])
        f.score[:] = 0
        f.score[300, 267] = 60.  # 下边附近的近峰（全分辨率 400,450）
        f.score[299, 463] = 80.  # 右下角（全分辨率 694,448）
        new = f.step(BOARD, None, 1.)
        self.assertLess(float(np.linalg.norm(new - [400., 450.])), 40.)

    def test_without_gate_it_jumps(self):
        """证明边界门真的在起作用（把扣分关掉就会跳到右下角）。"""
        with mock.patch.object(lie_shadow, "BORDER_GATE_PENALTY", 0.):
            new = make_finder().step(BOARD, None, 1.)
        self.assertGreater(float(np.linalg.norm(new - [200., 200.])), 300.)


if __name__ == "__main__":
    unittest.main()
