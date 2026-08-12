"""单元测试：血条/蓝条检测器。

验证 ``detect_bar_ratio`` 在不同场景下的准确率：
  - 满血、残血、空血
  - 不同亮度/对比度
  - 区域超出边界
  - 无颜色配置时的降级行为
"""
import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.hp_mp_detector import detect_bar_ratio, detect_region_color


class TestHpMpDetector(unittest.TestCase):
    """血蓝条检测测试（占位，待补充真实图像测试用例）。"""

    def test_empty_region(self):
        """空区域应返回 None。"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detect_bar_ratio(frame, None, (255, 0, 0))
        self.assertIsNone(result)

    def test_invalid_region(self):
        """零宽高区域应返回 None。"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detect_bar_ratio(frame, [10, 10, 0, 0], (255, 0, 0))
        self.assertIsNone(result)

    def test_detect_region_color_empty(self):
        """空帧应返回 None。"""
        result = detect_region_color(None, [0, 0, 10, 10])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
