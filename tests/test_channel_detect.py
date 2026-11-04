"""换频道的界面识别：菜单 / 频道窗口在"等比缩放"和"固定像素"两种 UI 下都能认出来。

背景：2026-11-04 实机换频道失败，日志卡在"按 Esc 后认不出菜单"。
常量是按 1920×1080 量的；怀旧服这类客户端的 UI 可能是固定像素（换分辨率不缩放），
按画面高度乘 0.711 去量就对不上。find_menu/find_dialog 现在两种都试。
"""
import unittest

import numpy as np

from src.execution.channel_switch import (
    find_menu, find_dialog, _scales, _cyan_mask,
    MENU_ROW, MENU_ROW_GAP, MENU_MIN_ROWS, GRID, CELL_OFFSET, CELL_GAP, REF_H,
)

CYAN = (200, 170, 60)   # BGR，满足 _cyan_mask


def blank(h=1080, w=1920):
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_menu(frame, s=1.0, x=200, y=200):
    w, h = max(1, int(MENU_ROW[0] * s)), max(1, int(MENU_ROW[1] * s))
    pitch = max(h + 1, int(MENU_ROW_GAP * s))
    for i in range(MENU_MIN_ROWS + 1):
        yy = y + i * pitch
        frame[yy:yy + h, x:x + w] = CYAN
    return x + w // 2, y + h // 2


def draw_dialog(frame, s=1.0, x=300, y=150):
    bar_w, bar_h = int(432 * s), int(22 * s)
    frame[y:y + bar_h, x:x + bar_w] = CYAN
    for i in range(GRID[0] * GRID[1]):
        row, col = divmod(i, GRID[1])
        cx = x + int((CELL_OFFSET[0] + col * CELL_GAP[0]) * s)
        cy = y + int((CELL_OFFSET[1] + row * CELL_GAP[1]) * s)
        # 每个格子左端一小块青色（find_dialog 就采这里）
        frame[max(0, cy - 2):cy + 3,
              max(0, cx - int(26 * s)):max(1, cx - int(22 * s))] = CYAN
    return x + bar_w // 2, y + bar_h // 2


class ScaleTests(unittest.TestCase):
    def test_scales_include_fixed_and_ratio(self):
        self.assertEqual(_scales(blank(768, 1366)), [768 / REF_H, 1.0])
        self.assertEqual(_scales(blank(1080, 1920)), [1.0])

    def test_cyan_mask_matches_menu_color(self):
        f = blank(100, 100)
        f[10:20, 10:20] = CYAN
        self.assertGreater(int(_cyan_mask(f).sum()), 0)


class MenuDetectTests(unittest.TestCase):
    def test_menu_scaled_with_frame(self):
        f = blank(1080, 1920)
        c = draw_menu(f, 1.0)
        self.assertEqual(find_menu(f), c)

    def test_menu_fixed_size_on_768(self):
        """1366×768 + 固定像素 UI：菜单行仍是 91×28（新行为）。"""
        f = blank(768, 1366)
        c = draw_menu(f, 1.0)
        self.assertEqual(find_menu(f), c)

    def test_menu_scaled_on_768(self):
        """1366×768 + 等比缩放 UI：菜单行 65×20（原有行为）。"""
        f = blank(768, 1366)
        c = draw_menu(f, 768 / REF_H)
        self.assertEqual(find_menu(f), c)

    def test_no_menu_returns_none(self):
        self.assertIsNone(find_menu(blank(768, 1366)))


class DialogDetectTests(unittest.TestCase):
    def test_dialog_fixed_size_on_768(self):
        f = blank(768, 1366)
        draw_dialog(f, 1.0)
        d = find_dialog(f)
        self.assertIsNotNone(d)
        self.assertEqual(d.current, GRID[0] * GRID[1] - 1)
        # 点击坐标按固定像素算，落在画面内
        x, y = d.cell(0)
        self.assertTrue(0 < x < 1366 and 0 < y < 768)

    def test_dialog_scaled_on_768(self):
        f = blank(768, 1366)
        draw_dialog(f, 768 / REF_H)
        self.assertIsNotNone(find_dialog(f))

    def test_no_dialog_returns_none(self):
        self.assertIsNone(find_dialog(blank(768, 1366)))


if __name__ == "__main__":
    unittest.main()
