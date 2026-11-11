"""定时结束游戏：时间计算、登录界面识别、退出流程（模拟游戏，不需要真窗口）。

同时覆盖 1366×768 的两种 UI：等比缩放的按钮，以及固定像素的按钮/菜单
（固定像素正是 2026-11-04 换频道失败的原因，退出游戏用的是同一套菜单识别）。
"""
import importlib.util
import tempfile
import time as real_time
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

# src.execution 的 __init__ 依赖 Windows，这里只按文件加载
_spec = importlib.util.spec_from_file_location(
    "channel_switch", Path(__file__).resolve().parents[1] / "src" / "execution" / "channel_switch.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

CYAN = (188, 163, 86)
BG = (170, 140, 90)


def _hud(f, x, y):
    cv2.rectangle(f, (x, y), (x + 67, y + 24), CYAN, -1)
    cv2.rectangle(f, (x + 90, y), (x + 157, y + 23), CYAN, -1)


def _menu(f, x, y, rh):
    for i in range(6):
        yy = y + i * 29
        cv2.rectangle(f, (x, yy), (x + 90, yy + rh), CYAN, -1)


def build(h=1080, fixed_ui=False, menu=False):
    """游戏画面。fixed_ui=True 时 768 高度仍用固定像素的按钮/菜单。"""
    if h == 1080:
        f = np.full((1080, 1920, 3), BG, np.uint8)
        _hud(f, 1460, 1044)
        if menu:
            _menu(f, 1455, 855, 25)
        return f
    if fixed_ui:
        w = 1366
        f = np.full((h, w, 3), BG, np.uint8)
        _hud(f, w - 120, h - 30)
        if menu:
            _menu(f, 900, 300, 27)  # rectangle 右下角含端点 → 实际 28px 高
        return f
    return cv2.resize(build(1080, False, menu),
                      (round(1920 * h / 1080), h), interpolation=cv2.INTER_AREA)


def login_screen(h=1080, fixed_ui=False):
    """登录界面：米黄色书页、棕色木牌，没有底部状态栏。"""
    if fixed_ui and h != 1080:
        f = np.full((h, 1366, 3), (170, 215, 235), np.uint8)
        cv2.rectangle(f, (700, int(h * 0.36)), (960, int(h * 0.60)), (40, 80, 140), -1)
        return f
    base = np.full((1080, 1920, 3), (170, 215, 235), np.uint8)
    cv2.rectangle(base, (970, 390), (1350, 650), (40, 80, 140), -1)
    return base if h == 1080 else cv2.resize(base, (round(1920 * h / 1080), h),
                                            interpolation=cv2.INTER_AREA)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    @staticmethod
    def strftime(fmt):
        return real_time.strftime(fmt)


class QuitGame:
    """模拟退出游戏：state = normal / menu / confirm / login / closed。"""

    def __init__(self, clock, h=1080, fixed_ui=False, needs_enter=False,
                 esc_works=True, click_works=True, leave_delay=0.5, closes_window=False):
        self.clock, self.h, self.fixed_ui = clock, h, fixed_ui
        self.needs_enter, self.esc_works = needs_enter, esc_works
        self.click_works, self.leave_delay = click_works, leave_delay
        self.closes_window = closes_window
        self.state, self.leave_at = "normal", None
        self.keys, self.clicks = [], []
        if h == 1080:
            self.menu_x, self.menu_y, self.click_scale = 1455, 855, 1.0
        elif fixed_ui:
            self.menu_x, self.menu_y, self.click_scale = 900, 300, 1.0
        else:
            self.click_scale = h / 1080
            self.menu_x, self.menu_y = 1455 * self.click_scale, 855 * self.click_scale

    def grab(self):
        if self.leave_at is not None and self.clock.now >= self.leave_at:
            self.state, self.leave_at = "closed" if self.closes_window else "login", None
        if self.state == "closed":
            raise RuntimeError("客户区尺寸无效")
        if self.state == "login":
            return login_screen(self.h, self.fixed_ui)
        return build(self.h, self.fixed_ui, menu=(self.state == "menu"))

    def press_key(self, key, *a, **k):
        self.keys.append(key)
        if key == "esc" and self.esc_works:
            self.state = {"normal": "menu", "menu": "normal", "confirm": "menu"}.get(
                self.state, self.state)
        elif key == "enter" and self.state == "confirm":
            self.state, self.leave_at = "normal", self.clock.now + self.leave_delay
        elif key == "enter" and self.state == "login":
            self.state = "normal"  # 登录界面按回车会重新登录：程序不能乱按
        return True

    def real_click(self, x, y):
        self.clicks.append((x, y))
        qy = self.menu_y + cs.MENU_ROW_GAP * cs.MENU_QUIT_ROW * self.click_scale
        if self.click_works and self.state == "menu" \
                and self.menu_x <= x <= self.menu_x + 90 * self.click_scale \
                and abs(y - qy) <= 20 * self.click_scale:
            if self.needs_enter:
                self.state = "confirm"
            else:
                self.state, self.leave_at = "normal", self.clock.now + self.leave_delay
        return True

    def key_down(self, key):
        return True

    def key_up(self, key):
        return True


class NextQuitTests(unittest.TestCase):
    def at(self, hour, minute):
        return real_time.mktime((2026, 11, 5, hour, minute, 0, 0, 0, -1))

    def test_later_today(self):
        self.assertEqual(cs.next_quit_at("23:30", self.at(22, 0)), self.at(23, 30))

    def test_already_past_means_tomorrow(self):
        got = real_time.localtime(cs.next_quit_at("06:30", self.at(23, 0)))
        self.assertEqual((got.tm_mday, got.tm_hour, got.tm_min), (6, 6, 30))

    def test_exactly_now_means_tomorrow(self):
        self.assertEqual(real_time.localtime(cs.next_quit_at("06:30", self.at(6, 30))).tm_mday, 6)

    def test_bad_values(self):
        for bad in ("", "abc", "25:00", "06:99", "6", 360, None):
            self.assertIsNone(cs.next_quit_at(bad, self.at(12, 0)), bad)


class InGameTests(unittest.TestCase):
    def test_game_screen_with_and_without_menu(self):
        self.assertTrue(cs.in_game(build()))
        self.assertTrue(cs.in_game(build(menu=True)))

    def test_login_and_black_are_not_the_game(self):
        self.assertFalse(cs.in_game(login_screen()))
        self.assertFalse(cs.in_game(np.zeros((1080, 1920, 3), np.uint8)))

    def test_scaled_ui_at_768(self):
        self.assertTrue(cs.in_game(build(768)))
        self.assertFalse(cs.in_game(login_screen(768)))

    def test_fixed_pixel_ui_at_768(self):
        self.assertTrue(cs.in_game(build(768, fixed_ui=True)))
        self.assertFalse(cs.in_game(login_screen(768, fixed_ui=True)))


class QuitTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self._time, cs.time = cs.time, self.clock
        self.tmp = tempfile.TemporaryDirectory()
        self._dir, cs.QUIT_DEBUG_DIR = cs.QUIT_DEBUG_DIR, self.tmp.name
        self.logs = []

    def tearDown(self):
        cs.time = self._time
        cs.QUIT_DEBUG_DIR = self._dir
        self.tmp.cleanup()

    def quitter(self, game):
        return cs.GameQuitter(game, game, SimpleNamespace(), on_log=self.logs.append)

    def test_esc_then_quit_row_stops_at_the_login_screen(self):
        g = QuitGame(self.clock)
        self.assertTrue(self.quitter(g).run())
        self.assertEqual(g.keys, ["esc"])          # 到了登录界面不再按任何键
        self.assertEqual(len(g.clicks), 1)
        self.assertEqual(g.state, "login")

    def test_window_closing_also_counts(self):
        g = QuitGame(self.clock, closes_window=True)
        self.assertTrue(self.quitter(g).run())
        self.assertEqual(g.keys, ["esc"])

    def test_fixed_pixel_ui_at_768_clicks_the_right_row(self):
        g = QuitGame(self.clock, h=768, fixed_ui=True)
        self.assertTrue(self.quitter(g).run())
        self.assertEqual(g.state, "login")
        # 行距按固定像素算（29×5），不是按 0.711 缩放；
        # 首行中心 ≈ menu_y+14，第 6 行再往下 145px
        self.assertAlmostEqual(g.clicks[0][1], g.menu_y + 14 + 29 * 5, delta=3)

    def test_scaled_ui_at_768(self):
        g = QuitGame(self.clock, h=768)
        self.assertTrue(self.quitter(g).run())
        self.assertEqual(g.state, "login")

    def test_confirm_dialog_gets_enter(self):
        g = QuitGame(self.clock, needs_enter=True)
        self.assertTrue(self.quitter(g).run())
        self.assertEqual(g.keys, ["esc", "enter"])
        self.assertEqual(g.state, "login")

    def test_slow_exit_gets_no_enter(self):
        g = QuitGame(self.clock, leave_delay=cs.QUIT_LEAVE_TIMEOUT - 1.0)
        self.assertTrue(self.quitter(g).run())
        self.assertNotIn("enter", g.keys)

    def test_already_at_the_login_screen_presses_nothing(self):
        g = QuitGame(self.clock)
        g.state = "login"
        self.assertTrue(self.quitter(g).run())
        self.assertEqual((g.keys, g.clicks), ([], []))

    def test_menu_already_open_is_not_closed_by_esc(self):
        g = QuitGame(self.clock)
        g.state = "menu"
        self.assertTrue(self.quitter(g).run())
        self.assertEqual(g.keys, [])

    def test_never_clicks_when_the_menu_does_not_open(self):
        g = QuitGame(self.clock, esc_works=False)
        self.assertFalse(self.quitter(g).run())
        self.assertEqual(g.clicks, [])
        self.assertEqual(g.keys.count("esc"), cs.ATTEMPTS)
        self.assertTrue(any("还在游戏里" in m for m in self.logs))

    def test_click_that_misses_is_retried_without_enter(self):
        g = QuitGame(self.clock, click_works=False)
        self.assertFalse(self.quitter(g).run())
        self.assertEqual(len(g.clicks), cs.ATTEMPTS)
        self.assertNotIn("enter", g.keys)
        self.assertEqual(g.keys, ["esc"])          # 菜单一直开着，不用再按 Esc

    def test_abort_touches_nothing(self):
        g = QuitGame(self.clock)
        q = self.quitter(g)
        q.abort = lambda: True
        self.assertFalse(q.run())
        self.assertTrue(q.aborted)
        self.assertEqual((g.keys, g.clicks), ([], []))

    def test_abort_leaves_a_log_line(self):
        """中途停下要在日志里留痕（对齐换频道：事后能看出退出被挂起、会接着退）。"""
        g = QuitGame(self.clock)
        q = self.quitter(g)
        q.abort = lambda: True
        q.run()
        self.assertTrue(any("中途停下" in m and "挂起" in m for m in self.logs))

    def test_run_resets_aborted_flag(self):
        """上一次 abort 留下的标志，下一次 run 开头要清掉（_do_quit 依赖它判断本次）。"""
        g = QuitGame(self.clock)
        q = self.quitter(g)
        q.aborted = True
        q.abort = lambda: False
        self.assertTrue(q.run())
        self.assertFalse(q.aborted)


if __name__ == "__main__":
    unittest.main()
